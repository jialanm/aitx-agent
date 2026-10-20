"""Qwen3-8B model loading, generation, and tool-call parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

MODEL_ID = "Qwen/Qwen3-8B"

# Precision modes and their labels, recorded in eval run metadata. "fp8"
# loads a checkpoint that was already quantised by its publisher (for
# example Qwen/Qwen3-32B-FP8); the other two quantise or cast at load time.
PRECISIONS = {
    "bf16": "bfloat16",
    "nf4": "nf4 4-bit (bitsandbytes, double quant)",
    "fp8": "fp8 (publisher's pre-quantised checkpoint)",
}

# Decoding settings, set once per run by the eval script and recorded in
# its metadata. Greedy everywhere is the default and what runs 1-5 used.
# "qwen_thinking" samples the thinking phase with the settings Qwen
# publishes for thinking mode and keeps the answer phase greedy, so the
# reasoning can explore while the final words stay the most probable ones.
# A thinking budget caps the thinking phase; when it is hit the model is
# told to answer from what it has, in Qwen's own wording for that case.
DECODING_PRESETS = {
    "greedy": None,
    "qwen_thinking": {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
}
THINKING_SAMPLING: dict | None = None
THINKING_BUDGET: int | None = None
BUDGET_EXHAUSTED_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the solution "
    "based on the thinking directly now.\n</think>\n\n"
)


def set_decoding(preset: str = "greedy", thinking_budget: int | None = None, seed: int | None = None) -> str:
    """Choose the decoding for this process and return its description for the run metadata."""
    global THINKING_SAMPLING, THINKING_BUDGET
    if preset not in DECODING_PRESETS:
        raise ValueError(f"unknown decoding preset {preset!r}; choose from {sorted(DECODING_PRESETS)}")
    THINKING_SAMPLING = DECODING_PRESETS[preset]
    THINKING_BUDGET = thinking_budget
    if seed is not None:
        torch.manual_seed(seed)
    parts = ["greedy (do_sample=False)" if THINKING_SAMPLING is None
             else "thinking sampled (temperature 0.6, top_p 0.95, top_k 20), answer greedy"]
    if thinking_budget:
        parts.append(f"thinking budget {thinking_budget} tokens")
    if seed is not None and THINKING_SAMPLING is not None:
        parts.append(f"seed {seed}")
    return "; ".join(parts)


# Regex for extracting tool calls from model output
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>(?:(?!</think>).)*$", re.DOTALL)


@dataclass
class ToolCall:
    """A parsed tool call from model output."""

    name: str
    arguments: dict


@dataclass
class ModelResponse:
    """Parsed model response containing text and/or tool calls."""

    text: str  # Final text after stripping tool calls and thinking
    tool_calls: list[ToolCall]
    raw: str  # Full raw output
    thinking: str  # Extracted thinking content
    thinking_truncated: bool = False  # the thinking budget was hit and the model was told to answer


class RepetitionStoppingCriteria(StoppingCriteria):
    """Stop generation when the model enters a repetitive loop.

    Checks every `check_interval` tokens whether a substring of length
    `window_size` has appeared `max_repeats` or more times in the output.
    """

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        window_size: int = 100,
        max_repeats: int = 3,
        check_interval: int = 64,
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.window_size = window_size
        self.max_repeats = max_repeats
        self.check_interval = check_interval
        self._step = 0
        self.stopped_early = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self._step += 1
        if self._step % self.check_interval != 0:
            return False

        # Decode only the new tokens
        new_tokens = input_ids[0][self.prompt_length:]
        if len(new_tokens) < self.window_size * 2:
            return False

        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        if len(text) < self.window_size * 2:
            return False

        # Check if the last `window_size` chars appear earlier in the output
        tail = text[-self.window_size:]
        # Count non-overlapping occurrences in the full text
        count = text.count(tail)
        if count >= self.max_repeats:
            self.stopped_early = True
            return True

        return False


def load_model(
    model_id: str = MODEL_ID,
    precision: str = "bf16",
) -> tuple:
    """Load the model at the given precision: bf16, nf4, or fp8.

    bf16 is the default so that eval numbers reflect the model itself, not
    rounding from quantization. nf4 quantises at load time for hardware
    that cannot hold the full weights. fp8 expects a checkpoint the
    publisher already quantised and refuses to run if the loaded model
    carries no quantization config, so the precision recorded for a run is
    the precision that actually ran.

    Returns:
        (model, tokenizer) tuple
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; choose from {sorted(PRECISIONS)}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if precision == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["quantization_config"] = bnb_config
    elif precision == "bf16":
        kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if precision == "fp8" and getattr(model.config, "quantization_config", None) is None:
        raise ValueError(
            f"{model_id} carries no quantization config, so it is not an fp8 checkpoint; "
            "pass the publisher's FP8 release or choose bf16/nf4"
        )
    model.eval()

    return model, tokenizer


def generate(
    model,
    tokenizer,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_new_tokens: int = 2048,
    enable_thinking: bool = True,
    stop_on_repetition: bool = True,
) -> ModelResponse:
    """Generate a response from the model.

    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        messages: List of chat messages in OpenAI format.
        tools: List of tool schemas (OpenAI format).
        max_new_tokens: Maximum tokens to generate.
        enable_thinking: Whether to enable Qwen3's thinking mode.
        stop_on_repetition: Stop when a 100-character window recurs three
            times. Right for prose, wrong for a JSON array of records, whose
            repeated field names and source URLs trip it: the report chain's
            goal-2 reply was cut mid-record twice (2026-10-20).

    Returns:
        Parsed ModelResponse.
    """
    # Try using apply_chat_template with tools parameter
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Fallback: apply_chat_template doesn't support tools/enable_thinking
        text = _build_chat_text_fallback(messages, tools, tokenizer)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    two_phase = enable_thinking and (THINKING_SAMPLING is not None or THINKING_BUDGET is not None)
    if not two_phase:
        new_tokens = _generate_tokens(model, tokenizer, inputs["input_ids"], max_new_tokens, prompt_length,
                                      stop_on_repetition=stop_on_repetition)
        return parse_response(tokenizer.decode(new_tokens, skip_special_tokens=True))

    # Phase 1, the thinking: sampled if a preset says so, stopped at </think>
    # or at the budget. Phase 2, the answer: always greedy, continuing from
    # the thinking (closed by the model, or closed for it when the budget ran
    # out, with the sentence Qwen uses for that).
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    budget = min(THINKING_BUDGET or max_new_tokens, max_new_tokens)
    thinking_ids = _generate_tokens(
        model, tokenizer, inputs["input_ids"], budget, prompt_length,
        sampling=THINKING_SAMPLING, stop_ids=[think_end], stop_on_repetition=stop_on_repetition,
    )
    truncated = len(thinking_ids) == 0 or int(thinking_ids[-1]) != think_end
    if truncated:
        closing = tokenizer(BUDGET_EXHAUSTED_TEXT, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(thinking_ids.device)
        thinking_ids = torch.cat([thinking_ids, closing])
    ids = torch.cat([inputs["input_ids"][0], thinking_ids]).unsqueeze(0)
    answer_budget = max(256, max_new_tokens - len(thinking_ids))
    answer_ids = _generate_tokens(model, tokenizer, ids, answer_budget, ids.shape[1],
                                  stop_on_repetition=stop_on_repetition)
    raw_output = tokenizer.decode(torch.cat([thinking_ids, answer_ids]), skip_special_tokens=True)
    response = parse_response(raw_output)
    response.thinking_truncated = truncated
    return response


def _generate_tokens(model, tokenizer, input_ids, max_new_tokens: int, prompt_length: int,
                     sampling: dict | None = None, stop_ids: list[int] | None = None,
                     stop_on_repetition: bool = True):
    """One generate call; returns only the new tokens. Greedy unless sampling settings are given."""
    criteria = [RepetitionStoppingCriteria(tokenizer=tokenizer, prompt_length=prompt_length)] if stop_on_repetition else []
    kwargs = dict(sampling) if sampling else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    if stop_ids:
        eos = tokenizer.eos_token_id
        kwargs["eos_token_id"] = list(dict.fromkeys(([eos] if isinstance(eos, int) else list(eos or [])) + stop_ids))
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            stopping_criteria=StoppingCriteriaList(criteria),
            **kwargs,
        )
    return outputs[0][prompt_length:]


def parse_response(raw_output: str) -> ModelResponse:
    """Parse raw model output into structured response."""
    # Extract thinking blocks (closed)
    thinking_parts = []
    for m in _THINK_RE.finditer(raw_output):
        content = m.group(0)[7:-8].strip()  # strip <think></think>
        thinking_parts.append(content)

    # Also capture unclosed <think> blocks (hit max_new_tokens mid-think)
    unclosed = _THINK_UNCLOSED_RE.search(raw_output)
    if unclosed:
        content = unclosed.group(0)[7:].strip()  # strip leading <think>
        thinking_parts.append(content)

    thinking = "\n".join(thinking_parts)

    # Extract tool calls
    tool_calls = []
    for m in _TOOL_CALL_RE.finditer(raw_output):
        try:
            data = json.loads(m.group(1))
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            tool_calls.append(ToolCall(name=name, arguments=arguments))
        except (json.JSONDecodeError, TypeError):
            continue

    # Extract final text (remove thinking and tool call blocks)
    text = raw_output
    text = _THINK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)  # also strip unclosed think blocks
    text = _TOOL_CALL_RE.sub("", text)
    text = text.strip()

    return ModelResponse(
        text=text,
        tool_calls=tool_calls,
        raw=raw_output,
        thinking=thinking,
    )


def _build_chat_text_fallback(
    messages: list[dict],
    tools: list[dict] | None,
    tokenizer,
) -> str:
    """Fallback: manually construct ChatML-style text with tool definitions."""
    # First try without tools
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Manual ChatML construction
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "tool":
                # ChatML has no tool role; Qwen renders tool results as a
                # user turn carrying <tool_response> tags.
                role, content = "user", f"<tool_response>\n{content}\n</tool_response>"
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        text += "<|im_start|>assistant\n"

    # Inject tool definitions into system message if needed
    if tools and messages and messages[0]["role"] == "system":
        tool_text = "\n\n## Available Tools\n"
        tool_text += "You can call tools using <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>\n\n"
        for tool in tools:
            func = tool.get("function", {})
            tool_text += f"### {func.get('name', '')}\n"
            tool_text += f"{func.get('description', '')}\n"
            params = func.get("parameters", {}).get("properties", {})
            if params:
                tool_text += "Parameters:\n"
                required = func.get("parameters", {}).get("required", [])
                for pname, pinfo in params.items():
                    req = " (required)" if pname in required else ""
                    tool_text += f"  - {pname}: {pinfo.get('description', '')}{req}\n"
            tool_text += "\n"

        # Insert tool definitions after the system message
        sys_end = text.find("<|im_end|>")
        if sys_end != -1:
            text = text[:sys_end] + tool_text + text[sys_end:]

    return text


def format_tool_result(tool_name: str, result_text: str) -> dict:
    """A tool result as a message with the tool role.

    Qwen3's chat template keeps the thinking of every assistant turn after
    the last real user message and drops the thinking before it; a tool
    result sent as a user message counted as a real user message, so each
    lookup cut the model off from its own earlier reasoning (March 2026
    design "for compatibility"; found 2026-10-15 by rendering run 3's
    history: one of four thinking blocks survived). With the tool role the
    template wraps the result in <tool_response> tags and keeps the
    thinking across the question's tool turns.
    """
    return {
        "role": "tool",
        "content": f"[Tool Result: {tool_name}]\n{result_text}",
    }
