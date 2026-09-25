"""Qwen3-8B model loading, generation, and tool-call parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

MODEL_ID = "Qwen/Qwen3-8B"

# Human-readable precision labels, recorded in eval run metadata.
PRECISION_LABELS = {False: "bfloat16", True: "nf4 4-bit (bitsandbytes, double quant)"}

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
    quantize: bool = False,
) -> tuple:
    """Load the model in bfloat16, or in 4-bit NF4 if quantize is set.

    Full precision is the default so that eval numbers reflect the model
    itself, not rounding from quantization. The 4-bit path is kept for a
    measured comparison and for hardware that cannot hold the full weights.

    Returns:
        (model, tokenizer) tuple
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    return model, tokenizer


def generate(
    model,
    tokenizer,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_new_tokens: int = 2048,
    enable_thinking: bool = True,
) -> ModelResponse:
    """Generate a response from the model.

    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        messages: List of chat messages in OpenAI format.
        tools: List of tool schemas (OpenAI format).
        max_new_tokens: Maximum tokens to generate.
        enable_thinking: Whether to enable Qwen3's thinking mode.

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

    # Detect think-loops: stop early if model repeats itself
    rep_criteria = RepetitionStoppingCriteria(
        tokenizer=tokenizer,
        prompt_length=prompt_length,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            stopping_criteria=StoppingCriteriaList([rep_criteria]),
        )

    # Decode only the new tokens
    new_tokens = outputs[0][prompt_length:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return parse_response(raw_output)


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
    """Format a tool result as a message for the conversation.

    Tries the 'tool' role first, falls back to wrapping in a user message.
    """
    # Use a user message with tool result prefix for maximum compatibility
    return {
        "role": "user",
        "content": f"[Tool Result: {tool_name}]\n{result_text}",
    }
