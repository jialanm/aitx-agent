"""Two-phase decoding: sampled or budgeted thinking, greedy answer. No GPU: a
scripted model and tokenizer stand in, so what is tested is the control flow
(stop at </think>, close the thinking when the budget is hit, keep the answer
greedy) and the metadata label."""

from __future__ import annotations

import torch
import pytest

import src.model as model_module
from src.model import generate, set_decoding


class _Batch(dict):
    def to(self, device):
        return self


class _Tokenizer:
    """Whole-string tokens: 1 prompt, 2 <think>, 3 thought, 4 </think>, 5 answer, 9 budget closing."""
    eos_token_id = 0
    _text = {1: "", 2: "<think>", 3: "let me think ", 4: "</think>", 5: "Answer", 9: model_module.BUDGET_EXHAUSTED_TEXT}

    def apply_chat_template(self, messages, **kwargs):
        return "PROMPT"

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [9] if text == model_module.BUDGET_EXHAUSTED_TEXT else [1]
        return _Batch(input_ids=torch.tensor([ids]))

    def convert_tokens_to_ids(self, token):
        return 4 if token == "</think>" else -1

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._text[int(i)] for i in ids)


class _Model:
    device = "cpu"

    def __init__(self, scripts):
        self.scripts = list(scripts)   # new tokens per generate call, before the max cut
        self.calls = []                # kwargs per call

    def generate(self, input_ids, attention_mask=None, max_new_tokens=None, stopping_criteria=None, **kwargs):
        self.calls.append({**kwargs, "criteria": len(stopping_criteria or [])})
        new = self.scripts.pop(0)[:max_new_tokens]
        stop = set(kwargs.get("eos_token_id") or [])
        out = []
        for t in new:
            out.append(t)
            if t in stop:
                break
        return torch.cat([input_ids[0], torch.tensor(out, dtype=input_ids.dtype)]).unsqueeze(0)


@pytest.fixture(autouse=True)
def _greedy_after():
    yield
    set_decoding("greedy")


def test_greedy_default_is_one_call_with_sampling_off():
    set_decoding("greedy")
    model = _Model([[2, 3, 4, 5]])
    r = generate(model, _Tokenizer(), [{"role": "user", "content": "q"}])
    assert r.text == "Answer" and "let me think" in r.thinking and not r.thinking_truncated
    assert len(model.calls) == 1 and model.calls[0]["do_sample"] is False


def test_qwen_thinking_samples_the_thinking_and_decodes_the_answer_greedily():
    set_decoding("qwen_thinking", seed=1)
    model = _Model([[2, 3, 4, 5, 5], [5]])
    r = generate(model, _Tokenizer(), [{"role": "user", "content": "q"}])
    assert r.text == "Answer" and not r.thinking_truncated
    assert model.calls[0]["do_sample"] is True and model.calls[0]["temperature"] == 0.6
    assert 4 in model.calls[0]["eos_token_id"]      # phase 1 stops at </think>
    assert model.calls[1]["do_sample"] is False       # phase 2 greedy


def test_budget_closes_the_thinking_and_the_model_still_answers():
    set_decoding("greedy", thinking_budget=3)
    model = _Model([[2, 3, 3, 3, 3, 3, 4, 5], [5]])
    r = generate(model, _Tokenizer(), [{"role": "user", "content": "q"}])
    assert r.thinking_truncated
    assert "Considering the limited time" in r.thinking
    assert r.text == "Answer"


def test_thinking_off_is_never_two_phase():
    set_decoding("qwen_thinking", thinking_budget=3)
    model = _Model([[5]])
    r = generate(model, _Tokenizer(), [{"role": "user", "content": "q"}], enable_thinking=False)
    assert r.text == "Answer" and len(model.calls) == 1 and model.calls[0]["do_sample"] is False


def test_metadata_label_states_what_ran():
    assert set_decoding("greedy") == "greedy (do_sample=False)"
    assert set_decoding("qwen_thinking", 512, 7) == (
        "thinking sampled (temperature 0.6, top_p 0.95, top_k 20), answer greedy; thinking budget 512 tokens; seed 7"
    )
    with pytest.raises(ValueError):
        set_decoding("beam")


def test_repetition_criterion_can_be_switched_off():
    """A JSON array of records repeats field names and URLs, which trips the
    think-loop detector: the report chain's goal-2 reply was cut mid-record
    twice (2026-10-20), its last 100 characters recurring four times."""
    set_decoding("greedy")
    tok = _Tokenizer()
    on = _Model([[2, 3, 4, 5]])
    generate(on, tok, [{"role": "user", "content": "q"}])
    off = _Model([[2, 3, 4, 5]])
    generate(off, tok, [{"role": "user", "content": "q"}], stop_on_repetition=False)

    assert on.calls[0]["criteria"] == 1
    assert off.calls[0]["criteria"] == 0

