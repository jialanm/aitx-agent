"""The LoRA script's example builder masks the prompt and keeps the answer, truncating from the front."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "hw12"))
from finetune_lora import IGNORE, build_example  # noqa: E402


def test_prompt_tokens_are_masked_and_answer_tokens_trained():
    ex = build_example(prompt_ids=[1, 2, 3, 4], answer_ids=[9, 8], max_length=16)

    assert ex["input_ids"] == [1, 2, 3, 4, 9, 8]
    assert ex["labels"] == [IGNORE, IGNORE, IGNORE, IGNORE, 9, 8]
    assert ex["attention_mask"] == [1] * 6
    assert ex["truncated"] is False


def test_long_prompt_is_cut_from_the_front_so_the_question_and_answer_survive():
    ex = build_example(prompt_ids=list(range(100)), answer_ids=[9, 8], max_length=10)

    assert ex["input_ids"] == list(range(92, 100)) + [9, 8]
    assert ex["labels"][:8] == [IGNORE] * 8 and ex["labels"][8:] == [9, 8]
    assert ex["truncated"] is True


def test_answer_that_cannot_fit_is_dropped():
    assert build_example(prompt_ids=[1], answer_ids=list(range(20)), max_length=10) is None
