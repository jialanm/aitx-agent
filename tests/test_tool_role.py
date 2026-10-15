"""Tool results carry the tool role, so Qwen3's template keeps the model's
thinking across a question's tool turns instead of cutting it at each result."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.model import format_tool_result


def test_tool_result_message_has_the_tool_role():
    msg = format_tool_result("search_clinvar", "one record")
    assert msg["role"] == "tool"
    assert msg["content"].startswith("[Tool Result: search_clinvar]\none record")


@pytest.mark.slow
def test_qwen_template_keeps_thinking_across_tool_turns_with_the_tool_role():
    """Needs the Qwen3-8B tokenizer (network or cache). A three-lookup history
    rendered both ways: with tool results as user messages the template keeps
    only the last turn's thinking; with the tool role it keeps all of them."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    def turn(i, final=False):
        body = f"<think>\nreasoning {i}\n</think>\n\n"
        body += "Answer" if final else '<tool_call>\n{"name": "search_clinvar", "arguments": {"gene": "CFTR"}}\n</tool_call>'
        return {"role": "assistant", "content": body}

    def history(tool_role):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}]
        for i in (1, 2, 3):
            msgs.append(turn(i))
            msgs.append({"role": "tool" if tool_role else "user", "content": f"[Tool Result: search_clinvar]\nrecord {i}"})
        msgs.append(turn(4, final=True))
        return msgs

    def rendered(msgs):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)

    assert rendered(history(tool_role=False)).count("reasoning ") == 1   # only the last turn survives
    assert rendered(history(tool_role=True)).count("reasoning ") == 4    # every turn survives


def test_harness_notices_carry_the_system_role():
    from src.agent import Agent
    msg = Agent._harness_message("You answered without looking anything up.")
    assert msg == {"role": "system", "content": "You answered without looking anything up."}


@pytest.mark.slow
def test_qwen_template_keeps_thinking_across_a_system_role_reminder():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    def turn(i, final=False):
        body = f"<think>\nreasoning {i}\n</think>\n\n"
        body += "Answer" if final else '<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>'
        return {"role": "assistant", "content": body}

    def history(role):
        return [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
                turn(1), {"role": "tool", "content": "r1"}, turn(2),
                {"role": role, "content": "You answered None; a recommended tool is uncalled."},
                turn(3), {"role": "tool", "content": "r3"}, turn(4, final=True)]

    def rendered(msgs):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)

    assert rendered(history("user")).count("reasoning ") == 2    # the reminder cut turns 1 and 2
    assert rendered(history("system")).count("reasoning ") == 4  # nothing cut

