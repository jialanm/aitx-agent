"""Question loading, answer loading, and exact-match scoring for the harness.

Questions and answers live in separate files so the agent is never handed a
record that contains the answer. Kept free of model and network dependencies
so the rules here can be unit tested, and so a change to a rule shows up as a
test change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# The fields the agent is allowed to see. Everything else in a challenge
# record is answer material and goes to the answers file.
INPUT_FIELDS = ("id", "patient", "question")

# Field the challenge's Phase1_Model_Validator records use for the answer key.
REFERENCE_FIELD = "answer_expected"


def load_questions(path: str | Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of question records")
    return data


def split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate challenge records into agent-visible questions and answer keys.

    Every field outside INPUT_FIELDS is treated as answer material, so a new
    upstream field defaults to the answers side rather than leaking to the
    agent.
    """
    questions, answers = [], []
    for r in records:
        questions.append({k: r[k] for k in INPUT_FIELDS if k in r})
        answers.append({"id": r["id"], **{k: v for k, v in r.items() if k not in INPUT_FIELDS}})
    return questions, answers


def reference_answers(answer_records: list[dict]) -> dict[str, str]:
    """Map question id to expected answer for records that carry one.

    Records without the reference field are skipped rather than failing, so a
    question set without an answer key still runs for format validation.
    """
    refs = {}
    for r in answer_records:
        if REFERENCE_FIELD in r:
            refs[r["id"]] = str(r[REFERENCE_FIELD])
    return refs


def load_answers(path: str | Path) -> dict[str, str]:
    with open(path) as f:
        return reference_answers(json.load(f))


def is_exact_match(response: str, expected: str) -> bool:
    """Case-insensitive comparison after trimming surrounding whitespace.

    This is deliberately the only leniency. The challenge auto-scored by exact
    match; anything looser here would inflate our number relative to theirs.
    """
    return response.strip().lower() == expected.strip().lower()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
