"""Question loading and exact-match scoring for the evaluation harness.

Kept free of model and network dependencies so scoring rules can be unit
tested, and so a change to a rule is visible as a test change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Field the challenge's Phase1_Model_Validator records use for the answer key.
REFERENCE_FIELD = "answer_expected"


def load_questions(path: str | Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of question records")
    return data


def reference_answers(questions: list[dict]) -> dict[str, str]:
    """Map question id to its expected answer, for records that carry one.

    Records without the reference field are skipped rather than failing, so a
    question set without an answer key still runs for format validation.
    """
    refs = {}
    for q in questions:
        if REFERENCE_FIELD in q:
            refs[q["id"]] = str(q[REFERENCE_FIELD])
    return refs


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
