"""The pair sampler balances labels and caps how much one patient contributes."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "hw12"))
from build_trec_pairs import sample_split, trial_text  # noqa: E402


def _qrels():
    # Patient A has many judgments of every label; patients B and C a few.
    rows = []
    for label in (0, 1, 2):
        rows += [{"topic_id": "A", "nct_id": f"NCT{label}{i:04d}", "judgment": label} for i in range(40)]
        rows += [{"topic_id": "B", "nct_id": f"NCTB{label}{i:03d}", "judgment": label} for i in range(6)]
        rows += [{"topic_id": "C", "nct_id": f"NCTC{label}{i:03d}", "judgment": label} for i in range(6)]
    return rows


def test_labels_are_balanced_and_one_patient_is_capped():
    pairs = sample_split(_qrels(), {"A", "B", "C"}, n_per_label=12, cap=5, rng=random.Random(0))

    assert len(pairs) == 36
    for label in (0, 1, 2):
        rows = [p for p in pairs if p["judgment"] == label]
        assert len(rows) == 12
        assert max(sum(1 for r in rows if r["topic_id"] == t) for t in "ABC") <= 5


def test_patients_outside_the_split_are_never_drawn():
    pairs = sample_split(_qrels(), {"B", "C"}, n_per_label=10, cap=6, rng=random.Random(0))

    assert {p["topic_id"] for p in pairs} == {"B", "C"}


def test_too_few_pairs_under_the_cap_stops_with_a_message():
    with pytest.raises(SystemExit, match="only 15 pairs available for label 0"):
        sample_split(_qrels(), {"A", "B", "C"}, n_per_label=20, cap=5, rng=random.Random(0))


def test_trial_text_keeps_the_whole_criteria_block():
    criteria = "Inclusion Criteria:\n* Age > 18\n\nExclusion Criteria:\n* Hemoglobin < 7"
    text = trial_text({"brief_title": "T", "conditions": ["X"], "interventions": [], "gender": "All",
                       "minimum_age": "18 Years", "maximum_age": None, "healthy_volunteers": "No",
                       "eligibility_criteria": criteria})

    assert text.endswith(criteria)
    assert "Interventions: not stated" in text and "Maximum age: not stated" in text
