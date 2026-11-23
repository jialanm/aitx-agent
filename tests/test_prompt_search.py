"""The dev-search runner's formats, label parsing, metrics and shot picking, without a model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "hw12"))
from prompt_search import FORMATS, build_messages, metrics, parse_label, pick_shots  # noqa: E402

PAIR = {
    "topic_id": "2021-23", "nct_id": "NCT03190759", "label": 1, "label_word": "excluded",
    "patient_text": "A 39-year-old man with cough and shortness of breath.",
    "trial_text": ("Title: Asthma study\nConditions: Bronchial Asthma\nInterventions: not stated\n"
                   "Sex: All | Minimum age: 5 Years | Maximum age: 17 Years | Healthy volunteers: No\n"
                   "Eligibility criteria:\nInclusion Criteria:\n* Age 5 to 17\n* Asthma exacerbation\n\n"
                   "Exclusion Criteria:\n* Pneumonia\n* Prior intubation"),
}


def test_criteria_format_keeps_title_and_criteria_only():
    text = FORMATS["criteria"](PAIR)
    assert text.startswith("Title: Asthma study\nEligibility criteria:")
    assert "Conditions:" not in text and "Prior intubation" in text


def test_split_format_numbers_inclusion_and_exclusion_items():
    text = FORMATS["split"](PAIR)
    assert "I1. Age 5 to 17" in text and "I2. Asthma exacerbation" in text
    assert "E1. Pneumonia" in text and "E2. Prior intubation" in text
    assert "Sex: All" in text  # the population line is kept


def test_messages_carry_shots_as_answered_examples():
    shot = dict(PAIR, label=2)
    msgs = build_messages(PAIR, "full", "words", [shot])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "eligible"
    assert "exactly one of: not relevant, excluded, or eligible" in msgs[0]["content"]


def test_parse_label_reads_each_verbalizer_and_rejects_junk():
    assert parse_label("Excluded.", "words") == 1
    assert parse_label("not relevant", "words") == 0
    assert parse_label("2", "digits") == 2
    assert parse_label("B", "letters") == 1
    assert parse_label("eligible", "digits") == 2  # a word answer to a digit prompt is still read
    assert parse_label("I think the patient", "words") is None


def test_metrics_report_false_eligible_rate_over_non_eligible_gold():
    golds = [0, 1, 2, 1, 0, 2]
    preds = [0, 2, 2, 1, 2, None]
    m = metrics(golds, preds)
    assert m["n"] == 6 and m["accuracy"] == round(3 / 6, 4)
    assert m["false_eligible_rate"] == 0.5  # 2 of the 4 non-eligible pairs were called eligible
    assert m["unparsed"] == 1


def test_shots_rotate_through_labels_from_the_train_split():
    train = [dict(PAIR, label=c, nct_id=f"NCT{c}{i}") for c in (0, 1, 2) for i in range(3)]
    shots = pick_shots(train, 3, seed=1)
    assert sorted(s["label"] for s in shots) == [0, 1, 2]
    assert pick_shots(train, 0, seed=1) == []
