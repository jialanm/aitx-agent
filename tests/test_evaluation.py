"""Tests for question and answer loading, the question/answer split, and scoring."""

import json
from pathlib import Path

import pytest

from src.evaluation import (
    INPUT_FIELDS,
    REFERENCE_FIELD,
    file_sha256,
    is_exact_match,
    load_answers,
    load_questions,
    reference_answers,
    split_records,
)
from src.schemas import TaskInput

DATA_DIR = Path(__file__).parent.parent / "data"
MANIFEST = json.loads((DATA_DIR / "phase1_validator.manifest.json").read_text())
QUESTIONS = DATA_DIR / MANIFEST["questions"]["file"]
ANSWERS = DATA_DIR / MANIFEST["answers"]["file"]


class TestCommittedFiles:
    def test_match_manifest_checksums(self):
        assert file_sha256(QUESTIONS) == MANIFEST["questions"]["sha256"]
        assert file_sha256(ANSWERS) == MANIFEST["answers"]["sha256"]
        assert len(load_questions(QUESTIONS)) == MANIFEST["n_records"]

    def test_questions_file_carries_no_answer_material(self):
        # The agent reads this file. Nothing beyond the input fields may be in it.
        for q in load_questions(QUESTIONS):
            assert set(q) <= set(INPUT_FIELDS), f"{q['id']} leaks fields {set(q) - set(INPUT_FIELDS)}"

    def test_every_question_has_an_answer_and_vice_versa(self):
        ids = {q["id"] for q in load_questions(QUESTIONS)}
        refs = load_answers(ANSWERS)
        assert set(refs) == ids
        assert all(refs.values())

    def test_questions_parse_as_task_input(self):
        for q in load_questions(QUESTIONS):
            task = TaskInput(**q)
            assert task.id == q["id"]
            assert task.question.answer_format in {
                "binary", "multiple_choice", "numeric_match", "string_match"
            }


class TestSplitRecords:
    RECORD = {
        "id": "X", "patient": {"genotype": []}, "question": {"prompt": "?"},
        REFERENCE_FIELD: "Yes", "answer_explanation": "because", "citations": [],
    }

    def test_questions_side_keeps_only_input_fields(self):
        questions, _ = split_records([self.RECORD])
        assert questions == [{"id": "X", "patient": {"genotype": []}, "question": {"prompt": "?"}}]

    def test_answers_side_keeps_id_and_everything_else(self):
        _, answers = split_records([self.RECORD])
        assert answers == [{"id": "X", REFERENCE_FIELD: "Yes", "answer_explanation": "because", "citations": []}]

    def test_unknown_upstream_field_goes_to_answers_not_questions(self):
        record = {**self.RECORD, "new_hint_field": "spoiler"}
        questions, answers = split_records([record])
        assert "new_hint_field" not in questions[0]
        assert answers[0]["new_hint_field"] == "spoiler"


class TestReferenceAnswers:
    def test_skips_records_without_reference(self):
        assert reference_answers([{"id": "A", REFERENCE_FIELD: "Yes"}, {"id": "B"}]) == {"A": "Yes"}

    def test_coerces_non_string_reference(self):
        assert reference_answers([{"id": "A", REFERENCE_FIELD: 66}]) == {"A": "66"}


class TestExactMatch:
    @pytest.mark.parametrize("response,expected", [
        ("Eteplirsen", "Eteplirsen"),
        ("eteplirsen", "Eteplirsen"),
        ("  L-serine\n", "L-Serine"),
        ("ascorbic acid, desmopressin", "ascorbic acid, desmopressin"),
        ("none", "None"),
        ("0.1", "0.1"),
    ])
    def test_matches(self, response, expected):
        assert is_exact_match(response, expected)

    @pytest.mark.parametrize("response,expected", [
        ("18", "18 months"),
        ("desmopressin, ascorbic acid", "ascorbic acid, desmopressin"),
        ("0.10", "0.1"),
        ("NCT05402384.", "NCT05402384"),
        ("", "No"),
    ])
    def test_does_not_match(self, response, expected):
        assert not is_exact_match(response, expected)
