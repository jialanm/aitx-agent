"""Tests for question loading and scoring, using the committed validator set."""

import json
from pathlib import Path

import pytest

from src.evaluation import (
    REFERENCE_FIELD,
    file_sha256,
    is_exact_match,
    load_questions,
    reference_answers,
)
from src.schemas import TaskInput

DATA_DIR = Path(__file__).parent.parent / "data"
VALIDATOR = DATA_DIR / "phase1_validator.json"
MANIFEST = DATA_DIR / "phase1_validator.manifest.json"


class TestValidatorFile:
    def test_matches_manifest_checksum(self):
        manifest = json.loads(MANIFEST.read_text())
        assert file_sha256(VALIDATOR) == manifest["sha256"]
        assert len(load_questions(VALIDATOR)) == manifest["n_records"]

    def test_every_record_has_a_reference_answer(self):
        questions = load_questions(VALIDATOR)
        refs = reference_answers(questions)
        assert set(refs) == {q["id"] for q in questions}
        assert all(refs.values())

    def test_records_parse_as_task_input_despite_extra_fields(self):
        # The challenge records carry answer_expected, citations, etc. alongside
        # the input fields. The agent must accept them without modification.
        for q in load_questions(VALIDATOR):
            task = TaskInput(**q)
            assert task.id == q["id"]
            assert task.question.answer_format in {
                "binary", "multiple_choice", "numeric_match", "string_match"
            }


class TestReferenceAnswers:
    def test_skips_records_without_reference(self):
        questions = [
            {"id": "A", REFERENCE_FIELD: "Yes"},
            {"id": "B"},
        ]
        assert reference_answers(questions) == {"A": "Yes"}

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
