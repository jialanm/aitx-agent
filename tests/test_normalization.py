"""Tests for answer normalization."""

import pytest

from src.normalization import normalize_answer


class TestThinkBlockStripping:
    """Safety net: <think> blocks should be stripped before normalization."""

    def test_closed_think_stripped(self):
        raw = "<think>Long reasoning...</think>Yes"
        assert normalize_answer(raw, "binary") == "Yes"

    def test_unclosed_think_stripped_binary(self):
        raw = "<think>The model gets stuck reasoning about this and never closes the tag..."
        assert normalize_answer(raw, "binary") in ("Yes", "No", "")

    def test_unclosed_think_stripped_string(self):
        raw = "beta-blockers<think>Wait let me reconsider this answer because..."
        assert normalize_answer(raw, "string_match") == "beta-blockers"

    def test_think_with_answer_after(self):
        raw = "<think>Reasoning</think>sirolimus"
        assert normalize_answer(raw, "string_match") == "sirolimus"


class TestBinaryNormalization:
    def test_simple_yes(self):
        assert normalize_answer("Yes", "binary") == "Yes"

    def test_simple_no(self):
        assert normalize_answer("No", "binary") == "No"

    def test_yes_with_explanation(self):
        assert normalize_answer("Yes, the patient is eligible.", "binary") == "Yes"

    def test_no_with_explanation(self):
        assert normalize_answer("No, this drug is not indicated.", "binary") == "No"

    def test_preamble_yes(self):
        assert normalize_answer("Based on the evidence, yes.", "binary") == "Yes"

    def test_preamble_no(self):
        assert normalize_answer("Based on current guidelines, no.", "binary") == "No"

    def test_quoted_yes(self):
        assert normalize_answer('"Yes"', "binary") == "Yes"

    def test_whitespace(self):
        assert normalize_answer("  Yes  \n", "binary") == "Yes"

    def test_multiline_yes(self):
        text = "After reviewing the evidence:\nYes"
        assert normalize_answer(text, "binary") == "Yes"


class TestMultipleChoiceNormalization:
    def test_letter_match(self):
        prompt = "Which drug? (A) DrugA (B) DrugB (C) DrugC (D) DrugD"
        assert normalize_answer("(C) DrugC", "multiple_choice", prompt) == "DrugC"

    def test_letter_only(self):
        prompt = "Which drug? (A) DrugA (B) DrugB (C) DrugC (D) DrugD"
        assert normalize_answer("C", "multiple_choice", prompt) == "DrugC"

    def test_text_match(self):
        prompt = "Which drug? (A) DrugA (B) DrugB (C) Carbamazepine (D) DrugD"
        assert normalize_answer("Carbamazepine", "multiple_choice", prompt) == "Carbamazepine"

    def test_option_with_explanation(self):
        prompt = "Which drug? (A) DrugA (B) DrugB (C) DrugC (D) DrugD"
        assert normalize_answer("A) DrugA because it is effective", "multiple_choice", prompt) == "DrugA"

    def test_full_option_text(self):
        prompt = "Which gene therapy? (A) Onasemnogene abeparvovec (Zolgensma) (B) Nusinersen (Spinraza) (C) Risdiplam (Evrysdi) (D) Valoctocogene roxaparvovec (Roctavian)"
        result = normalize_answer("A", "multiple_choice", prompt)
        assert "Onasemnogene abeparvovec" in result


class TestNumericNormalization:
    def test_integer(self):
        assert normalize_answer("42", "numeric_match") == "42"

    def test_float(self):
        assert normalize_answer("3.14", "numeric_match") == "3.14"

    def test_with_text(self):
        assert normalize_answer("The answer is 42.", "numeric_match") == "42"

    def test_negative(self):
        assert normalize_answer("-5", "numeric_match") == "-5"

    def test_whole_float(self):
        assert normalize_answer("7.0", "numeric_match") == "7"


class TestStringNormalization:
    def test_gof(self):
        assert normalize_answer("GOF", "string_match") == "GOF"

    def test_gain_of_function(self):
        assert normalize_answer("gain-of-function", "string_match") == "GOF"

    def test_lof(self):
        assert normalize_answer("LOF", "string_match") == "LOF"

    def test_loss_of_function(self):
        assert normalize_answer("loss-of-function", "string_match") == "LOF"

    def test_drug_name(self):
        assert normalize_answer("everolimus", "string_match") == "everolimus"

    def test_strip_preamble(self):
        assert normalize_answer("The answer is everolimus.", "string_match") == "everolimus"

    def test_strip_quotes(self):
        assert normalize_answer('"nonsense"', "string_match") == "nonsense"

    def test_preserve_case(self):
        assert normalize_answer("Vosoritide", "string_match") == "Vosoritide"

    def test_trailing_period(self):
        assert normalize_answer("beta-blockers.", "string_match") == "beta-blockers"
