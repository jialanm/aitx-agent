"""Tests for prompt construction that does not need the model."""

from src.prompts import ANSWER_FORMAT_INSTRUCTIONS, build_system_prompt


def _prompt(answer_format, options=None):
    return build_system_prompt(
        category="Established_Targeted",
        parsed_variants=[],
        clinical_context="ctx",
        answer_format=answer_format,
        date_submitted="2025-08-01",
        options=options,
    )


class TestMultipleChoiceInstruction:
    OPTS = ["Golodirsen", "Viltolarsen", "Eteplirsen", "Casimersen", "Ataluren", "None"]

    def test_verified_options_are_spelled_out(self):
        text = _prompt("multiple_choice", self.OPTS)
        assert "exactly one of these options, spelled as given: " + "; ".join(self.OPTS) in text
        assert ANSWER_FORMAT_INSTRUCTIONS["multiple_choice"] not in text

    def test_no_options_falls_back_to_generic_instruction(self):
        text = _prompt("multiple_choice", [])
        assert ANSWER_FORMAT_INSTRUCTIONS["multiple_choice"] in text
        assert "spelled as given" not in text

    def test_options_ignored_for_other_formats(self):
        text = _prompt("binary", self.OPTS)
        assert ANSWER_FORMAT_INSTRUCTIONS["binary"] in text
        assert "Golodirsen" not in text
