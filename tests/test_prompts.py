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


class TestValidationInPrompt:
    """Validator results are shown under each variant in the system prompt."""

    @staticmethod
    def _parsed():
        from src.preprocessing import parse_variant
        from src.schemas import Variant
        return [parse_variant(Variant(gene="DMD", transcript="NM_004006.2", variant_cdna="c.70T>C",
                                      variant_protein="p.(Trp24Arg)", zygosity="hemizygous"))]

    def test_rewritten_variant_is_shown_with_superseded_note(self):
        from src.prompts import _format_variant_summary
        from src.variant_validation import ValidatedVariant
        val = ValidatedVariant(submitted="NM_004006.2:c.70T>C", valid=True, transcript_variant="NM_004006.3:c.70T>C",
                               genomic_variant="NC_000023.11:g.33020162A>G", gene_symbol="DMD",
                               superseded_transcript="NM_004006.2")

        text = _format_variant_summary(self._parsed(), [val])

        assert "Validated (VariantValidator, GRCh38): NM_004006.3:c.70T>C" in text
        assert "Genomic position: NC_000023.11:g.33020162A>G" in text
        assert "NM_004006.2 is superseded" in text
        assert "VALIDATION FAILED" not in text

    def test_failed_validation_is_shown_with_warnings(self):
        from src.prompts import _format_variant_summary
        from src.variant_validation import ValidatedVariant
        val = ValidatedVariant(submitted="NM_004006.2:c.70T>C", valid=False,
                               warnings=["validator unavailable: timeout after retries"])

        text = _format_variant_summary(self._parsed(), [val])

        assert "VALIDATION FAILED for NM_004006.2:c.70T>C: validator unavailable: timeout after retries" in text
        assert "cDNA: c.70T>C" in text  # the submitted strings are still shown

    def test_no_validation_leaves_summary_unchanged(self):
        from src.prompts import _format_variant_summary
        text = _format_variant_summary(self._parsed())
        assert "Validated" not in text and "VALIDATION" not in text
