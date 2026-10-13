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


class TestPromptVariants:
    """The sensitivity controls change wording or order only, never content."""

    def _prompt(self):
        from src.prompts import build_system_prompt
        return build_system_prompt(category="Established_Targeted", parsed_variants=[], clinical_context="x",
                                   answer_format="binary", date_submitted="2025-08-01")

    def test_reword_swaps_only_the_two_headings(self, monkeypatch):
        from src import prompts
        base = self._prompt()
        monkeypatch.setattr(prompts, "PROMPT_VARIANT", "reword")
        reworded = self._prompt()

        assert "## How to use the tools" in reworded and "## Tool Usage Strategy" not in reworded
        assert "Suggested order of tools for this category:" in reworded
        restored = reworded.replace("## How to use the tools", "## Tool Usage Strategy").replace(
            "Suggested order of tools for this category:", "Recommended tool order for this category:")
        assert restored == base  # nothing else differs

    def test_unknown_variant_is_refused(self):
        from src.prompts import set_prompt_variant
        import pytest
        with pytest.raises(ValueError):
            set_prompt_variant("louder")


class TestMemoryRule:
    def test_prompt_tells_the_model_a_remembered_fact_is_a_hypothesis(self):
        from src.preprocessing import ParsedVariant
        from src.prompts import build_system_prompt
        v = ParsedVariant(gene="DMD", transcript="NM_004006.3", variant_cdna="c.70T>C", variant_protein="p.(Trp24Arg)",
                          zygosity="hemizygous", variant_type="missense", cdna_position="70", protein_position="24",
                          amino_acid_change="Trp24Arg")
        prompt = build_system_prompt("Established_Targeted", [v], "Progressive muscle weakness", "string_match", "2025-08-01")
        assert "A fact you remember is a hypothesis, not evidence" in prompt
        assert "must appear in a tool result of this conversation" in prompt

