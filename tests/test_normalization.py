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
    PROMPT = "Which drug? (A) DrugA (B) DrugB (C) DrugC (D) DrugD"
    OPTIONS = ["DrugA", "DrugB", "DrugC", "DrugD"]

    def test_letter_with_text(self):
        assert normalize_answer("(C) DrugC", "multiple_choice", self.PROMPT, self.OPTIONS) == "DrugC"

    def test_letter_only_maps_through_lettered_prompt(self):
        assert normalize_answer("C", "multiple_choice", self.PROMPT, self.OPTIONS) == "DrugC"

    def test_text_match(self):
        prompt = "Which drug? (A) DrugA (B) DrugB (C) Carbamazepine (D) DrugD"
        options = ["DrugA", "DrugB", "Carbamazepine", "DrugD"]
        assert normalize_answer("Carbamazepine", "multiple_choice", prompt, options) == "Carbamazepine"

    def test_option_with_explanation(self):
        assert normalize_answer("A) DrugA because it is effective", "multiple_choice", self.PROMPT, self.OPTIONS) == "DrugA"

    def test_no_options_leaves_text_alone(self):
        assert normalize_answer("C", "multiple_choice", self.PROMPT, []) == "C"


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


class TestOptionVerification:
    """Model-proposed option lists are accepted only when the prompt backs them."""

    DMD = ("To which of the following targeted therapies would this variant be most "
           "likely amenable: Golodirsen, Viltolarsen, Eteplirsen, Casimersen, Ataluren, or None?")
    GRIN2B = "Is it more likely amenable to treatment with Memantine, L-serine, or Radiprodil"
    NF1 = "In which functional domain does this variant occur? Answer choices: CSRD, TBD, GRD, Sec14-PH, HLR, NLS, SBR."

    def test_accepts_list_backed_by_prompt(self):
        from src.normalization import verify_options
        cands = ["Golodirsen", "Viltolarsen", "Eteplirsen", "Casimersen", "Ataluren", "None"]
        assert verify_options(cands, self.DMD) == (cands, None)

    def test_returns_prompt_spelling(self):
        from src.normalization import verify_options
        options, reason = verify_options(["memantine", "L-Serine", "RADIPRODIL"], self.GRIN2B)
        assert reason is None
        assert options == ["Memantine", "L-serine", "Radiprodil"]

    def test_rejects_option_not_in_prompt(self):
        from src.normalization import verify_options
        options, reason = verify_options(["CSRD", "TBD", "Nusinersen"], self.NF1)
        assert options == [] and "not in prompt" in reason

    def test_rejects_partial_word_match(self):
        from src.normalization import verify_options
        # "GRD" is in the prompt; "RD" is only inside other words.
        options, reason = verify_options(["GRD", "RD"], self.NF1)
        assert options == [] and "not in prompt" in reason

    def test_rejects_single_option(self):
        from src.normalization import verify_options
        assert verify_options(["GRD"], self.NF1)[0] == []

    def test_rejects_duplicates(self):
        from src.normalization import verify_options
        options, reason = verify_options(["GRD", "grd", "TBD"], self.NF1)
        assert options == [] and "duplicate" in reason

    def test_rejects_overlapping_options(self):
        from src.normalization import verify_options
        prompt = "Which two? Answer choices: ascorbic acid, acid, desmopressin."
        options, reason = verify_options(["ascorbic acid", "acid", "desmopressin"], prompt)
        assert options == [] and "contained in" in reason

    def test_rejects_dropped_middle_option(self):
        from src.normalization import verify_options
        cands = ["Golodirsen", "Viltolarsen", "Eteplirsen", "Ataluren", "None"]  # Casimersen missing
        options, reason = verify_options(cands, self.DMD)
        assert options == [] and "dropped option" in reason and "Casimersen" in reason

    def test_rejects_dropped_last_option(self):
        from src.normalization import verify_options
        cands = ["Golodirsen", "Viltolarsen", "Eteplirsen", "Casimersen", "Ataluren"]  # None missing
        options, reason = verify_options(cands, self.DMD)
        assert options == [] and "dropped option" in reason and "None" in reason

    def test_rejects_dropped_last_option_without_oxford_comma(self):
        from src.normalization import verify_options
        options, reason = verify_options(["Memantine", "L-serine"], self.GRIN2B)  # Radiprodil missing
        assert options == [] and "Radiprodil" in reason

    def test_rejects_dropped_first_option(self):
        from src.normalization import verify_options
        options, reason = verify_options(["TBD", "GRD", "Sec14-PH", "HLR", "NLS", "SBR"], self.NF1)
        assert options == [] and "dropped option before" in reason

    def test_rejects_truncated_list(self):
        from src.normalization import verify_options
        options, reason = verify_options(["CSRD", "TBD", "GRD"], self.NF1)
        assert options == [] and "Sec14-PH" in reason

    def test_complete_lists_pass_for_every_challenge_prompt(self):
        from src.normalization import verify_options
        cases = [
            (self.DMD, ["Golodirsen", "Viltolarsen", "Eteplirsen", "Casimersen", "Ataluren", "None"]),
            (self.GRIN2B, ["Memantine", "L-serine", "Radiprodil"]),
            (self.NF1, ["CSRD", "TBD", "GRD", "Sec14-PH", "HLR", "NLS", "SBR"]),
            ("Which is more likely: gain of function, loss of function, or dominant negative?",
             ["gain of function", "loss of function", "dominant negative"]),
            ("Choose between eteplirsen and ataluren.", ["eteplirsen", "ataluren"]),
        ]
        for prompt, cands in cases:
            options, reason = verify_options(cands, prompt)
            assert reason is None, (prompt, reason)
            assert options == cands

    def test_order_in_json_does_not_matter(self):
        from src.normalization import verify_options
        options, reason = verify_options(["Radiprodil", "Memantine", "L-serine"], self.GRIN2B)
        assert reason is None


class TestOptionJsonParsing:
    def test_plain_array(self):
        from src.normalization import parse_option_json
        assert parse_option_json('["CSRD", "TBD"]') == ["CSRD", "TBD"]

    def test_code_fence_and_preamble(self):
        from src.normalization import parse_option_json
        text = 'Here are the choices:\n```json\n["Memantine", "L-serine", "Radiprodil"]\n```'
        assert parse_option_json(text) == ["Memantine", "L-serine", "Radiprodil"]

    def test_no_array(self):
        from src.normalization import parse_option_json
        assert parse_option_json("The choices are CSRD and TBD.") is None

    def test_non_string_items(self):
        from src.normalization import parse_option_json
        assert parse_option_json('[1, 2]') is None

    def test_empty_string_item(self):
        from src.normalization import parse_option_json
        assert parse_option_json('["GRD", ""]') is None


class TestInlineMultipleChoice:
    """Mapping model output onto verified inline options from the challenge set."""

    DMD = TestOptionVerification.DMD
    GRIN2B = TestOptionVerification.GRIN2B
    NF1 = TestOptionVerification.NF1
    DMD_OPTS = ["Golodirsen", "Viltolarsen", "Eteplirsen", "Casimersen", "Ataluren", "None"]
    GRIN2B_OPTS = ["Memantine", "L-serine", "Radiprodil"]
    NF1_OPTS = ["CSRD", "TBD", "GRD", "Sec14-PH", "HLR", "NLS", "SBR"]

    def test_exact_option(self):
        assert normalize_answer("Eteplirsen", "multiple_choice", self.DMD, self.DMD_OPTS) == "Eteplirsen"

    def test_case_and_punctuation(self):
        assert normalize_answer("eteplirsen.", "multiple_choice", self.DMD, self.DMD_OPTS) == "Eteplirsen"

    def test_none_option(self):
        assert normalize_answer("None", "multiple_choice", self.DMD, self.DMD_OPTS) == "None"

    def test_option_inside_sentence(self):
        text = "This exon 51 skipping variant is amenable to Eteplirsen."
        assert normalize_answer(text, "multiple_choice", self.DMD, self.DMD_OPTS) == "Eteplirsen"

    def test_earliest_mention_wins(self):
        text = "Ataluren, not Eteplirsen, because this is a nonsense variant"
        assert normalize_answer(text, "multiple_choice", self.DMD, self.DMD_OPTS) == "Ataluren"

    def test_returns_prompt_spelling(self):
        assert normalize_answer("L-Serine", "multiple_choice", self.GRIN2B, self.GRIN2B_OPTS) == "L-serine"

    def test_answer_choices_form(self):
        assert normalize_answer("GRD (GAP-related domain)", "multiple_choice", self.NF1, self.NF1_OPTS) == "GRD"

    def test_letter_answer_ignored_without_lettered_prompt(self):
        assert normalize_answer("C", "multiple_choice", self.NF1, self.NF1_OPTS) == "C"

    def test_unmatched_answer_passes_through(self):
        assert normalize_answer("Nusinersen", "multiple_choice", self.DMD, self.DMD_OPTS) == "Nusinersen"

    def test_partial_word_does_not_match(self):
        assert normalize_answer("outbound", "multiple_choice", self.NF1, self.NF1_OPTS) == "outbound"
