"""Live claims: facts whose truth is a database state, and whether we hold it.

The sentences are the ones the 32B actually wrote in the runs of 2026-10-26
and 2026-10-28.
"""

from __future__ import annotations

from src.live_claims import attempted, notice, sources_held, unsourced

CHAPTER = {"tool": "search_genereviews", "arguments": {"gene": "AGXT"},
           "summary": "GeneReviews: Primary Hyperoxaluria Type 1\nMANAGEMENT: RNA interference therapeutics "
                      "(lumasiran and nedosiran) that target hepatic enzymes"}
LABEL = {"tool": "search_fda_labels", "arguments": {"drug": "lumasiran"},
         "summary": "FDA Drug Label for lumasiran:\n\nIndications: OXLUMO is indicated for PH1"}
EMPTY_LABEL = {"tool": "search_fda_labels", "arguments": {"drug": "Ataluren"},
               "summary": "No FDA drug labels found for: Ataluren. openFDA holds only drugs with a United States label"}
TRIAL_SEARCH = {"tool": "search_clinical_trials", "arguments": {"query": "SLC35A2-CDG", "status_filter": "upcoming"},
                "summary": "NCT05402384 Title: AVTX-801 D-galactose Supplementation in SLC35A2-CDG Status: NOT_YET_RECRUITING"}

APPROVAL_FROM_MEMORY = "Yes, lumasiran is approved, and nedosiran was approved in 2023."
ATALUREN_FROM_MEMORY = ("The FDA label search for Ataluren didn't return any results. "
                        "That means it's not FDA-approved in the US for DMD.")


def test_approval_claimed_with_only_a_chapter_mention_is_unsourced():
    """AITX-00006: both drugs are named by GeneReviews and neither label was
    read, so the chapter mention is not a source for their approval."""
    found = unsourced(APPROVAL_FROM_MEMORY, ["lumasiran", "nedosiran"], sources_held([CHAPTER]))

    assert [(name, category, tool) for name, _k, category, tool in found] == [
        ("lumasiran", "status", "search_fda_labels"),
        ("nedosiran", "status", "search_fda_labels"),
    ]


def test_the_drug_whose_label_was_read_is_not_challenged():
    found = unsourced(APPROVAL_FROM_MEMORY, ["lumasiran", "nedosiran"], sources_held([CHAPTER, LABEL]))

    assert [name for name, *_ in found] == ["nedosiran"]


def test_an_empty_reply_is_not_a_source_and_the_next_lookup_is_named():
    """AITX-00002: the label search returned nothing, which is not evidence,
    and PubMed is the lookup that settled it in run 4."""
    calls = [EMPTY_LABEL]
    found = unsourced(ATALUREN_FROM_MEMORY, ["Ataluren"], sources_held(calls), tried=attempted(calls))

    assert found == [("ataluren", "drug", "status", "search_pubmed")]


def test_the_entity_carries_from_the_sentence_before_the_claim():
    found = unsourced("Ataluren was considered. It is not approved.", ["Ataluren"], {})

    assert [name for name, *_ in found] == ["ataluren"]


def test_a_trial_identifier_in_a_summary_counts_as_a_source():
    """AITX-00009: the trial was fetched by condition, so the call's
    arguments do not name it but its record is in hand."""
    held = sources_held([TRIAL_SEARCH])

    assert ("trial", "nct05402384") in held
    assert unsourced("NCT05402384 is not yet recruiting.", [], held) == []


def test_a_mechanism_claim_is_not_challenged():
    """The model's knowledge is wanted for mechanism and drug class."""
    assert unsourced("Memantine is an NMDA receptor antagonist, so it lowers receptor activity.",
                     ["Memantine"], {}) == []


def test_a_classification_claim_needs_clinvar():
    found = unsourced("This variant is pathogenic.", [], {}, gene="DMD")

    assert found == [("dmd", "variant", "classification", "search_clinvar")]


def test_the_notice_names_the_lookup_and_does_not_call_the_claim_false():
    text = notice([("ataluren", "drug", "status", "search_pubmed")])

    assert "search_pubmed" in text and "ataluren" in text
    assert "not verified against a current source" in text
    assert "false" not in text.lower() or "rather than treating it as false" in text


class TestNamesTheModelIntroduces:
    """A drug the harness has never seen, named in the model's own words
    (AITX-00006's thinking: "there's a drug called pegalysine (Oxepduo)
    approved in the US", which RxNorm does not know and which no option,
    answer or lookup mentioned)."""

    SENTENCE = "Wait, I remember that for PH1, there's a drug called pegalysine (Oxepduo) approved in the US."

    def test_a_name_the_model_calls_a_drug_needs_no_confirmation(self):
        from src.live_claims import introduced_names

        assert introduced_names(self.SENTENCE) == [("pegalysine", True), ("Oxepduo", False)]

    def test_the_claim_is_reported_and_the_label_lookup_named(self):
        found = unsourced(self.SENTENCE, [], {}, confirm=lambda name: False, cache={})

        assert found == [("pegalysine", "drug", "status", "search_fda_labels")]

    def test_a_shaped_name_is_reported_only_when_rxnorm_confirms_it(self):
        sentence = "Eteplirsen is approved for exon 51 skipping."
        assert unsourced(sentence, [], {}, confirm=lambda name: True, cache={})
        assert unsourced(sentence, [], {}, confirm=lambda name: False, cache={}) == []

    def test_an_ordinary_capitalised_word_is_not_challenged(self):
        sentence = "The patient has Duchenne muscular dystrophy and no therapy is approved for exon 74."
        assert unsourced(sentence, [], {}, confirm=lambda name: False, cache={}) == []

    def test_rxnorm_being_unreachable_yields_no_candidate(self):
        """Its unknown is not a drug and not a licence to challenge a guess."""
        sentence = "Golodirsen is approved for exon 53 skipping."
        assert unsourced(sentence, [], {}, confirm=lambda name: None, cache={}) == []

    def test_one_rxnorm_answer_is_reused_per_name(self):
        calls = []

        def confirm(name):
            calls.append(name)
            return True

        cache: dict[str, bool | None] = {}
        text = "Golodirsen is approved for exon 53. Golodirsen is indicated for exon 53."
        unsourced(text, [], {}, confirm=confirm, cache=cache)

        assert calls == ["Golodirsen"]
