"""The model's description of the variant against VEP's reported terms.

Sentences are the recorded ones from the 32B runs of 2026-10-26 and
2026-10-28 wherever possible, so the exclusions are tested against text the
model actually wrote.
"""

from __future__ import annotations

from src.consequence_check import contradiction, notice, vep_consequences

RUN4_00002 = "The patient's variant is a frameshifting insertion/deletion leading to a stop codon."
BLOCK_00002 = (
    "VEP (Ensembl, GRCh38) for NC_000023.11:g.31169542_31169543delinsTA on transcript NM_004006.3:\n"
    "Frame (VEP): stop_gained; no change in length\n"
    "Consequence: stop_gained\n"
    "Exon: 74/79\n"
)
BLOCK_00001 = (
    "Frame, as written: 1743 coding bases, a multiple of three deleted (VEP: inframe_deletion).\n"
    "Consequence (for the letters as written): splice_acceptor_variant, splice_donor_variant, inframe_deletion\n"
)


def test_vep_terms_are_read_from_either_consequence_heading():
    assert vep_consequences([BLOCK_00002]) == ["stop_gained"]
    assert vep_consequences([BLOCK_00001]) == ["splice_acceptor_variant", "splice_donor_variant", "inframe_deletion"]


def test_the_recorded_frameshift_claim_is_flagged():
    found = contradiction(RUN4_00002, ["stop_gained"])

    assert found is not None
    word, sentence = found
    assert word.lower() == "frameshifting"
    assert sentence == RUN4_00002


def test_premature_stop_beside_a_frameshift_is_not_a_contradiction():
    """A frameshift does create a premature stop downstream (AITX-00005,
    00006 and 00015 all say so correctly)."""
    assert contradiction("The patient's variant introduces a premature stop codon in exon 1.",
                         ["frameshift_variant"]) is None


def test_the_decay_mechanism_s_name_is_not_a_claim_about_the_variant():
    assert contradiction("The question is whether this variant is likely to result in nonsense-mediated decay.",
                         ["frameshift_variant"]) is None


def test_a_negated_sentence_is_not_a_claim():
    assert contradiction("So, the deletion as written is inframe, meaning no frameshift.",
                         vep_consequences([BLOCK_00001])) is None


def test_a_hypothetical_is_not_a_claim():
    assert contradiction("However, if the deletion leads to a frameshift, skipping an adjacent exon could restore it.",
                         vep_consequences([BLOCK_00001])) is None


def test_a_general_rule_sentence_is_not_a_claim_about_the_patient():
    assert contradiction("Read-through applies to premature stops from substitutions, not to stops from a frameshift.",
                         ["stop_gained"]) is None


def test_a_missense_claim_against_a_stop_is_flagged():
    found = contradiction("This mutation is a missense change in exon 74.", ["stop_gained"])
    assert found and found[0].lower() == "missense"


def test_nothing_is_flagged_when_vep_returned_no_terms():
    assert contradiction(RUN4_00002, []) is None


def test_the_notice_states_vep_s_values_and_no_conclusion():
    text = notice("frameshifting", ["stop_gained"])

    assert "stop_gained" in text and "frameshifting" in text
    assert "ataluren" not in text.lower() and "read-through" not in text.lower()
