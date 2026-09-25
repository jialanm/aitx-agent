"""Tests for HGVS parsing and preprocessing."""

import pytest

from src.preprocessing import ParsedVariant, parse_variant, preprocess_input
from src.schemas import Patient, Question, TaskInput, Variant


@pytest.fixture
def sample_variants():
    """Sample variants covering different types."""
    return [
        Variant(
            gene="CFTR",
            transcript="NM_000492.4",
            variant_cdna="c.1521_1523del",
            variant_protein="p.Phe508del",
            zygosity="homozygous",
        ),
        Variant(
            gene="SCN1A",
            transcript="NM_001165963.4",
            variant_cdna="c.2836C>T",
            variant_protein="p.Arg946Ter",
            zygosity="heterozygous",
        ),
        Variant(
            gene="BRCA1",
            transcript="NM_007294.4",
            variant_cdna="c.5266dup",
            variant_protein="p.Gln1756ProfsTer74",
            zygosity="heterozygous",
        ),
        Variant(
            gene="BRAF",
            transcript="NM_004333.6",
            variant_cdna="c.1799T>A",
            variant_protein="p.Val600Glu",
            zygosity="heterozygous",
        ),
        Variant(
            gene="HTT",
            transcript="NM_002111.8",
            variant_cdna="c.52CAG[42]",
            variant_protein="p.Gln18[42]",
            zygosity="heterozygous",
        ),
        Variant(
            gene="RB1",
            transcript="NM_000321.3",
            variant_cdna="c.2359C>T",
            variant_protein="p.Arg787Ter",
            zygosity="heterozygous",
        ),
        Variant(
            gene="GAA",
            transcript="NM_000152.5",
            variant_cdna="c.-32-13T>G",
            variant_protein="p.?",
            zygosity="homozygous",
        ),
        Variant(
            gene="PCCA",
            transcript="NM_000282.4",
            variant_cdna="c.1218_1231del",
            variant_protein="p.Gly407AspfsTer14",
            zygosity="homozygous",
        ),
    ]


class TestParseVariant:
    def test_in_frame_deletion(self, sample_variants):
        result = parse_variant(sample_variants[0])  # CFTR p.Phe508del
        assert result.variant_type == "in-frame deletion"
        assert result.gene == "CFTR"
        assert result.protein_position == "508"
        assert result.cdna_position == "1521_1523"

    def test_nonsense(self, sample_variants):
        result = parse_variant(sample_variants[1])  # SCN1A p.Arg946Ter
        assert result.variant_type == "nonsense"
        assert result.protein_position == "946"

    def test_frameshift(self, sample_variants):
        result = parse_variant(sample_variants[2])  # BRCA1 p.Gln1756ProfsTer74
        assert result.variant_type == "frameshift"
        assert result.protein_position == "1756"

    def test_missense(self, sample_variants):
        result = parse_variant(sample_variants[3])  # BRAF p.Val600Glu
        assert result.variant_type == "missense"
        assert result.protein_position == "600"

    def test_repeat_expansion(self, sample_variants):
        result = parse_variant(sample_variants[4])  # HTT p.Gln18[42]
        assert result.variant_type == "repeat expansion"

    def test_nonsense_rb1(self, sample_variants):
        result = parse_variant(sample_variants[5])  # RB1 p.Arg787Ter
        assert result.variant_type == "nonsense"
        assert result.protein_position == "787"

    def test_splice_variant(self, sample_variants):
        result = parse_variant(sample_variants[6])  # GAA c.-32-13T>G, p.?
        assert result.variant_type == "splice"
        assert result.cdna_position == "-32-13"

    def test_frameshift_pcca(self, sample_variants):
        result = parse_variant(sample_variants[7])  # PCCA p.Gly407AspfsTer14
        assert result.variant_type == "frameshift"
        assert result.protein_position == "407"


class TestPreprocessInput:
    def test_basic_preprocessing(self):
        task = TaskInput(
            id="AITX-00001",
            patient=Patient(
                genotype=[
                    Variant(
                        gene="CFTR",
                        transcript="NM_000492.4",
                        variant_cdna="c.1521_1523del",
                        variant_protein="p.Phe508del",
                        zygosity="homozygous",
                    )
                ],
                clinical_context="A 15-year-old male with cystic fibrosis.",
            ),
            question=Question(
                category="Established_Targeted",
                answer_format="binary",
                prompt="Is this variant eligible for Trikafta?",
                date_submitted="2024-12-10",
            ),
        )

        result = preprocess_input(task)
        assert result["genes"] == ["CFTR"]
        assert result["category"] == "Established_Targeted"
        assert result["answer_format"] == "binary"
        assert len(result["parsed_variants"]) == 1
        assert result["parsed_variants"][0].variant_type == "in-frame deletion"
