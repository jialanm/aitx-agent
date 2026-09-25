"""Integration tests for API tools.

These tests make real API calls. Run with: pytest tests/test_tools.py -v -s
Mark with -m slow to skip in quick runs.
"""

import pytest

from src.tools.clinvar import ClinVarTool
from src.tools.clinical_trials import ClinicalTrialsTool
from src.tools.ensembl import EnsemblTool
from src.tools.genereviews import GeneReviewsTool
from src.tools.pubmed import PubMedTool
from src.tools.uniprot import UniProtTool
from src.tools.pharmgkb import PharmGKBTool
from src.tools.omim import OMIMTool
from src.tools.openfda import OpenFDATool


class TestClinVarTool:
    def test_schema(self):
        tool = ClinVarTool()
        schema = tool.schema()
        assert schema["function"]["name"] == "search_clinvar"
        assert "gene" in schema["function"]["parameters"]["properties"]

    @pytest.mark.slow
    def test_cftr_variant(self):
        tool = ClinVarTool()
        summary, evidence = tool.execute(gene="CFTR", variant="c.1521_1523del")
        assert len(evidence) > 0
        assert "clinvar" in evidence[0].url.lower()
        assert summary  # non-empty

    @pytest.mark.slow
    def test_brca1_variant(self):
        tool = ClinVarTool()
        summary, evidence = tool.execute(gene="BRCA1", variant="c.5266dup")
        assert summary


class TestEnsemblTool:
    def test_schema(self):
        tool = EnsemblTool()
        schema = tool.schema()
        assert schema["function"]["name"] == "query_ensembl"

    @pytest.mark.slow
    def test_gene_lookup(self):
        tool = EnsemblTool()
        summary, evidence = tool.execute(
            hgvs_cdna="NM_004333.6:c.1799T>A",
            gene="BRAF",
        )
        assert summary
        assert any("ensembl" in e.url.lower() for e in evidence)


class TestGeneReviewsTool:
    def test_schema(self):
        tool = GeneReviewsTool()
        schema = tool.schema()
        assert schema["function"]["name"] == "search_genereviews"

    @pytest.mark.slow
    def test_cftr_lookup(self):
        tool = GeneReviewsTool()
        summary, evidence = tool.execute(gene="CFTR")
        assert summary
        assert len(evidence) > 0


class TestClinicalTrialsTool:
    def test_schema(self):
        tool = ClinicalTrialsTool()
        schema = tool.schema()
        assert schema["function"]["name"] == "search_clinical_trials"

    @pytest.mark.slow
    def test_tp53_trials(self):
        tool = ClinicalTrialsTool()
        summary, evidence = tool.execute(query="TP53 cancer targeted therapy")
        assert summary
        assert len(evidence) > 0
        assert any("clinicaltrials.gov" in e.url for e in evidence)


class TestPubMedTool:
    def test_schema(self):
        tool = PubMedTool()
        schema = tool.schema()
        assert schema["function"]["name"] == "search_pubmed"

    @pytest.mark.slow
    def test_cftr_search(self):
        tool = PubMedTool()
        summary, evidence = tool.execute(query="CFTR F508del Trikafta treatment")
        assert summary
        assert len(evidence) > 0
        assert any("pubmed" in e.url for e in evidence)


class TestUniProtTool:
    def test_schema(self):
        tool = UniProtTool()
        schema = tool.schema()
        assert schema["function"]["name"] == "query_uniprot"
        assert "gene" in schema["function"]["parameters"]["properties"]
        assert "protein_position" in schema["function"]["parameters"]["properties"]

    @pytest.mark.slow
    def test_cftr_lookup(self):
        tool = UniProtTool()
        summary, evidence = tool.execute(gene="CFTR", protein_position=508)
        assert summary
        assert len(evidence) > 0
        assert any("uniprot" in e.url.lower() for e in evidence)

    @pytest.mark.slow
    def test_brca1_lookup(self):
        tool = UniProtTool()
        summary, evidence = tool.execute(gene="BRCA1")
        assert summary
        assert len(evidence) > 0


class TestPharmGKBTool:
    def test_schema(self):
        tool = PharmGKBTool()
        schema = tool.schema()
        assert schema["function"]["name"] == "search_pharmgkb"
        assert "gene" in schema["function"]["parameters"]["properties"]
        assert "drug" in schema["function"]["parameters"]["properties"]

    @pytest.mark.slow
    def test_cyp2d6_lookup(self):
        tool = PharmGKBTool()
        summary, evidence = tool.execute(gene="CYP2D6")
        assert summary
        assert len(evidence) > 0
        assert any("clinpgx.org" in e.url for e in evidence)

    @pytest.mark.slow
    def test_dpyd_with_drug(self):
        tool = PharmGKBTool()
        summary, evidence = tool.execute(gene="DPYD", drug="fluorouracil")
        assert summary


class TestOMIMTool:
    def test_schema(self):
        tool = OMIMTool()
        schema = tool.schema()
        assert schema["function"]["name"] == "search_omim"
        assert "gene" in schema["function"]["parameters"]["properties"]
        assert "condition" in schema["function"]["parameters"]["properties"]

    @pytest.mark.slow
    def test_cftr_lookup(self):
        tool = OMIMTool()
        summary, evidence = tool.execute(gene="CFTR")
        assert summary
        assert len(evidence) > 0

    @pytest.mark.slow
    def test_pcca_with_condition(self):
        tool = OMIMTool()
        summary, evidence = tool.execute(gene="PCCA", condition="propionic acidemia")
        assert summary


class TestOpenFDATool:
    def test_schema(self):
        tool = OpenFDATool()
        schema = tool.schema()
        assert schema["function"]["name"] == "search_fda_labels"
        assert "drug" in schema["function"]["parameters"]["properties"]
        assert "indication" in schema["function"]["parameters"]["properties"]

    @pytest.mark.slow
    def test_olaparib_lookup(self):
        tool = OpenFDATool()
        summary, evidence = tool.execute(drug="olaparib")
        assert summary
        assert len(evidence) > 0
        assert "indications" in summary.lower() or "Indications" in summary

    @pytest.mark.slow
    def test_olaparib_with_brca(self):
        tool = OpenFDATool()
        summary, evidence = tool.execute(drug="olaparib", indication="BRCA")
        assert summary
        assert len(evidence) > 0

    @pytest.mark.slow
    def test_unknown_drug(self):
        tool = OpenFDATool()
        summary, evidence = tool.execute(drug="xyznotadrug12345")
        assert "No FDA drug labels" in summary or "error" in summary.lower()
