"""ClinGen dosage tool against the curation file recorded on 2026-10-08."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from src.tools import clingen_dosage
from src.tools.clingen_dosage import ClinGenDosageTool

FIXTURES = Path(__file__).parent / "fixtures"


class _Reply:
    def __init__(self, status, text):
        self.status_code, self.text = status, text


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        return _Reply(200, (FIXTURES / "clingen_gene_curation_list_GRCh38.tsv").read_text())

    monkeypatch.setattr(clingen_dosage.requests, "get", fake_get)
    return calls


def test_dmd_is_haploinsufficient_with_no_triplosensitivity(recorded):
    summary, evidence = ClinGenDosageTool().execute(gene="DMD")

    assert "Haploinsufficiency score: 3 (sufficient evidence for dosage pathogenicity)" in summary
    assert "Haploinsufficiency disease: MONDO:0010679" in summary
    assert "Triplosensitivity score: 0 (no evidence available)" in summary
    assert "Last evaluated: 2019-11-20" in summary
    assert "file dated 07 Oct,2026" in summary
    assert [e.url for e in evidence] == ["https://www.ncbi.nlm.nih.gov/projects/dbvar/clingen/clingen_gene.cgi?sym=DMD"]


def test_agxt_is_a_recessive_gene_with_triplosensitivity_not_evaluated(recorded):
    summary, _ = ClinGenDosageTool().execute(gene="AGXT")

    assert "Haploinsufficiency score: 30 (gene associated with autosomal recessive phenotype)" in summary
    assert "Triplosensitivity score: Not yet evaluated (not yet evaluated)" in summary


def test_grin2b_lists_its_evidence_pmids(recorded):
    summary, _ = ClinGenDosageTool().execute(gene="GRIN2B")

    assert "Haploinsufficiency evidence PMIDs: 23160955, 20890276, 28377535" in summary


def test_uncurated_gene_is_said_to_be_uncurated_not_absent(recorded):
    summary, evidence = ClinGenDosageTool().execute(gene="ANO10")

    assert summary.startswith("ClinGen dosage: ANO10 has no dosage curation in ClinGen's list (1532 genes")
    assert "Not curated is not evidence of absence" in summary
    assert evidence == []


def test_file_is_downloaded_once_per_tool(recorded):
    tool = ClinGenDosageTool()
    tool.execute(gene="DMD")
    tool.execute(gene="NF1")
    tool.execute(gene="KCNT1")

    assert len(recorded) == 1


def test_download_failure_is_reported(monkeypatch, caplog):
    monkeypatch.setattr(clingen_dosage.time, "sleep", lambda s: None)
    monkeypatch.setattr(clingen_dosage.requests, "get", lambda url, timeout=None: _Reply(503, "down"))

    summary, evidence = ClinGenDosageTool().execute(gene="DMD")

    assert summary == "ClinGen dosage returned nothing for DMD: HTTP 503 after 5 attempts"
    assert evidence == []
    assert caplog.text.count("server error") == 5
