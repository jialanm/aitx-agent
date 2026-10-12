"""GeneReviews adapter against the NCBI index files and chapter abstracts recorded on 2026-10-11."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from src.tools import base, genereviews
from src.tools.genereviews import GeneReviewsTool

FIXTURES = Path(__file__).parent / "fixtures"


class _Reply:
    def __init__(self, status, text):
        self.status_code, self.text = status, text


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        if url.endswith("NBKid_shortname_genesymbol.txt"):
            return _Reply(200, (FIXTURES / "genereviews_NBKid_shortname_genesymbol.txt").read_text())
        if url.endswith("GRtitle_shortname_NBKid.txt"):
            return _Reply(200, (FIXTURES / "genereviews_GRtitle_shortname_NBKid.txt").read_text())
        return _Reply(200, (FIXTURES / f"genereviews_abstract_{params['id']}.txt").read_text())

    monkeypatch.setattr(genereviews.requests, "get", fake_get)
    return calls


def test_gene_with_several_chapters_lists_them_for_the_model_to_choose(recorded):
    summary, evidence = GeneReviewsTool().execute(gene="COL1A1")

    assert summary.startswith("GeneReviews chapters indexed to COL1A1 (3)")
    assert "- NBK1244: Classic Ehlers-Danlos Syndrome (PMID 20301422)" in summary
    assert "- NBK1295: COL1A1- and COL1A2-Related Osteogenesis Imperfecta (PMID 20301472)" in summary
    assert "- NBK99168: Caffey Disease (PMID 22855962)" in summary
    assert [e.url for e in evidence] == [f"https://www.ncbi.nlm.nih.gov/books/{n}/" for n in ("NBK1244", "NBK1295", "NBK99168")]
    assert len(recorded) == 2  # the two index files, no abstract fetched


def test_chosen_chapter_returns_the_full_management_section(recorded):
    summary, evidence = GeneReviewsTool().execute(gene="COL1A1", nbk="NBK1244")

    assert summary.startswith("GeneReviews: Classic Ehlers-Danlos Syndrome (NBK1244, PMID 20301422, https://www.ncbi.nlm.nih.gov/books/NBK1244/)")
    assert "MANAGEMENT:" in summary
    assert "ascorbic acid" in summary.lower() and "desmopressin" in summary.lower()
    assert "CLINICAL CHARACTERISTICS" not in summary
    assert [e.url for e in evidence] == ["https://www.ncbi.nlm.nih.gov/books/NBK1244/"]
    assert recorded[-1][1]["id"] == "20301422"


def test_gene_with_one_chapter_is_fetched_at_once_and_names_the_drugs(recorded):
    summary, evidence = GeneReviewsTool().execute(gene="AGXT")

    assert "Primary Hyperoxaluria Type 1 (NBK1283" in summary
    assert "Pyridoxine (vitamin B6)" in summary and "missense AGXT variants" in summary
    assert "lumasiran" in summary and "nedosiran" in summary
    assert evidence[0].url == "https://www.ncbi.nlm.nih.gov/books/NBK1283/"


def test_all_sections_can_be_requested(recorded):
    summary, _ = GeneReviewsTool().execute(gene="AGXT", section="all")

    for heading in ("CLINICAL CHARACTERISTICS:", "DIAGNOSIS/TESTING:", "MANAGEMENT:", "GENETIC COUNSELING:"):
        assert heading in summary


def test_overview_chapter_without_sections_returns_whole_abstract(recorded):
    summary, _ = GeneReviewsTool().execute(gene="ANO10")

    assert "Hereditary Ataxia Overview (NBK1138" in summary
    assert "no structured sections; full abstract" in summary


def test_gene_without_a_chapter_is_said_so(recorded):
    summary, evidence = GeneReviewsTool().execute(gene="SLC35A2")

    assert summary == "GeneReviews has no chapter indexed to gene SLC35A2."
    assert evidence == []


def test_index_is_downloaded_once_per_tool(recorded):
    tool = GeneReviewsTool()
    tool.execute(gene="AGXT")
    tool.execute(gene="NF1")

    assert sum(1 for u, _ in recorded if u.endswith(".txt")) == 2


def test_index_download_failure_is_reported(monkeypatch, caplog):
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    monkeypatch.setattr(genereviews.requests, "get", lambda url, params=None, timeout=None: _Reply(503, "down"))

    summary, evidence = GeneReviewsTool().execute(gene="AGXT")

    assert summary.startswith("GeneReviews returned nothing: gene index unavailable (request failed (503 Server Error after 5 attempts")
    assert evidence == []
