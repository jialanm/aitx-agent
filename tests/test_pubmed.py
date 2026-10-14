"""PubMed adapter against a search and article records recorded on 2026-10-11."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools import pubmed
from src.tools.pubmed import PubMedTool

FIXTURES = Path(__file__).parent / "fixtures"


class _Reply:
    def __init__(self, path: Path):
        self.status_code, self.text = 200, path.read_text()
        self.url = "https://eutils.ncbi.nlm.nih.gov/x"

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        pass


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url.rsplit("/", 1)[-1], dict(params or {})))
        name = "pubmed_esearch_GRIN2B_LoF_Lserine.json" if "esearch" in url else "pubmed_efetch_GRIN2B_LoF_Lserine.xml"
        return _Reply(FIXTURES / name)

    monkeypatch.setattr(pubmed.requests, "get", fake_get)
    return calls


def test_articles_come_back_with_their_abstracts(recorded):
    summary, evidence = PubMedTool().execute(query="GRIN2B loss of function L-serine")

    assert "Total results: 20; showing 5 with abstracts" in summary
    assert "PMID: 41265180" in summary
    assert "Title: Potential benefits of l-serine in children with GRIN2B loss-of-function variants" in summary
    assert "Journal: Molecular genetics and metabolism (2025)" in summary
    assert "Abstract: BACKGROUND:" in summary  # structured abstract keeps its part labels
    assert summary.lower().count("l-serine") >= 5
    assert [e.url for e in evidence] == [f"https://pubmed.ncbi.nlm.nih.gov/{p}/" for p in ("41265180", "38380699", "38144875", "36758276", "31213567")]


def test_search_then_fetch_in_xml(recorded):
    PubMedTool().execute(query="GRIN2B loss of function L-serine", max_results=5)

    assert [c[0] for c in recorded] == ["esearch.fcgi", "efetch.fcgi"]
    assert recorded[0][1]["retmax"] == 5 and recorded[0][1]["sort"] == "relevance"
    assert recorded[1][1] == {"db": "pubmed", "id": "41265180,38380699,38144875,36758276,31213567", "retmode": "xml"}


def test_long_abstracts_are_kept_whole_up_to_the_cap(recorded):
    summary, _ = PubMedTool().execute(query="GRIN2B loss of function L-serine")

    block = summary.split("PMID: 38380699")[1].split("PMID:")[0]
    abstract = block.split("Abstract: ")[1]
    assert len(abstract.strip()) == pubmed.MAX_ABSTRACT_CHARS  # 2784 chars in the record, cut with "..."
    assert abstract.strip().endswith("...")


def test_max_results_is_clamped_to_ten(recorded):
    PubMedTool().execute(query="GRIN2B loss of function L-serine", max_results=50)

    assert recorded[0][1]["retmax"] == 10


def test_gene_mention_count_shows_whether_the_results_are_on_topic(recorded):
    on_topic, _ = PubMedTool().execute(query="GRIN2B loss of function L-serine", gene="GRIN2B")
    assert "Articles mentioning GRIN2B in title or abstract: 5 of 5" in on_topic

    # The same recorded articles against another gene: the count says the set missed.
    off_topic, _ = PubMedTool().execute(query="GRIN2B loss of function L-serine", gene="ANO10")
    assert "Articles mentioning ANO10 in title or abstract: 0 of 5" in off_topic

    without, _ = PubMedTool().execute(query="GRIN2B loss of function L-serine")
    assert "Articles mentioning" not in without

