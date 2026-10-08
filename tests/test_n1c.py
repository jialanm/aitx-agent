"""N1C guidelines tool: built-in rules, and paper sections from the PMC XML recorded 2026-10-08."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from src.tools import n1c
from src.tools.n1c import N1CGuidelinesTool

FIXTURES = Path(__file__).parent / "fixtures"


class _Reply:
    def __init__(self, status, text):
        self.status_code, self.text = status, text


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params or {}))
        return _Reply(200, (FIXTURES / "pmc_efetch_PMC12120168.xml").read_text())

    monkeypatch.setattr(n1c.requests, "get", fake_get)
    return calls


def test_overview_states_strategies_verdicts_and_exon_rules_without_network(monkeypatch):
    monkeypatch.setattr(n1c.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network expected")))

    summary, evidence = N1CGuidelinesTool().execute()

    for term in ("Splice correction", "Canonical exon skipping", "Transcript knockdown", "Upregulation from the wild-type allele",
                 "eligible (", "likely eligible", "unlikely eligible", "not eligible", "unable to assess",
                 "first coding exon", "last coding exon", "out of frame", "stop codon", "one coding exon", "haploinsufficient"):
        assert term in summary, term
    assert [e.url for e in evidence] == ["https://doi.org/10.1016/j.ajhg.2025.02.017", "http://eligibilitycalculator.n1collaborative.org/"]


def test_classification_section_returns_the_papers_text(recorded):
    summary, evidence = N1CGuidelinesTool().execute(section="classification")

    assert summary.startswith("From Cheerie et al. 2025 (PMC12120168, CC BY 4.0), section 'Classification terms':")
    assert "five tiers for all classifications: eligible, likely eligible, unlikely eligible, not eligible, and unable to assess" in summary
    assert "single-exon genes in the context of exon-skipping ASOs" in summary
    assert recorded == [{"db": "pmc", "id": "12120168", "retmode": "xml"}]
    assert [e.url for e in evidence] == ["https://doi.org/10.1016/j.ajhg.2025.02.017"]


def test_paper_is_fetched_once_per_tool(recorded):
    tool = N1CGuidelinesTool()
    tool.execute(section="purpose")
    tool.execute(section="structure")

    assert len(recorded) == 1


def test_unknown_section_is_refused_with_the_list():
    summary, _ = N1CGuidelinesTool().execute(section="appendix")
    assert summary.startswith("N1C guidelines: unknown section 'appendix'. Available: introduction, purpose")


def test_fetch_failure_is_reported(monkeypatch, caplog):
    monkeypatch.setattr(n1c.time, "sleep", lambda s: None)
    monkeypatch.setattr(n1c.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout("simulated")))

    summary, evidence = N1CGuidelinesTool().execute(section="discussion")

    assert summary == "N1C guidelines: could not fetch the paper's text from PubMed Central: timeout after 5 attempts"
    assert caplog.text.count("PMC efetch timeout") == 5
