"""openFDA adapter against the lumasiran label recorded on 2026-10-12."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools import openfda
from src.tools.openfda import OpenFDATool

FIXTURES = Path(__file__).parent / "fixtures"


class _Reply:
    def __init__(self, path: Path):
        self.status_code, self.text, self.url = 200, path.read_text(), "https://api.fda.gov/x"

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        pass


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params or {}))
        return _Reply(FIXTURES / "openfda_lumasiran_0.json")

    monkeypatch.setattr(openfda.requests, "get", fake_get)
    return calls


def test_label_is_cited_by_its_own_dailymed_page(recorded):
    summary, evidence = OpenFDATool().execute(drug="lumasiran")

    assert "Indications:" in summary and "primary hyperoxaluria type 1" in summary
    assert [e.url for e in evidence] == ["https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=16985a31-f5e4-4557-9266-fc78d4bc5055"]
    assert "search.cfm" not in evidence[0].url


def test_empty_label_result_states_the_coverage_and_names_pubmed(monkeypatch):
    """32B run 4, AITX-00002: ataluren has no United States label, and the bare
    "No FDA drug labels found" was read as evidence that it does not apply."""
    class _Empty:
        status_code, text, url = 200, '{"results": []}', "https://api.fda.gov/x"

        def json(self):
            return {"results": []}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(openfda.requests, "get", lambda *a, **k: _Empty())

    summary, evidence = OpenFDATool().execute(drug="Ataluren")

    assert summary.startswith("No FDA drug labels found for: Ataluren.")
    assert "openFDA holds only drugs with a United States label" in summary
    assert "not evidence that it does not exist" in summary
    assert "Search PubMed for Ataluren with the gene or condition" in summary
    assert evidence == []

