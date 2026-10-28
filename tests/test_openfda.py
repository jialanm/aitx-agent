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



class TestCandidateChoice:
    """openFDA is asked for several labels and the adapter chooses, after a
    single result gave AITX-00012 a homeopathic preparation for L-serine
    (32B run 5, 2026-10-28). Fixtures recorded 2026-10-28 from
    api.fda.gov/drug/label.json, search openfda.generic_name:"<drug>"."""

    def _reply(self, monkeypatch, fixture):
        seen = []

        def fake_get(url, params=None, timeout=None):
            seen.append(dict(params or {}))
            return _Reply(FIXTURES / fixture)

        monkeypatch.setattr(openfda.requests, "get", fake_get)
        return seen

    def test_several_candidates_are_requested(self, monkeypatch):
        seen = self._reply(monkeypatch, "openfda_label_memantine.json")

        OpenFDATool().execute(drug="memantine")

        assert seen[0]["limit"] == openfda.SEARCH_LIMIT > 1

    def test_a_combination_label_says_the_drug_is_one_ingredient_of_many(self, monkeypatch):
        """Every L-serine label on openFDA is an over-the-counter combination
        product; the summary states both facts instead of presenting it as
        the drug's own label."""
        self._reply(monkeypatch, "openfda_label_L-serine.json")

        summary, evidence = OpenFDATool().execute(drug="L-serine")

        assert "Product type (openFDA): HUMAN OTC DRUG" in summary
        assert "The searched name L-serine is 1 of 29 ingredient names on this label." in summary
        assert evidence and evidence[0].url.startswith("https://dailymed.nlm.nih.gov/")

    def test_a_single_ingredient_prescription_label_carries_no_ingredient_note(self, monkeypatch):
        self._reply(monkeypatch, "openfda_label_memantine.json")

        summary, _ = OpenFDATool().execute(drug="memantine")

        assert "Product type (openFDA): HUMAN PRESCRIPTION DRUG" in summary
        assert "ingredient names on this label" not in summary
        assert "Indications:" in summary

    def test_the_drug_named_alone_outranks_a_combination_product(self):
        """Ranking over the recorded records themselves, lower sorts first."""
        import json
        combo = json.loads((FIXTURES / "openfda_label_L-serine.json").read_text())["results"][0]
        single = json.loads((FIXTURES / "openfda_label_memantine.json").read_text())["results"][0]

        assert OpenFDATool._rank(single, "memantine") < OpenFDATool._rank(combo, "memantine")
        assert OpenFDATool._rank(combo, "L-serine") > OpenFDATool._rank(single, "memantine")
