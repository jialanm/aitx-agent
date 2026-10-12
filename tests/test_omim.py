"""OMIM adapter against the search and summary recorded for SLC35A2 on 2026-10-11."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from src.tools import base, omim
from src.tools.omim import OMIMTool

FIXTURES = Path(__file__).parent / "fixtures"


class _Reply:
    def __init__(self, path: Path):
        self.status_code, self.text, self.url = 200, path.read_text(), "https://eutils.ncbi.nlm.nih.gov/x"

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        pass


class _Empty:
    """A search with no hits, for the phenotype and MedGen branches not under test."""
    status_code, url = 200, "https://eutils.ncbi.nlm.nih.gov/x"
    text = '{"esearchresult": {"idlist": []}, "result": {"uids": []}}'

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        pass


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params or {}))
        if params.get("db") == "omim" and params.get("term") == "SLC35A2[gene symbol]":
            return _Reply(FIXTURES / "omim_esearch_SLC35A2_gene.json")
        if params.get("db") == "omim" and params.get("id") == "314375":
            return _Reply(FIXTURES / "omim_esummary_SLC35A2.json")
        return _Empty()

    monkeypatch.setattr(omim.requests, "get", fake_get)
    return calls


def test_gene_entry_is_returned_with_its_omim_url(recorded):
    summary, evidence = OMIMTool().execute(gene="SLC35A2")

    assert "MIM #314375" in summary
    assert "SOLUTE CARRIER FAMILY 35 (UDP-GALACTOSE TRANSPORTER), MEMBER 2; SLC35A2" in summary
    assert [e.url for e in evidence] == ["https://www.omim.org/entry/314375"]


def test_gene_search_goes_through_the_shared_retry_with_the_key(recorded, monkeypatch):
    monkeypatch.setenv(base.NCBI_API_KEY_ENV, "0123456789abcdef0123456789abcdef0123")

    OMIMTool().execute(gene="SLC35A2")

    assert recorded[0]["term"] == "SLC35A2[gene symbol]"
    assert all(p.get("api_key") == "0123456789abcdef0123456789abcdef0123" for p in recorded)


def test_a_bug_in_the_gene_search_is_surfaced_not_read_as_no_entries(monkeypatch, caplog):
    def broken(*a, **k):
        raise NameError("name 'ncbi_get' is not defined")

    monkeypatch.setattr(omim, "ncbi_get", broken)

    with pytest.raises(NameError):
        OMIMTool().execute(gene="SLC35A2")

    assert "OMIM gene search failed for SLC35A2: NameError" in caplog.text
