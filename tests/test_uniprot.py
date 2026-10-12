"""UniProt adapter against the NF1 reply recorded on 2026-10-12."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools import uniprot
from src.tools.uniprot import UniProtTool

FIXTURES = Path(__file__).parent / "fixtures"


class _Reply:
    def __init__(self, path: Path):
        self.status_code, self.text, self.url = 200, path.read_text(), "https://rest.uniprot.org/x"

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        pass


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return _Reply(FIXTURES / "uniprot_NF1_0.json")

    monkeypatch.setattr(uniprot.requests, "get", fake_get)
    return calls


def test_nf1_domains_and_position_verdict(recorded):
    summary, evidence = UniProtTool().execute(gene="NF1", protein_position=1243)

    assert "Protein: Neurofibromin (P21359)" in summary
    assert "Domains: Ras-GAP (1251-1482); CRAL-TRIO (1580-1738)" in summary
    assert "Position 1243: no annotated domain/site overlap" in summary  # UniProt's boundary; the key's GRD starts at 1198
    assert [e.url for e in evidence] == ["https://www.uniprot.org/uniprotkb/P21359/entry"]
    assert recorded[0]["timeout"] == uniprot.TIMEOUT == 30
    assert "gene_exact:NF1" in recorded[0]["params"]["query"]


def test_position_inside_a_domain_is_named(recorded):
    summary, _ = UniProtTool().execute(gene="NF1", protein_position=1300)

    assert "Position 1300 overlaps: " in summary and "Ras-GAP" in summary
