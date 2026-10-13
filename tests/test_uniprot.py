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
        if "interpro" in url:
            return _Reply(FIXTURES / "interpro_P21359.json")
        return _Reply(FIXTURES / "uniprot_NF1_0.json")

    monkeypatch.setattr(uniprot.requests, "get", fake_get)
    return calls


def test_nf1_domains_and_position_verdict(recorded):
    summary, evidence = UniProtTool().execute(gene="NF1", protein_position=1243)

    assert "Protein: Neurofibromin (P21359)" in summary
    assert "Domains: Ras-GAP (1251-1482); CRAL-TRIO (1580-1738)" in summary
    assert "Position 1243: no annotated domain/site overlap" in summary  # UniProt's boundary; the key's GRD starts at 1198
    assert [e.url for e in evidence] == ["https://www.uniprot.org/uniprotkb/P21359/entry",
                                         "https://www.ebi.ac.uk/interpro/protein/UniProt/P21359/"]
    assert recorded[0]["timeout"] == uniprot.TIMEOUT == 30
    assert "gene_exact:NF1" in recorded[0]["params"]["query"]


def test_position_inside_a_domain_is_named(recorded):
    summary, _ = UniProtTool().execute(gene="NF1", protein_position=1300)

    assert "Position 1300 overlaps: " in summary and "Ras-GAP" in summary


def test_interpro_matches_covering_the_position_are_listed_with_their_boundaries(recorded):
    # AITX-00016: UniProt's Domain feature (PROSITE profile) starts at 1251; InterPro, CDD,
    # SMART and SUPERFAMILY place the Ras GTPase-activating domain from 1187-1198, covering 1243.
    summary, _ = UniProtTool().execute(gene="NF1", protein_position=1243)

    line = next(l for l in summary.splitlines() if l.startswith("InterPro matches covering position 1243"))
    assert "interpro IPR065505 Neurofibromin, Ras GTPase-activating domain (1198-1551)" in line
    assert "cdd cd05130 Ras-GTPase Activating Domain of neurofibromin (1198-1551)" in line
    assert "smart SM00323" in line and "(1187-1557)" in line
    assert "PS50018" not in line   # PROSITE's profile starts at 1251 and does not cover 1243
    assert recorded[1]["url"] == "https://www.ebi.ac.uk/interpro/api/entry/all/protein/uniprot/P21359"


def test_interpro_nearest_matches_when_none_covers_the_position(recorded):
    summary, _ = UniProtTool().execute(gene="NF1", protein_position=1560)

    line = next(l for l in summary.splitlines() if l.startswith("InterPro matches covering position 1560"))
    assert line.startswith("InterPro matches covering position 1560: none; nearest: ")
    assert "Ras GTPase-activating domain (1187-1557)" in line   # three residues away; CRAL-TRIO starts at 1580


def test_interpro_failure_is_reported_not_hidden(monkeypatch, caplog):
    def fake_get(url, params=None, timeout=None, headers=None):
        if "interpro" in url:
            raise uniprot.requests.ConnectionError("down")
        return _Reply(FIXTURES / "uniprot_NF1_0.json")
    monkeypatch.setattr(uniprot.requests, "get", fake_get)

    summary, evidence = UniProtTool().execute(gene="NF1", protein_position=1243)

    assert "InterPro domain matches: not available" in summary
    assert [e.url for e in evidence] == ["https://www.uniprot.org/uniprotkb/P21359/entry"]
    assert "InterPro lookup failed for P21359" in caplog.text

