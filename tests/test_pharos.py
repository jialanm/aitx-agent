"""PHAROS adapter against target replies recorded on 2026-10-08 for the eval genes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import requests

from src.tools import pharos
from src.tools.pharos import PharosTool

FIXTURES = Path(__file__).parent / "fixtures"
EVAL_GENES = ["DMD", "AGXT", "DDC", "COL1A1", "SLC35A2", "KCNT1", "GRIN2B", "ANO10", "KMT2B", "NF1"]


class _Reply:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json["query"])
        gene = re.search(r'sym:"([^"]+)"', json["query"]).group(1)
        return _Reply(200, __import__("json").loads((FIXTURES / f"pharos_target_{gene}.json").read_text()))

    monkeypatch.setattr(pharos.requests, "post", fake_post)
    return calls


def test_dmd_lists_the_exon_skipping_drugs_with_mechanism(recorded):
    summary, evidence = PharosTool().execute(gene="DMD")

    assert "Target development level: Tclin (an approved drug acts on this protein)" in summary
    for drug in ("golodirsen", "eteplirsen", "viltolarsen", "casimersen"):
        assert f"- {drug}:" in summary
    assert "- golodirsen: antisense inhibitor" in summary
    assert "Most associated diseases: Duchenne muscular dystrophy;" in summary
    assert [e.url for e in evidence] == ["https://pharos.nih.gov/targets/P11532"]


def test_agxt_has_no_approved_drug_and_says_so(recorded):
    summary, evidence = PharosTool().execute(gene="AGXT")

    assert "Target development level: Tbio (biology known, no potent binder)" in summary
    assert "Approved drugs acting on this target: none in PHAROS" in summary
    assert "primary hyperoxaluria type 1" in summary
    assert len(evidence) == 1


@pytest.mark.parametrize("gene", EVAL_GENES)
def test_every_eval_gene_yields_level_family_and_diseases(recorded, gene):
    summary, evidence = PharosTool().execute(gene=gene)

    assert summary.startswith(f"PHAROS target {gene} (")
    assert "Target development level: T" in summary
    assert "Protein family:" in summary
    assert "Most associated diseases:" in summary
    assert len(evidence) == 1 and evidence[0].url.startswith("https://pharos.nih.gov/targets/")


def test_timeout_is_retried_then_reported(monkeypatch, caplog):
    attempts = []
    monkeypatch.setattr(pharos.time, "sleep", lambda s: None)

    def always_timeout(url, json=None, timeout=None):
        attempts.append(url)
        raise requests.Timeout("simulated")

    monkeypatch.setattr(pharos.requests, "post", always_timeout)

    summary, evidence = PharosTool().execute(gene="DMD")

    assert len(attempts) == 5
    assert summary == "PHAROS returned nothing for DMD: timeout after 5 attempts"
    assert evidence == []
    assert caplog.text.count("PHAROS timeout") == 5


def test_query_error_is_reported_not_hidden(monkeypatch, caplog):
    monkeypatch.setattr(pharos.requests, "post",
                        lambda url, json=None, timeout=None: _Reply(400, {"errors": [{"message": "Cannot query field"}]}))

    summary, evidence = PharosTool().execute(gene="DMD")

    assert summary.startswith("PHAROS returned nothing for DMD: query error Cannot query field")
    assert evidence == []
    assert "PHAROS query error" in caplog.text
