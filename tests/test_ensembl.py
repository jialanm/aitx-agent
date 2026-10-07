"""Ensembl VEP adapter against replies recorded on 2026-10-07 (genomic form, refseq+numbers+hgvs)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import requests

from src.tools import ensembl
from src.tools.ensembl import ENSEMBL_LOCATION_URL, EnsemblTool

FIXTURES = Path(__file__).parent / "fixtures"

# (patient, genomic HGVS from the validator, current transcript)
EVAL_VARIANTS = [
    ("AITX-00001", "NC_000023.11:g.31260956_31729748del", "NM_004006.3"),
    ("AITX-00002", "NC_000023.11:g.31169542_31169543delinsTA", "NM_004006.3"),
    ("AITX-00003", "NC_000023.11:g.33020162A>G", "NM_004006.3"),
    ("AITX-00004", "NC_000002.12:g.240871433G>A", "NM_000030.3"),
    ("AITX-00005", "NC_000002.12:g.240868898dup", "NM_000030.3"),
    ("AITX-00007", "NC_000007.14:g.50539944C>T", "NM_001082971.2"),
    ("AITX-00008", "NC_000017.11:g.50194032C>T", "NM_000088.4"),
    ("AITX-00009", "NC_000023.11:g.48911634C>T", "NM_005660.3"),
    ("AITX-00011", "NC_000009.12:g.135784031G>A", "NM_020822.3"),
    ("AITX-00012", "NC_000012.12:g.13564483G>A", "NM_000834.5"),
    ("AITX-00013", "NC_000003.12:g.43600432del", "NM_018075.5"),
    ("AITX-00015", "NC_000019.10:g.35738488del", "NM_014727.3"),
    ("AITX-00016", "NC_000017.11:g.31235630T>C", "NM_001042492.3"),
]


class _Reply:
    def __init__(self, status, body, content_type="application/json"):
        self.status_code = status
        self.headers = {"content-type": content_type}
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def recorded(monkeypatch):
    """Serve the recorded VEP reply for the genomic notation in each request."""
    calls: list[dict] = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append({"url": url, "params": params, "timeout": timeout})
        notation = url.rsplit("/hgvs/", 1)[1]
        name = "ensembl_vep_" + re.sub(r"[^A-Za-z0-9_.-]", "_", notation) + ".json"
        return _Reply(200, json.loads((FIXTURES / name).read_text()))

    monkeypatch.setattr(ensembl.requests, "get", fake_get)
    return calls


def test_dmd_deletion_reads_the_patients_transcript_not_the_headline(recorded):
    summary, evidence = EnsemblTool().execute(
        genomic_hgvs="NC_000023.11:g.31260956_31729748del", transcript="NM_004006.3"
    )

    assert recorded[0]["params"] == {"refseq": 1, "numbers": 1, "hgvs": 1}
    assert recorded[0]["timeout"] == ensembl.VEP_TIMEOUT
    assert "on transcript NM_004006.3:" in summary
    assert "inframe_deletion" in summary
    assert "Exon: 52-63/79" in summary
    assert "Protein position: 2515-3095" in summary
    assert "HGVS protein: NP_003997.2:p.Thr2516_Ala3096del" in summary
    # The headline across all transcripts is reported separately, not as the consequence.
    assert "Most severe consequence across all transcripts: transcript_ablation" in summary
    assert "not returned" not in summary
    assert [e.url for e in evidence] == [
        ENSEMBL_LOCATION_URL.format(chrom="X", start=31260956, end=31729748)
    ]


@pytest.mark.parametrize("patient, genomic, transcript", EVAL_VARIANTS)
def test_every_eval_variant_yields_all_fields(recorded, caplog, patient, genomic, transcript):
    summary, evidence = EnsemblTool().execute(genomic_hgvs=genomic, transcript=transcript)

    assert f"on transcript {transcript}:" in summary
    assert "not returned" not in summary
    assert "VEP returned nothing" not in summary
    assert "lacks" not in caplog.text
    assert len(evidence) == 1 and evidence[0].url.startswith("https://www.ensembl.org/Homo_sapiens/Location/View?r=")


def test_timeout_is_retried_five_times_then_reported(monkeypatch, caplog):
    attempts: list[str] = []
    sleeps: list[int] = []

    def always_timeout(url, headers=None, params=None, timeout=None):
        attempts.append(url)
        raise requests.Timeout("simulated")

    monkeypatch.setattr(ensembl.requests, "get", always_timeout)
    monkeypatch.setattr(ensembl.time, "sleep", sleeps.append)

    summary, evidence = EnsemblTool().execute(genomic_hgvs="NC_000002.12:g.240871433G>A", transcript="NM_000030.3")

    assert len(attempts) == 5
    assert sleeps == [5, 5, 5, 5]
    assert summary == "VEP returned nothing for NC_000002.12:g.240871433G>A: timeout after 5 attempts"
    assert evidence == []
    assert caplog.text.count("VEP timeout") == 5


def test_http_error_is_logged_once_and_not_retried(monkeypatch, caplog):
    attempts: list[str] = []

    def error_page(url, headers=None, params=None, timeout=None):
        attempts.append(url)
        return _Reply(500, "<html>Error: 500</html>", content_type="text/html")

    monkeypatch.setattr(ensembl.requests, "get", error_page)

    summary, evidence = EnsemblTool().execute(genomic_hgvs="NC_000002.12:g.240871433G>A", transcript="NM_000030.3")

    assert len(attempts) == 1
    assert summary.startswith("VEP returned nothing for NC_000002.12:g.240871433G>A: HTTP 500")
    assert evidence == []
    assert "VEP error" in caplog.text
