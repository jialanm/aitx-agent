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
        if "/vep/human/hgvs/" in url:
            notation = url.rsplit("/hgvs/", 1)[1]
            name = "ensembl_vep_" + re.sub(r"[^A-Za-z0-9_.-]", "_", notation) + ".json"
        elif "/xrefs/symbol/homo_sapiens/" in url:
            name = "ensembl_xrefs_" + url.rsplit("/", 1)[1] + ".json"
        elif "/lookup/id/" in url:
            name = "ensembl_lookup_" + url.rsplit("/", 1)[1] + ".json"
        else:
            return _Reply(404, {"error": "no fixture for " + url})
        path = FIXTURES / name
        if not path.exists():
            # Recorded only for the transcripts the exon-length tests use.
            return _Reply(404, {"error": "no fixture"})
        return _Reply(200, json.loads(path.read_text()))

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


def test_server_error_is_retried_then_succeeds(monkeypatch, caplog):
    """Ensembl's transient 500 page, then the recorded reply on the second try."""
    attempts: list[str] = []
    sleeps: list[int] = []
    recorded = json.loads((FIXTURES / "ensembl_vep_NC_000002.12_g.240871433G_A.json").read_text())

    def flaky(url, headers=None, params=None, timeout=None):
        if "/vep/human/hgvs/" not in url:
            return _Reply(200, [])  # exon-map lookups are not under test here
        attempts.append(url)
        if len(attempts) == 1:
            return _Reply(500, "<html>Error: 500</html>", content_type="text/html")
        return _Reply(200, recorded)

    monkeypatch.setattr(ensembl.requests, "get", flaky)
    monkeypatch.setattr(ensembl.time, "sleep", sleeps.append)

    summary, evidence = EnsemblTool().execute(genomic_hgvs="NC_000002.12:g.240871433G>A", transcript="NM_000030.3")

    assert len(attempts) == 2
    assert sleeps == [5]
    assert "Consequence: missense_variant" in summary
    assert caplog.text.count("VEP server error") == 1


def test_server_error_five_times_is_reported(monkeypatch, caplog):
    attempts: list[str] = []
    monkeypatch.setattr(ensembl.time, "sleep", lambda s: None)

    def error_page(url, headers=None, params=None, timeout=None):
        attempts.append(url)
        return _Reply(500, "<html>Error: 500</html>", content_type="text/html")

    monkeypatch.setattr(ensembl.requests, "get", error_page)

    summary, evidence = EnsemblTool().execute(genomic_hgvs="NC_000002.12:g.240871433G>A", transcript="NM_000030.3")

    assert len(attempts) == 5
    assert summary == "VEP returned nothing for NC_000002.12:g.240871433G>A: HTTP 500 after 5 attempts"
    assert evidence == []


def test_client_error_is_not_retried(monkeypatch, caplog):
    attempts: list[str] = []

    def bad_request(url, headers=None, params=None, timeout=None):
        attempts.append(url)
        return _Reply(400, {"error": "Unable to parse HGVS notation"})

    monkeypatch.setattr(ensembl.requests, "get", bad_request)

    summary, evidence = EnsemblTool().execute(genomic_hgvs="NC_000002.12:g.240871433G>A", transcript="NM_000030.3")

    assert len(attempts) == 1
    assert summary.startswith("VEP returned nothing for NC_000002.12:g.240871433G>A: HTTP 400")
    assert "VEP error" in caplog.text


def test_duplication_link_puts_the_smaller_coordinate_first(recorded):
    # AITX-00005: VEP reports the inserted base as start 240868898, end 240868897.
    summary, evidence = EnsemblTool().execute(genomic_hgvs="NC_000002.12:g.240868898dup", transcript="NM_000030.3")

    assert evidence[0].url == ENSEMBL_LOCATION_URL.format(chrom="2", start=240868897, end=240868898)


def test_exon_coding_length_for_ano10_matches_the_reference(recorded):
    # AITX-00013/00014: exon 3 of NM_018075.5 encodes 66 aa, 10.0% of a 660-aa protein.
    summary, _ = EnsemblTool().execute(genomic_hgvs="NC_000003.12:g.43600432del", transcript="NM_018075.5")

    assert "Exon 3 of 13 (ENST00000292246): 198 coding bases = 66 codons (in frame), 10.0% of the 1983-base coding sequence; protein length 660 aa" in summary
    assert [c["url"].rsplit("/", 2)[-2:] for c in recorded[1:]] == [
        ["homo_sapiens", "NM_018075.5"], ["id", "ENST00000292246"]
    ]
    assert recorded[2]["params"] == {"expand": 1}


def test_exon_range_sums_the_deleted_exons_for_dmd(recorded):
    # AITX-00001: exons 52-63 total 1744 bases, one short of a multiple of three, so
    # the deletion breaks the frame; adding exon 51 (233 bases) restores it, which is
    # why the reference answer is the exon-51-skipping drug.
    summary, _ = EnsemblTool().execute(genomic_hgvs="NC_000023.11:g.31260956_31729748del", transcript="NM_004006.3")

    line = next(l for l in summary.splitlines() if l.startswith("Exons 52-63 of 79"))
    assert "1744 coding bases = 581 codons + 1 bases (out of frame by 1)" in line
    assert "protein length 3685 aa" in line


def test_exon_length_is_marked_not_returned_when_ensembl_has_no_transcript(monkeypatch, caplog):
    """VEP answers, but the cross-reference lookup finds no Ensembl transcript: say so, never guess."""
    vep = json.loads((FIXTURES / "ensembl_vep_NC_000002.12_g.240871433G_A.json").read_text())

    def get(url, headers=None, params=None, timeout=None):
        if "/vep/human/hgvs/" in url:
            return _Reply(200, vep)
        return _Reply(200, [])  # xrefs: empty list for both the versioned and the bare accession

    monkeypatch.setattr(ensembl.requests, "get", get)

    summary, _ = EnsemblTool().execute(genomic_hgvs="NC_000002.12:g.240871433G>A", transcript="NM_000030.3")

    assert "Exon coding length: not returned (no Ensembl transcript found for NM_000030.3)" in summary
    assert "Exon: 4/11" in summary
    assert "No Ensembl transcript for NM_000030.3" in caplog.text
