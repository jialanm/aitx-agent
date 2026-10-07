"""validate_variant() against recorded VariantValidator replies (2026-10-07)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import requests

from src import variant_validation as vv
from src.variant_validation import validate_variant

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_for(url: str) -> Path:
    # Fixtures are named after the request path: <variant>/<transcript>.
    path = url.split(f"/{vv.GENOME_BUILD}/", 1)[1]
    return FIXTURES / ("variantvalidator_" + re.sub(r"[^A-Za-z0-9_.-]", "_", path) + ".json")


class _Reply:
    def __init__(self, body):
        self.status_code = 200
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def recorded(monkeypatch):
    """Serve the recorded reply for each request and remember the requests."""
    urls: list[str] = []

    def fake_get(url, headers=None, timeout=None):
        urls.append(url)
        return _Reply(json.loads(_fixture_for(url).read_text()))

    monkeypatch.setattr(vv.requests, "get", fake_get)
    return urls


def test_current_transcript_needs_one_call(recorded):
    # AITX-00004: AGXT missense on a transcript version the validator still considers current.
    v = validate_variant("NM_000030.3", "c.508G>A")

    assert len(recorded) == 1
    assert v.valid
    assert v.warnings == []
    assert v.transcript_variant == "NM_000030.3:c.508G>A"
    assert v.genomic_variant == "NC_000002.12:g.240871433G>A"
    assert v.gene_symbol == "AGXT"
    assert v.superseded_transcript == ""


def test_outdated_transcript_is_rewritten_via_genomic_position(recorded):
    v = validate_variant("NM_004006.2", "c.7544_9286del")

    assert len(recorded) == 2
    assert recorded[1].endswith("/NC_000023.11:g.31260956_31729748del/NM_004006.3")
    assert v.valid
    assert v.transcript_variant == "NM_004006.3:c.7544_9286del"
    assert v.superseded_transcript == "NM_004006.2"
    assert v.source_url == recorded[1]


@pytest.mark.parametrize("transcript, cdna", [
    ("NM_004006.2", "c.7544_9286del"), ("NM_004006.2", "c.10453_10454delinsTA"), ("NM_004006.2", "c.70T>C"),
    ("NM_000030.3", "c.508G>A"), ("NM_000030.3", "c.33dup"), ("NM_001082971.2", "c.286G>A"),
    ("NM_000088.4", "c.1678G>A"), ("NM_005660.3", "c.3G>A"), ("NM_020822.3", "c.2849G>A"),
    ("NM_000834.5", "c.2755C>T"), ("NM_018075.5", "c.289del"), ("NM_014727.3", "c.8079delC"),
    ("NM_001042492.3", "c.3728T>C"),
])
def test_every_eval_variant_yields_all_fields(recorded, transcript, cdna):
    v = validate_variant(transcript, cdna)

    assert v.valid, v.warnings
    assert v.transcript_variant.startswith(transcript.split(".")[0])
    assert v.genomic_variant.startswith("NC_")
    assert v.gene_symbol


def test_timeout_is_retried_five_times_then_reported(monkeypatch, caplog):
    attempts: list[str] = []
    sleeps: list[int] = []

    def always_timeout(url, headers=None, timeout=None):
        attempts.append(url)
        raise requests.Timeout("simulated")

    monkeypatch.setattr(vv.requests, "get", always_timeout)
    monkeypatch.setattr(vv.time, "sleep", sleeps.append)

    v = validate_variant("NM_000030.3", "c.508G>A")

    assert len(attempts) == 5
    assert sleeps == [5, 5, 5, 5]
    assert not v.valid
    assert v.warnings == ["validator unavailable: timeout after retries"]
    assert caplog.text.count("VariantValidator timeout") == 5
