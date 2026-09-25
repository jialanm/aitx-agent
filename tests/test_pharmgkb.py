"""PharmGKB adapter tests against recorded ClinPGx responses.

The fixtures under tests/fixtures/clinpgx_*_RYR1.json are verbatim API
responses recorded on 2026-09-25 (list endpoints truncated to the five
records the adapter reads). Serving them through a fake HTTP layer lets
the tests check request routing and evidence URLs without the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools import pharmgkb
from src.tools.pharmgkb import PharmGKBTool

FIXTURES = Path(__file__).parent / "fixtures"


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def recorded_api(monkeypatch):
    """Serve the RYR1 fixtures by endpoint and record every request made."""
    calls: list[tuple[str, dict]] = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        endpoint = url.rsplit("/", 1)[-1]
        path = FIXTURES / f"clinpgx_{endpoint}_RYR1.json"
        if not path.exists():
            return _Response(404, {})
        return _Response(200, json.loads(path.read_text()))

    monkeypatch.setattr(pharmgkb.requests, "get", fake_get)
    return calls


def test_requests_clinpgx_and_cites_clinpgx(recorded_api):
    summary, evidence = PharmGKBTool().execute(gene="RYR1")

    hosts = {url.split("/")[2] for url, _ in recorded_api}
    assert hosts == {"api.clinpgx.org"}
    assert "PharmGKB Gene: RYR1 (PA34896)" in summary
    assert evidence[0].url == "https://www.clinpgx.org/gene/PA34896"
    assert all(e.url.startswith("https://www.clinpgx.org/") for e in evidence)


def test_clinical_annotations_use_default_view(recorded_api):
    summary, _ = PharmGKBTool().execute(gene="RYR1")

    params = [p for url, p in recorded_api if url.endswith("/clinicalAnnotation")]
    assert params == [{"location.genes.accessionId": "PA34896"}]
    assert "## Clinical Annotations" in summary
    assert summary.count("| Drugs: desflurane") == 5
