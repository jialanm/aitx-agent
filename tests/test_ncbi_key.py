"""NCBI API key handling for the E-utilities adapters."""

from src.tools.base import NCBI_API_KEY_ENV, NCBI_KEY_PLACEHOLDER, ncbi_params, redact_ncbi_key


def test_params_are_unchanged_when_no_key_is_set(monkeypatch):
    # Every machine without the key, including CI and this test run, takes
    # this path; it must send the request exactly as the adapter built it.
    monkeypatch.delenv(NCBI_API_KEY_ENV, raising=False)

    assert ncbi_params(db="pubmed", term="CFTR") == {"db": "pubmed", "term": "CFTR"}


def test_key_is_removed_from_error_text(monkeypatch):
    fake_key = "0123456789abcdef0123456789abcdef0123"
    monkeypatch.setenv(NCBI_API_KEY_ENV, fake_key)
    message = (
        "429 Client Error: Too Many Requests for url: https://eutils.ncbi.nlm.nih.gov/"
        f"entrez/eutils/esearch.fcgi?db=pubmed&term=CFTR&api_key={fake_key}"
    )

    redacted = redact_ncbi_key(message)

    assert fake_key not in redacted
    assert redacted.endswith(f"&api_key={NCBI_KEY_PLACEHOLDER}")


# --- adapters -------------------------------------------------------------
#
# The fixtures are verbatim E-utilities replies recorded on 2026-09-25 with
# no key set. A stand-in for requests.get serves them by endpoint and records
# every request, so the tests can check what each adapter sent.

import json
from pathlib import Path

import pytest
import requests

from src.tools import pubmed

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_KEY = "0123456789abcdef0123456789abcdef0123"


class _Recorded:
    def __init__(self, path: Path):
        self.status_code = 200
        self._body = json.loads(path.read_text())

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


class _RateLimited:
    """A recorded HTTP 429; like requests, the error text carries the URL."""

    def __init__(self, url: str, params: dict):
        self.status_code = 429
        self.url = requests.Request("GET", url, params=params).prepare().url
        self._body = json.loads((FIXTURES / "eutils_429_rate_limit.json").read_text())

    def json(self):
        return self._body

    def raise_for_status(self):
        raise requests.HTTPError(
            f"429 Client Error: Too Many Requests for url: {self.url}"
        )


@pytest.fixture
def eutils(monkeypatch):
    """Serve recorded replies and record requests; optionally 429 the first."""
    calls: list[tuple[str, dict]] = []
    state = {"rate_limit_first": False}

    def stand_in(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        if state["rate_limit_first"] and len(calls) == 1:
            return _RateLimited(url, params)
        endpoint = url.rsplit("/", 1)[-1].removesuffix(".fcgi")
        return _Recorded(FIXTURES / f"{params['db']}_{endpoint}_CFTR_F508del_Trikafta.json")

    monkeypatch.setattr(pubmed.requests, "get", stand_in)
    return calls, state


def test_pubmed_sends_key_on_every_request(eutils, monkeypatch):
    calls, _ = eutils
    monkeypatch.setenv(NCBI_API_KEY_ENV, FAKE_KEY)

    summary, evidence = pubmed.PubMedTool().execute(query="CFTR F508del Trikafta treatment")

    assert [u.rsplit("/", 1)[-1] for u, _ in calls] == ["esearch.fcgi", "esummary.fcgi"]
    assert all(p["api_key"] == FAKE_KEY for _, p in calls)
    assert evidence[0].url == "https://pubmed.ncbi.nlm.nih.gov/34268058/"


def test_pubmed_sends_no_key_when_unset(eutils, monkeypatch):
    calls, _ = eutils
    monkeypatch.delenv(NCBI_API_KEY_ENV, raising=False)

    pubmed.PubMedTool().execute(query="CFTR F508del Trikafta treatment")

    assert len(calls) == 2
    assert not any("api_key" in p for _, p in calls)


def test_pubmed_keeps_key_out_of_rate_limit_error(eutils, monkeypatch):
    calls, state = eutils
    monkeypatch.setenv(NCBI_API_KEY_ENV, FAKE_KEY)
    state["rate_limit_first"] = True

    summary, evidence = pubmed.PubMedTool().execute(query="CFTR F508del Trikafta treatment")

    assert summary.startswith("PubMed API error: 429")
    assert FAKE_KEY not in summary
    assert NCBI_KEY_PLACEHOLDER in summary
    assert evidence == []
