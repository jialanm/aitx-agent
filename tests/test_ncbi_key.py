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

from src.tools import base, clinvar, genereviews, pubmed

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_KEY = "0123456789abcdef0123456789abcdef0123"


class _Recorded:
    """A recorded reply: JSON for .json fixtures, plain text otherwise."""

    def __init__(self, path: Path):
        self.status_code = 200
        self.text = path.read_text()
        self.url = "https://eutils.ncbi.nlm.nih.gov/recorded"
        self._body = json.loads(self.text) if path.suffix == ".json" else None

    def json(self):
        if self._body is None:
            raise ValueError("fixture is not JSON")
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


class _Server:
    """Serves recorded replies by endpoint and records every request.

    replies maps an endpoint name ("esearch", "esummary", ...) to one
    fixture file, or to a list of files handed out in order so a test can
    make a first search miss and a second one hit.
    """

    def __init__(self, replies: dict[str, str | list[str]]):
        self.replies = {k: list(v) if isinstance(v, list) else [v] for k, v in replies.items()}
        self.calls: list[tuple[str, dict]] = []
        self.rate_limit_first = False

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        # rate_limit_first: every attempt of the first request is answered
        # 429, so the shared retry policy spends its tries and reports.
        if self.rate_limit_first and all(u == url for u, _ in self.calls):
            return _RateLimited(url, params)
        endpoint = url.rsplit("/", 1)[-1].removesuffix(".fcgi")
        queue = self.replies[endpoint]
        name = queue.pop(0) if len(queue) > 1 else queue[0]
        return _Recorded(FIXTURES / name)

    def endpoints(self) -> list[str]:
        return [u.rsplit("/", 1)[-1].removesuffix(".fcgi") for u, _ in self.calls]


PUBMED_REPLIES = {
    "esearch": "pubmed_esearch_GRIN2B_LoF_Lserine.json",
    "efetch": "pubmed_efetch_GRIN2B_LoF_Lserine.xml",
}


@pytest.fixture
def eutils(monkeypatch):
    server = _Server(PUBMED_REPLIES)
    monkeypatch.setattr(pubmed.requests, "get", server.get)
    return server


def test_pubmed_sends_key_on_every_request(eutils, monkeypatch):
    monkeypatch.setenv(NCBI_API_KEY_ENV, FAKE_KEY)

    summary, evidence = pubmed.PubMedTool().execute(query="GRIN2B loss of function L-serine")

    assert eutils.endpoints() == ["esearch", "efetch"]
    assert all(p["api_key"] == FAKE_KEY for _, p in eutils.calls)
    assert evidence[0].url == "https://pubmed.ncbi.nlm.nih.gov/41265180/"


def test_pubmed_sends_no_key_when_unset(eutils, monkeypatch):
    monkeypatch.delenv(NCBI_API_KEY_ENV, raising=False)

    pubmed.PubMedTool().execute(query="GRIN2B loss of function L-serine")

    assert len(eutils.calls) == 2
    assert not any("api_key" in p for _, p in eutils.calls)


def test_pubmed_keeps_key_out_of_rate_limit_error(eutils, monkeypatch):
    monkeypatch.setenv(NCBI_API_KEY_ENV, FAKE_KEY)
    eutils.rate_limit_first = True
    monkeypatch.setattr(base.time, "sleep", lambda s: None)

    summary, evidence = pubmed.PubMedTool().execute(query="GRIN2B loss of function L-serine")

    assert summary.startswith("PubMed API error: 429 Too Many Requests after 5 attempts")
    assert len(eutils.calls) == 5
    assert FAKE_KEY not in summary
    assert NCBI_KEY_PLACEHOLDER in summary
    assert evidence == []


CLINVAR_REPLIES = {
    "esearch": "clinvar_esearch_CFTR_c1521_1523del.json",
    "esummary": "clinvar_esummary_CFTR_c1521_1523del.json",
}


@pytest.fixture
def clinvar_eutils(monkeypatch):
    server = _Server(CLINVAR_REPLIES)
    monkeypatch.setattr(clinvar.requests, "get", server.get)
    return server


def test_clinvar_sends_key_on_every_request(clinvar_eutils, monkeypatch):
    monkeypatch.setenv(NCBI_API_KEY_ENV, FAKE_KEY)
    # First search misses so the broader fallback search runs as well.
    clinvar_eutils.replies["esearch"] = [
        "clinvar_esearch_no_hits.json",
        "clinvar_esearch_CFTR_c1521_1523del.json",
    ]

    summary, evidence = clinvar.ClinVarTool().execute(gene="CFTR", variant="c.1521_1523del")

    assert clinvar_eutils.endpoints() == ["esearch", "esearch", "esummary"]
    assert all(p["api_key"] == FAKE_KEY for _, p in clinvar_eutils.calls)
    assert evidence[0].url == "https://www.ncbi.nlm.nih.gov/clinvar/variation/1705126/"


def test_clinvar_sends_no_key_when_unset(clinvar_eutils, monkeypatch):
    monkeypatch.delenv(NCBI_API_KEY_ENV, raising=False)

    clinvar.ClinVarTool().execute(gene="CFTR", variant="c.1521_1523del")

    assert len(clinvar_eutils.calls) == 2
    assert not any("api_key" in p for _, p in clinvar_eutils.calls)


def test_clinvar_keeps_key_out_of_rate_limit_error(clinvar_eutils, monkeypatch):
    monkeypatch.setenv(NCBI_API_KEY_ENV, FAKE_KEY)
    clinvar_eutils.rate_limit_first = True
    monkeypatch.setattr(base.time, "sleep", lambda s: None)

    summary, evidence = clinvar.ClinVarTool().execute(gene="CFTR", variant="c.1521_1523del")

    assert summary.startswith("ClinVar API error: 429 Too Many Requests after 5 attempts")
    assert len(clinvar_eutils.calls) == 5
    assert FAKE_KEY not in summary
    assert NCBI_KEY_PLACEHOLDER in summary
    assert evidence == []


def _genereviews_server(monkeypatch, fail_abstract=None):
    """Serve the recorded index files and chapter abstract; record every request."""
    calls: list[tuple[str, dict]] = []

    def get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        if url.endswith("NBKid_shortname_genesymbol.txt"):
            return _Recorded(FIXTURES / "genereviews_NBKid_shortname_genesymbol.txt")
        if url.endswith("GRtitle_shortname_NBKid.txt"):
            return _Recorded(FIXTURES / "genereviews_GRtitle_shortname_NBKid.txt")
        if fail_abstract is not None:
            raise fail_abstract(url, params)
        return _Recorded(FIXTURES / f"genereviews_abstract_{params['id']}.txt")

    monkeypatch.setattr(genereviews.requests, "get", get)
    return calls


def test_genereviews_sends_key_on_the_pubmed_fetch_only(monkeypatch):
    monkeypatch.setenv(NCBI_API_KEY_ENV, FAKE_KEY)
    calls = _genereviews_server(monkeypatch)

    summary, evidence = genereviews.GeneReviewsTool().execute(gene="AGXT")

    index_calls = [p for u, p in calls if u.endswith(".txt")]
    efetch_calls = [p for u, p in calls if "efetch" in u]
    assert len(index_calls) == 2 and not any("api_key" in p for p in index_calls)  # FTP files take no key
    assert efetch_calls == [{"db": "pubmed", "id": "20301460", "rettype": "abstract", "retmode": "text", "api_key": FAKE_KEY}]
    assert evidence[0].url == "https://www.ncbi.nlm.nih.gov/books/NBK1283/"


def test_genereviews_sends_no_key_when_unset(monkeypatch):
    monkeypatch.delenv(NCBI_API_KEY_ENV, raising=False)
    calls = _genereviews_server(monkeypatch)

    genereviews.GeneReviewsTool().execute(gene="AGXT")

    assert not any("api_key" in p for _, p in calls)


def test_genereviews_keeps_key_out_of_request_errors(monkeypatch):
    monkeypatch.setenv(NCBI_API_KEY_ENV, FAKE_KEY)

    def connection_error(url, params):
        full = requests.Request("GET", url, params=params).prepare().url
        return requests.ConnectionError(f"connection refused for url: {full}")

    _genereviews_server(monkeypatch, fail_abstract=connection_error)

    summary, evidence = genereviews.GeneReviewsTool().execute(gene="AGXT")

    assert summary.startswith("GeneReviews returned nothing for NBK1283: request failed")
    assert FAKE_KEY not in summary
    assert NCBI_KEY_PLACEHOLDER in summary
    assert evidence == []


# --- shared retry policy ---------------------------------------------------


def test_ncbi_get_retries_a_rate_limit_once_then_returns(monkeypatch):
    sleeps: list[float] = []
    replies = [_RateLimited("https://x/efetch.fcgi", {"db": "pubmed"}),
               _Recorded(FIXTURES / "pubmed_esearch_GRIN2B_LoF_Lserine.json")]
    monkeypatch.setattr(base.time, "sleep", sleeps.append)
    monkeypatch.setattr(requests, "get", lambda url, params=None, timeout=None: replies.pop(0))

    resp = base.ncbi_get(url="https://x/efetch.fcgi", params={"db": "pubmed"}, timeout=10, label="T")

    assert resp.status_code == 200
    assert sleeps == [base.NCBI_RATE_LIMIT_PAUSE_SECONDS]


def test_ncbi_get_raises_after_five_rate_limits(monkeypatch, caplog):
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    monkeypatch.setattr(requests, "get", lambda url, params=None, timeout=None: _RateLimited(url, params))

    with pytest.raises(requests.HTTPError) as err:
        base.ncbi_get(url="https://x/efetch.fcgi", params={}, timeout=10, label="T")

    assert "429 Too Many Requests after 5 attempts" in str(err.value)
    assert caplog.text.count("T rate limited") == 5


def test_ncbi_get_returns_other_client_errors_at_once(monkeypatch):
    calls: list[str] = []

    class _Bad:
        status_code = 400

    monkeypatch.setattr(requests, "get", lambda url, params=None, timeout=None: calls.append(url) or _Bad())

    resp = base.ncbi_get(url="https://x/esearch.fcgi", params={}, timeout=10, label="T")

    assert resp.status_code == 400 and len(calls) == 1
