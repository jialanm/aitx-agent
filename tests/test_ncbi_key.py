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
