"""NCBI API key handling for the E-utilities adapters."""

from src.tools.base import NCBI_API_KEY_ENV, ncbi_params


def test_params_are_unchanged_when_no_key_is_set(monkeypatch):
    # Every machine without the key, including CI and this test run, takes
    # this path; it must send the request exactly as the adapter built it.
    monkeypatch.delenv(NCBI_API_KEY_ENV, raising=False)

    assert ncbi_params(db="pubmed", term="CFTR") == {"db": "pubmed", "term": "CFTR"}
