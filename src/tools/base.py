"""Base tool infrastructure for the ReAct agent."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# NCBI E-utilities allow 3 requests/s per IP without a key and 10/s with
# one. The key is a personal credential: read from the environment on
# every call, never stored in code, fixtures or committed files.
NCBI_API_KEY_ENV = "NCBI_API_KEY"
NCBI_KEY_PLACEHOLDER = "[NCBI_API_KEY]"

# E-utilities answer HTTP 429 when the per-second limit is exceeded (3/s
# without a key, 10/s with one); a second later the same request succeeds.
# Timeouts and 5xx are retried at a longer interval. 4xx other than 429
# means the request is wrong and is returned at once.
NCBI_RETRIES = 5
NCBI_RATE_LIMIT_PAUSE_SECONDS = 1
NCBI_RETRY_INTERVAL_SECONDS = 5

logger = logging.getLogger(__name__)


@dataclass
class EvidenceRecord:
    """A single piece of evidence collected from a tool execution."""

    url: str
    time_accessed: int = field(default_factory=lambda: int(time.time()))


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    @abstractmethod
    def schema(self) -> dict:
        """Return OpenAI-style function tool definition.

        Returns a dict like:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { ... }
            }
        }
        """

    @abstractmethod
    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        """Execute the tool with given arguments.

        Returns:
            A tuple of (summary_text, evidence_records) where summary_text
            is <=2000 chars for the LLM context, and evidence_records are
            the URLs and timestamps for the output JSON.
        """

    @property
    def name(self) -> str:
        """Tool name extracted from schema."""
        return self.schema()["function"]["name"]


def ncbi_params(**params) -> dict:
    """Build E-utilities query parameters, adding the API key when one is set."""
    key = os.environ.get(NCBI_API_KEY_ENV)
    if key:
        params["api_key"] = key
    return params


def redact_ncbi_key(text: str) -> str:
    """Replace the API key wherever it appears in text.

    requests puts the full request URL, query string included, into
    HTTPError messages, and the adapters return those messages as tool
    results that reach the model context and the saved eval outputs.
    """
    key = os.environ.get(NCBI_API_KEY_ENV)
    if key:
        text = text.replace(key, NCBI_KEY_PLACEHOLDER)
    return text


def ncbi_get(url: str, params: dict, timeout: float, label: str = "NCBI") -> "requests.Response":
    """GET an E-utilities URL, retrying 429, 5xx and timeouts; raises after the last try.

    Drop-in for requests.get: returns the response for any status below
    500 except 429, so callers keep their own raise_for_status() handling.
    """
    import requests  # local import keeps base.py free of a hard dependency at import time

    last_exc: Exception | None = None
    for attempt in range(1, NCBI_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except requests.Timeout as e:
            last_exc = e
            logger.warning(f"{label} timeout (attempt {attempt}/{NCBI_RETRIES}): {url}")
            pause = NCBI_RETRY_INTERVAL_SECONDS
        else:
            if resp.status_code == 429:
                logger.warning(f"{label} rate limited, HTTP 429 (attempt {attempt}/{NCBI_RETRIES}): {url}")
                # resp.url carries the query string, api_key included, like requests'
                # own HTTPError text; callers redact before returning it.
                last_exc = requests.HTTPError(f"429 Too Many Requests after {NCBI_RETRIES} attempts for url: {resp.url}", response=resp)
                pause = NCBI_RATE_LIMIT_PAUSE_SECONDS
            elif resp.status_code >= 500:
                logger.warning(f"{label} server error, HTTP {resp.status_code} (attempt {attempt}/{NCBI_RETRIES}): {url}")
                last_exc = requests.HTTPError(f"{resp.status_code} Server Error after {NCBI_RETRIES} attempts for url: {resp.url}", response=resp)
                pause = NCBI_RETRY_INTERVAL_SECONDS
            else:
                return resp
        if attempt < NCBI_RETRIES:
            time.sleep(pause)
    raise last_exc  # type: ignore[misc]
