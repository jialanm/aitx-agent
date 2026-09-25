"""Base tool infrastructure for the ReAct agent."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# NCBI E-utilities allow 3 requests/s per IP without a key and 10/s with
# one. The key is a personal credential: read from the environment on
# every call, never stored in code, fixtures or committed files.
NCBI_API_KEY_ENV = "NCBI_API_KEY"
NCBI_KEY_PLACEHOLDER = "[NCBI_API_KEY]"


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
