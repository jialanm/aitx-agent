"""Base tool infrastructure for the ReAct agent."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
