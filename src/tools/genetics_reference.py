"""Genetics reference tool: short, cited rules for interpreting a variant's consequence.

The rules live in genetics_reference.md beside this file, one section per
topic, each ending in a Source line whose PubMed links become the evidence.
The model reads a section and quotes it; nothing here is generated.
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import BaseTool, EvidenceRecord

REFERENCE_PATH = Path(__file__).with_name("genetics_reference.md")
_SECTION = re.compile(r"^## ([a-z-]+): (.+)$", re.M)
_URL = re.compile(r"https://pubmed\.ncbi\.nlm\.nih\.gov/\d+/")


def load_sections(path: Path = REFERENCE_PATH) -> dict[str, dict]:
    """{topic: {"title", "text", "sources"}} from the markdown file."""
    text = path.read_text()
    heads = list(_SECTION.finditer(text))
    out: dict[str, dict] = {}
    for i, m in enumerate(heads):
        body = text[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(text)].strip()
        out[m.group(1)] = {"title": m.group(2).strip(), "text": body, "sources": _URL.findall(body)}
    return out


class GeneticsReferenceTool(BaseTool):
    """Return one cited rule of molecular genetics, or the list of rules."""

    name = "genetics_reference"

    def __init__(self, path: Path = REFERENCE_PATH):
        self.sections = load_sections(path)

    def schema(self) -> dict:
        topics = ", ".join(f"{k} ({v['title'].lower()})" for k, v in self.sections.items())
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Short, cited rules of molecular genetics for interpreting a variant's "
                    "predicted consequence: " + topics + ". Each rule states when it applies in "
                    "terms of the VEP block's lines (Frame, exon, coverage) and gives its source, "
                    "which is the citation. Call it when a step of your reasoning rests on such a "
                    "rule rather than on a fact about the patient or the gene, and quote the rule. "
                    "With no topic it lists the rules."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "enum": list(self.sections),
                            "description": "The rule to read, by its id",
                        },
                    },
                    "required": [],
                },
            },
        }

    def execute(self, topic: str | None = None, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        if not topic:
            lines = ["Genetics reference, rules available (call again with a topic):"]
            lines += [f"- {k}: {v['title']}" for k, v in self.sections.items()]
            return "\n".join(lines), []
        section = self.sections.get(topic)
        if section is None:
            return (f"Genetics reference has no rule '{topic}'. Rules: " + ", ".join(self.sections)), []
        text = f"Genetics reference, {topic}: {section['title']}\n\n{section['text']}"
        return text, [EvidenceRecord(url=u) for u in section["sources"]]
