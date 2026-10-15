"""A tool with some arguments fixed by the harness.

"Make mistakes harder" applied to arguments (decided 2026-10-15): the
report chain knows the patient's gene, validated variant, genomic
position and protein position before any goal runs, so those are bound
into the tools that take them and removed from the schema the model sees.
The model chooses whether to call the tool and fills only the free
parameters (a chapter accession, a drug name, a PubMed query); it cannot
re-type the variant wrongly or look up another gene.
"""

from __future__ import annotations

import copy

from .base import BaseTool, EvidenceRecord


class BoundTool(BaseTool):
    def __init__(self, tool: BaseTool, fixed: dict):
        self._tool = tool
        # Only parameters the tool declares are bound; the rest are ignored.
        declared = tool.schema()["function"].get("parameters", {}).get("properties", {})
        self.fixed = {k: v for k, v in fixed.items() if k in declared and v not in (None, "")}

    @property
    def name(self) -> str:
        return self._tool.name

    def schema(self) -> dict:
        schema = copy.deepcopy(self._tool.schema())
        params = schema["function"].setdefault("parameters", {})
        props = params.get("properties", {})
        for key in self.fixed:
            props.pop(key, None)
        params["required"] = [r for r in params.get("required", []) if r not in self.fixed]
        if self.fixed:
            shown = ", ".join(f"{k}={v}" for k, v in self.fixed.items())
            schema["function"]["description"] += f" Fixed for this patient by the harness: {shown}."
        return schema

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        merged = {**kwargs, **self.fixed}  # the bound value wins over anything the model wrote
        return self._tool.execute(**merged)
