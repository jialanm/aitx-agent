"""The actionability report, built as a chain of goals.

Each goal is one short conversation with its own tools and a fixed output
record (decided 2026-10-15, from the pattern in Anthropic's "Building
effective agents"): the model never plans the whole report. Between goals
the code checks every record against its source, a second model call
judges whether the quoted sentence supports the claim, and the markdown is
assembled by code so the template's sections are always present and in
order. Goal 1 (existing targeted therapies) is built; the other goals are
rendered as not yet assessed until their configurations are added.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict

from .agent import Agent
from .model import generate
from .preprocessing import parse_variant
from .prompts import _format_variant_summary
from .schemas import ReportInput, ReportOutput
from .tools.base import BaseTool
from .tools.bound import BoundTool
from .variant_validation import validate_variant

logger = logging.getLogger(__name__)

MAX_RECORDS_PER_GOAL = 8

# The template's sections, in the required order. A goal fills one
# subsection; unfilled subsections say so rather than being left out.
TEMPLATE = [
    ("Summary", []),
    ("Tier 1: Established Actions", ["Existing Targeted Therapies", "Supportive Management", "Contraindications & Precautions"]),
    ("Tier 2: Emerging Actions", ["Clinical Trial Eligibility", "Off-label/Expanded-Access Therapies"]),
    ("Tier 3: Hypothesized Actions", ["Bespoke Therapies", "Drug Repurposing", "Outcome Measures & Ranking"]),
]

# One goal: which subsection it fills, which tools it may call and which of
# their arguments the harness binds, the instruction, and the record fields.
GOALS = [
    {
        "key": "goal1_existing_targeted",
        "subsection": "Existing Targeted Therapies",
        "tools": ["search_genereviews", "search_fda_labels", "search_pubmed", "genetics_reference"],
        "bind": {"search_genereviews": ["gene"], "search_pubmed": ["gene"]},
        "instruction": (
            "List the targeted therapies that are established for this patient's condition: approved "
            "drugs or biologics directed at the gene, its protein, or the pathway, and for each say "
            "whether this patient's variant qualifies. Start from the disease chapter, confirm each "
            "candidate's indication and variant restrictions in its label, and use PubMed for a therapy "
            "the chapter does not name or that is approved outside the United States. A condition that "
            "limits the therapy (variant type, age, kidney function) is part of the record."
        ),
        "fields": {
            "therapy": "generic name",
            "status": "approved | investigational",
            "indication": "the indication as the source states it",
            "applies_to_patient": "yes | no | conditional: <the condition>",
            "source_url": "a URL returned by a tool in this conversation",
            "quote": "one sentence copied verbatim from that source's text that states the claim",
        },
        "claim": "{therapy} is {status} for this condition; applies to this patient: {applies_to_patient}",
    },
]

RECORD_FORMAT_NOTE = (
    "When you have the evidence, reply with ONLY a JSON array of records, one per therapy, and nothing "
    "else; reply with [] if none is established. Each record has exactly these fields:\n{fields}\n"
    "Every record must cite a source_url that a tool returned in this conversation and a quote copied "
    "verbatim from that source's text; a record without both will be discarded by the harness."
)

RULES = (
    "Rules: a fact you remember is a hypothesis, not evidence; confirm it with a tool before it enters a "
    "record. Every record needs a tool result that supports it; a result that names a drug without saying "
    "it applies to this condition and variant does not support a record about it. Never invent a URL, a "
    "drug name, or a sentence."
)

NOT_ASSESSED = "Not yet assessed in this version."


class ReportBuilder:
    """Runs the goal chain for one patient and renders the markdown."""

    def __init__(self, agent: Agent):
        self.agent = agent

    # ----- the chain -----

    def build(self, report_input: ReportInput) -> ReportOutput:
        agent = self.agent
        agent.last_trace = {
            "id": report_input.id, "tool_calls": [], "events": [], "messages": [],
            "validation": [], "model_turns": [], "goals": {}, "final_text": "", "answer": "",
        }
        patient_block, inputs = self._prepare_patient(report_input)
        filled: dict[str, list[dict]] = {}
        for goal in GOALS:
            agent.last_trace["events"].append(f"goal_start:{goal['key']}")
            calls_before = len(agent.last_trace["tool_calls"])
            records = self._run_goal(goal, patient_block, inputs)
            goal_calls = agent.last_trace["tool_calls"][calls_before:]
            records = self._check_sources(records, goal_calls)
            records = self._evaluate(goal, records, goal_calls)
            agent.last_trace["goals"][goal["key"]] = {
                "records": records,
                "sources_consulted": sorted({u for c in goal_calls for u in c.get("evidence_urls", [])}),
            }
            filled[goal["subsection"]] = records
        markdown = self.render(report_input, patient_block, filled, agent.last_trace["goals"])
        agent.last_trace["final_text"] = markdown
        return ReportOutput(id=report_input.id, report_markdown=markdown)

    def _prepare_patient(self, report_input: ReportInput) -> tuple[str, dict]:
        """Step 1 of the chain: validate every variant, fetch its consequence,
        and collect the values the harness binds into the goals' tools."""
        agent = self.agent
        parsed = [parse_variant(v) for v in report_input.patient.genotype]
        validated = [validate_variant(g.transcript, g.variant_cdna) for g in report_input.patient.genotype]
        agent.last_trace["validation"] = [asdict(v) for v in validated]
        step1_evidence = []
        consequences = [agent._predict_consequence(g.gene, v, step1_evidence)
                        for g, v in zip(report_input.patient.genotype, validated)]
        block = _format_variant_summary(parsed, validated, consequences)
        first_gene = report_input.patient.genotype[0].gene if report_input.patient.genotype else ""
        first_valid = next((v for v in validated if v.valid), None)
        position = None
        for text in consequences:
            m = re.search(r"^Protein position: (\d+)", text, re.M)
            if m:
                position = int(m.group(1)); break
        inputs = {
            "gene": first_gene,
            "transcript": first_valid.transcript_variant.split(":")[0] if first_valid else "",
            "variant": first_valid.transcript_variant.split(":", 1)[1] if first_valid and ":" in first_valid.transcript_variant else "",
            "genomic_hgvs": first_valid.genomic_variant if first_valid else "",
            "protein_position": position,
            "clinical_context": report_input.patient.clinical_context,
        }
        return block, inputs

    def _tools_for(self, goal: dict, inputs: dict) -> list[BaseTool]:
        tools = []
        for name in goal["tools"]:
            tool = self.agent.tools.get(name)
            if tool is None:
                logger.warning(f"Goal {goal['key']} names tool {name}, which is not registered")
                continue
            bound = {k: inputs.get(k) for k in goal.get("bind", {}).get(name, [])}
            tools.append(BoundTool(tool, bound) if any(v not in (None, "") for v in bound.values()) else tool)
        return tools

    def _run_goal(self, goal: dict, patient_block: str, inputs: dict) -> list[dict]:
        fields = "\n".join(f"  {k}: {v}" for k, v in goal["fields"].items())
        system_prompt = (
            "You are a precision medicine expert assistant building one section of a therapeutic "
            "actionability report for a rare-disease patient, using the tools provided.\n\n"
            f"## Patient\n{patient_block}\n\n### Clinical Context\n{inputs['clinical_context']}\n\n"
            f"## This section's goal\n{goal['instruction']}\n\n"
            f"## Output\n{RECORD_FORMAT_NOTE.format(fields=fields)}\n\n{RULES}"
        )
        user_message = f"Build the section: {goal['subsection']}."
        text, _ = self.agent.converse(system_prompt, user_message, self._tools_for(goal, inputs), kind=goal["key"])
        records = self._parse_records(text)
        if records is None:
            # One retry with thinking off, asking for the array alone.
            self.agent.last_trace["events"].append(f"records_unparsable:{goal['key']}")
            retry = generate(
                self.agent.model, self.agent.tokenizer,
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": f"Your previous reply was not a JSON array. Reply with ONLY the JSON array of records "
                                             f"for: {goal['subsection']}, using the tool results you already have:\n\n{text}"}],
                tools=None, max_new_tokens=1024, enable_thinking=False,
            )
            self.agent._record_turn(f"{goal['key']}_records_retry", 0, retry)
            records = self._parse_records(retry.text) or []
        return records[:MAX_RECORDS_PER_GOAL]

    @staticmethod
    def _parse_records(text: str) -> list[dict] | None:
        """The JSON array in the text, or None if there is none."""
        if not text:
            return None
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list):
            return None
        return [r for r in data if isinstance(r, dict)]

    # ----- the checks -----

    def _check_sources(self, records: list[dict], goal_calls: list[dict]) -> list[dict]:
        """Keep a record only if its URL came from a tool in this goal and its
        quote is verbatim in that source's text."""
        urls = {u for c in goal_calls for u in c.get("evidence_urls", [])}
        kept = []
        for r in records:
            url, quote = str(r.get("source_url") or ""), str(r.get("quote") or "")
            label = str(r.get("therapy") or r.get("name") or "?")
            if not label.strip():
                self.agent.last_trace["events"].append("record_dropped:no_therapy")
                continue
            if url not in urls:
                self.agent.last_trace["events"].append(f"record_dropped:no_source:{label}")
                logger.warning(f"Record {label!r} cites {url!r}, which no tool returned in this goal")
                continue
            text = next((c.get("summary", "") for c in goal_calls if url in c.get("evidence_urls", [])), "")
            if not Agent._quote_in_text(quote, text) or Agent._is_adapter_boilerplate(quote):
                self.agent.last_trace["events"].append(f"record_dropped:quote_not_in_source:{label}")
                logger.warning(f"Record {label!r}: quote not found verbatim in {url}")
                continue
            kept.append(r)
        return kept

    def _evaluate(self, goal: dict, records: list[dict], goal_calls: list[dict]) -> list[dict]:
        """A second model call per record, thinking off: does the quoted
        sentence support the claim? NOT SUPPORTED drops the record; every
        verdict is kept in the trace."""
        kept = []
        for r in records:
            claim = goal["claim"].format(**{k: r.get(k, "?") for k in goal["fields"]})
            messages = [
                {"role": "system", "content": "You judge whether a quoted sentence from a source supports a claim. "
                                              "Reply with SUPPORTED or NOT SUPPORTED on the first line, then one sentence saying why."},
                {"role": "user", "content": f"Claim: {claim}\nSource: {r.get('source_url')}\nQuote: \"{r.get('quote')}\""},
            ]
            try:
                verdict = generate(self.agent.model, self.agent.tokenizer, messages, tools=None, max_new_tokens=120, enable_thinking=False)
            except Exception as e:
                logger.error(f"Evaluator failed for {r.get('therapy')}: {e}")
                self.agent.last_trace["events"].append(f"record_verdict:{r.get('therapy')}:ERROR")
                kept.append(r)
                continue
            self.agent._record_turn(f"{goal['key']}_evaluator", 0, verdict)
            supported = verdict.text.strip().upper().startswith("SUPPORTED")
            r["evaluator"] = verdict.text.strip()[:300]
            self.agent.last_trace["events"].append(f"record_verdict:{r.get('therapy')}:{'SUPPORTED' if supported else 'NOT_SUPPORTED'}")
            if supported:
                kept.append(r)
        return kept

    # ----- the markdown -----

    @staticmethod
    def _render_records(records: list[dict], goal: dict) -> str:
        lines = []
        for r in records:
            lines.append(
                f"- **{r.get('therapy')}** ({r.get('status')}). {r.get('indication')} "
                f"Applies to this patient: {r.get('applies_to_patient')}. "
                f"Source: \"{r.get('quote')}\" [{r.get('source_url')}]({r.get('source_url')})"
            )
        return "\n".join(lines)

    def render(self, report_input: ReportInput, patient_block: str, filled: dict[str, list[dict]], goals: dict) -> str:
        by_subsection = {g["subsection"]: g for g in GOALS}
        out = [f"# Therapeutic Actionability Report: {report_input.id}", ""]
        for section, subsections in TEMPLATE:
            out.append(f"## {section}")
            if section == "Summary":
                out.append(self._summary(report_input, filled, goals))
            out.append("")
            for sub in subsections:
                out.append(f"### {sub}")
                if sub in filled:
                    records = filled[sub]
                    if records:
                        out.append(self._render_records(records, by_subsection[sub]))
                    else:
                        consulted = goals[by_subsection[sub]["key"]]["sources_consulted"]
                        out.append("None found. Sources consulted: " + (", ".join(consulted) if consulted else "none returned a record") + ".")
                else:
                    out.append(NOT_ASSESSED)
                out.append("")
        return "\n".join(out).rstrip() + "\n"

    @staticmethod
    def _summary(report_input: ReportInput, filled: dict[str, list[dict]], goals: dict) -> str:
        variants = "; ".join(f"{v.gene} {v.transcript}:{v.variant_cdna} ({v.variant_protein}, {v.zygosity})"
                             for v in report_input.patient.genotype)
        lines = [f"Patient: {variants}. {report_input.patient.clinical_context}", ""]
        targeted = filled.get("Existing Targeted Therapies", [])
        if targeted:
            lines.append("Tier 1, established targeted therapy: " + "; ".join(
                f"{r.get('therapy')} ({r.get('status')}; applies: {r.get('applies_to_patient')})" for r in targeted) + ".")
        elif "Existing Targeted Therapies" in filled:
            lines.append("Tier 1: no established targeted therapy was found in the sources consulted.")
        lines.append("Tier 2 and Tier 3 sections are not yet assessed in this version.")
        return "\n".join(lines)
