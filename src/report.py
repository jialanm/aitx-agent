"""The actionability report, built as a chain of goals.

Each goal is one short conversation with its own tools and a fixed output
record (decided 2026-10-15, from the pattern in Anthropic's "Building
effective agents"): the model never plans the whole report. Between goals
the code checks every record against its source, a second model call
judges whether the quoted sentence supports the claim, and the markdown is
assembled by code so the template's sections are always present and in
order. Five goals fill the template's eight subsections; Outcome Measures &
Ranking is written by code from the Tier 3 records. The challenge's three
cross-section rules are enforced: every record cites a source or carries
its reasoning, conditional actions are stated as "If X, then Y; otherwise
Z", and an action appears in one section only.
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

# The template's sections, in the required order. A goal fills one or two
# subsections; an unfilled subsection says so rather than being left out.
TEMPLATE = [
    ("Summary", []),
    ("Tier 1: Established Actions", ["Existing Targeted Therapies", "Supportive Management", "Contraindications & Precautions"]),
    ("Tier 2: Emerging Actions", ["Clinical Trial Eligibility", "Off-label or Expanded-Access Therapies"]),
    ("Tier 3: Hypothesized Actions", ["Bespoke Therapies", "Drug Repurposing", "Outcome Measures & Ranking"]),
]

# One goal: which subsections it fills, which tools it may call and which of
# their arguments the harness binds, the instruction, the record fields, the
# field that names a record, and the claim the evaluator judges. A goal that
# fills two subsections sorts its records by one field.
GOALS = [
    {
        "key": "goal1_existing_targeted",
        "subsections": ["Existing Targeted Therapies"],
        "tools": ["search_genereviews", "search_fda_labels", "search_pubmed", "genetics_reference"],
        "bind": {"search_genereviews": ["gene"], "search_pubmed": ["gene"]},
        "instruction": (
            "List the targeted therapies that are established for this patient's condition: approved or "
            "guideline-supported treatments directed at the gene, its protein, the pathway or the disease "
            "mechanism, and for each say whether this patient's variant qualifies. Start from the disease "
            "chapter, confirm each candidate's indication and variant restrictions in its label, and use "
            "PubMed for a therapy the chapter does not name or that is approved outside the United States."
        ),
        "fields": {
            "therapy": "generic name",
            "status": "approved | guideline-supported | investigational",
            "indication": "the indication as the source states it",
            "applies_to_patient": "yes | no | conditional: <'If <missing information or finding>, then <action>; otherwise <alternative>'>",
        },
        "label": "therapy",
        "claim": "{therapy} is {status} for this condition; applies to this patient: {applies_to_patient}",
    },
    {
        "key": "goal2_supportive_and_precautions",
        "subsections": ["Supportive Management", "Contraindications & Precautions"],
        "sort_by": {"field": "kind", "map": {"supportive": "Supportive Management", "avoid": "Contraindications & Precautions"}},
        "tools": ["search_genereviews", "search_pubmed", "search_fda_labels"],
        "bind": {"search_genereviews": ["gene"], "search_pubmed": ["gene"]},
        "instruction": (
            "List (a) the standard-of-care pharmacologic or procedural management of this condition's symptoms "
            "and complications, not of the underlying defect, and (b) the agents, doses or procedures to avoid "
            "or modify because of the variant or disease, including anesthesia risks and clinically relevant "
            "drug interactions. Surveillance and monitoring are excluded, and so is any therapy already "
            "recorded as an established targeted therapy. The disease chapter's management text is the "
            "primary source. If the clinical context names medications the patient takes, fetch each one's "
            "label and record any interaction or warning that the disease or an established therapy makes "
            "relevant. One record per agent or procedure, with kind 'supportive' or 'avoid'."
        ),
        "fields": {
            "agent": "drug, procedure or measure",
            "kind": "supportive | avoid",
            "purpose": "what it treats or why it is avoided, as the source states it",
            "applies_to_patient": "yes | no | conditional: <'If ..., then ...; otherwise ...'>",
        },
        "label": "agent",
        "claim": "{agent} is {kind} management for this condition: {purpose}",
    },
    {
        "key": "goal3_trials",
        "subsections": ["Clinical Trial Eligibility"],
        "tools": ["search_clinical_trials", "search_omim"],
        "bind": {"search_omim": ["gene"]},
        "instruction": (
            "Find actively recruiting clinical trials (status RECRUITING or NOT_YET_RECRUITING) for which this "
            "patient may qualify. Search by the gene symbol alone and, if that returns nothing, by the disease "
            "name; then fetch every candidate by its NCT ID and judge eligibility against its inclusion and "
            "exclusion criteria using the patient's age, findings and treatments as given. A criterion the "
            "patient record does not settle is stated as a condition, not assumed. One record per trial."
        ),
        "fields": {
            "nct_id": "the trial identifier, exactly as returned",
            "title": "the trial title as returned",
            "status": "the status as returned",
            "eligibility": "likely eligible | likely ineligible: <criterion> | conditional: <'If <missing information>, then eligible; otherwise ...'>",
        },
        "label": "nct_id",
        "claim": "trial {nct_id} ({status}) is relevant to this patient and the patient is {eligibility}",
    },
    {
        "key": "goal4_offlabel_and_repurposing",
        "subsections": ["Off-label or Expanded-Access Therapies", "Drug Repurposing"],
        "sort_by": {"field": "human_precedent", "map": {"yes": "Off-label or Expanded-Access Therapies", "no": "Drug Repurposing"}},
        "tools": ["query_pharos", "search_pubmed", "search_fda_labels", "genetics_reference"],
        "bind": {"query_pharos": ["gene"], "search_pubmed": ["gene"]},
        "instruction": (
            "Find existing approved or investigational compounds, not already established for this condition, "
            "that could act on the molecular defect. For each, state the mechanism and whether it runs in the "
            "right direction for this variant's effect (a loss of function needs activity raised, a gain of "
            "function lowered; read the genetics reference, topic mechanism-direction). Then search PubMed for "
            "a published report of the compound given to a patient with the same gene, disease or variant "
            "context: if one exists, human_precedent is 'yes' and the record belongs under off-label use, with "
            "that report as its source; if none exists, 'no' and it belongs under drug repurposing, with the "
            "underlying facts as its source and the reasoning that links them. Also search PubMed for the disease "
            "with 'treatment' or 'case report', without a compound in mind: a therapy with a published human "
            "report may act on the pathway rather than the protein and would not appear in PHAROS. The PHAROS "
            "drug list is measured activity at the protein in any direction, not a treatment lead."
        ),
        "fields": {
            "compound": "generic name",
            "mechanism": "how it acts on the protein or pathway, and the direction",
            "human_precedent": "yes | no",
            "rationale": "why it fits this variant's effect, as the sources state it",
            "outcome_measure": "the objective, age-appropriate metric a source uses for this compound's safety or efficacy, as stated, or 'none stated'",
        },
        "label": "compound",
        "claim": "{compound} ({mechanism}) is a candidate for this variant; human precedent: {human_precedent}; {rationale}",
    },
    {
        "key": "goal5_bespoke",
        "subsections": ["Bespoke Therapies"],
        "tools": ["n1c_aso_guidelines", "genetics_reference", "clingen_dosage", "query_uniprot", "search_pubmed"],
        "bind": {"clingen_dosage": ["gene"], "query_uniprot": ["gene", "protein_position"], "search_pubmed": ["gene"]},
        "instruction": (
            "Assess this variant's amenability to DNA- and RNA-directed strategies not already established or "
            "emerging: splice correction, exon skipping, transcript knockdown (ASO or siRNA), gene replacement "
            "or other gene therapy. An approach already approved for this variant belongs under Existing "
            "Targeted Therapies and is not repeated here; a feasible approach not yet developed or tested is. "
            "Read the N1C guidelines first and apply their criteria to the variant's consequence, exon and "
            "frame from the patient block; use ClinGen dosage sensitivity for knockdown, the genetics reference "
            "for the frame and splicing rules, and PubMed for any such approach already developed for the gene, "
            "exon or variant. One record per strategy with a verdict and its basis."
        ),
        "fields": {
            "strategy": "splice correction | exon skipping | transcript knockdown | gene replacement | other: <name>",
            "verdict": "eligible | likely eligible | unlikely eligible | not eligible | unable to assess",
            "basis": "the criterion that decides the verdict, as the source states it",
            "outcome_measure": "the readout a source names for judging this strategy's effect (transcript, protein or clinical), as stated, or 'none stated'",
        },
        "label": "strategy",
        "claim": "{strategy} is {verdict} for this variant because {basis}",
    },
]

# Fields every record carries, after the goal's own: the source, the quote,
# and the reasoning for a claim the quote alone does not state.
COMMON_FIELDS = {
    "source_url": "a URL returned by a tool in this conversation",
    "quote": "one sentence copied verbatim from that source's text that states the fact the claim rests on",
    "reasoning": "empty if the quote states the claim; otherwise the steps from the quoted fact to the claim",
}

RECORD_FORMAT_NOTE = (
    "When you have the evidence, reply with ONLY a JSON array of records and nothing else; reply with [] if "
    "there is none. Each record has exactly these fields:\n{fields}\n"
    "Every record must cite a source_url that a tool returned in this conversation and a quote copied "
    "verbatim from that source's text; a record without both will be discarded by the harness. A source "
    "need not be a paper: a database record, drug label, guideline or trial registry entry is one. If no "
    "report states the claim, cite the underlying fact and put the reasoning that reaches the claim in the "
    "reasoning field. Never cite a source that does not support the claim."
)

RULES = (
    "Rules: a fact you remember is a hypothesis, not evidence; confirm it with a tool before it enters a "
    "record. Every record needs a tool result that supports it; a result that names a drug without saying "
    "it applies to this condition and variant does not support a record about it. If missing patient "
    "information would change a recommendation, say so in the form 'If X, then Y; otherwise Z'. Never "
    "invent a URL, a drug name, or a sentence."
)

NOT_ASSESSED = "Not yet assessed in this version."


def by_sub_label(subsection: str) -> str:
    """The record field that names a record in this subsection."""
    return next(g["label"] for g in GOALS if subsection in g["subsections"])


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
            try:
                records = self._run_goal(goal, patient_block, inputs, filled)
            except Exception as e:
                # One goal's failure must not lose the others' sections.
                logger.error(f"Goal {goal['key']} failed: {e}")
                agent.last_trace["events"].append(f"goal_failed:{goal['key']}:{e}")
                records = []
            goal_calls = agent.last_trace["tool_calls"][calls_before:]
            records = self._check_sources(goal, records, goal_calls)
            records = self._evaluate(goal, records)
            records = self._drop_repeats(goal, records, filled)
            agent.last_trace["goals"][goal["key"]] = {
                "records": records,
                "sources_consulted": sorted({u for c in goal_calls for u in c.get("evidence_urls", [])}),
            }
            for sub in goal["subsections"]:
                filled[sub] = []
            sort = goal.get("sort_by")
            for r in records:
                sub = goal["subsections"][0]
                if sort:
                    sub = sort["map"].get(str(r.get(sort["field"], "")).strip().lower(), sub)
                filled[sub].append(r)
        markdown = self.render(report_input, filled, agent.last_trace["goals"])
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
                position = int(m.group(1))
                break
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

    def _run_goal(self, goal: dict, patient_block: str, inputs: dict, filled: dict[str, list[dict]]) -> list[dict]:
        all_fields = {**goal["fields"], **COMMON_FIELDS}
        fields = "\n".join(f"  {k}: {v}" for k, v in all_fields.items())
        already = [f"{sub}: " + ", ".join(str(r.get(by_sub_label(sub))) for r in recs)
                   for sub, recs in filled.items() if recs]
        earlier = ("\n\n## Already recorded in earlier sections (do not repeat these here)\n" + "\n".join(already)) if already else ""
        system_prompt = (
            "You are a precision medicine expert assistant building one section of a therapeutic "
            "actionability report for a rare-disease patient, using the tools provided.\n\n"
            f"## Patient\n{patient_block}\n\n### Clinical Context\n{inputs['clinical_context']}\n\n"
            f"## This section's goal\n{goal['instruction']}{earlier}\n\n"
            f"## Output\n{RECORD_FORMAT_NOTE.format(fields=fields)}\n\n{RULES}"
        )
        sections = " and ".join(goal["subsections"])
        text, _ = self.agent.converse(system_prompt, f"Build the section: {sections}.", self._tools_for(goal, inputs), kind=goal["key"])
        records = self._parse_records(text)
        if records is None:
            # One retry with thinking off, asking for the array alone.
            self.agent.last_trace["events"].append(f"records_unparsable:{goal['key']}")
            retry = generate(
                self.agent.model, self.agent.tokenizer,
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": "Your previous reply was not a JSON array. Reply with ONLY the JSON array of "
                                             f"records for: {sections}, using the tool results you already have:\n\n{text}"}],
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

    def _check_sources(self, goal: dict, records: list[dict], goal_calls: list[dict]) -> list[dict]:
        """Keep a record only if its URL came from a tool in this goal and its
        quote is verbatim in that source's text."""
        urls = {u for c in goal_calls for u in c.get("evidence_urls", [])}
        kept = []
        for r in records:
            url, quote = str(r.get("source_url") or ""), str(r.get("quote") or "")
            label = str(r.get(goal["label"]) or "").strip()
            if not label:
                self.agent.last_trace["events"].append(f"record_dropped:no_{goal['label']}")
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

    def _evaluate(self, goal: dict, records: list[dict]) -> list[dict]:
        """A second model call per record, thinking off: does the quoted
        sentence, with the reasoning given if any, support the claim? NOT
        SUPPORTED drops the record; every verdict is kept in the trace."""
        kept = []
        for r in records:
            label = r.get(goal["label"])
            claim = goal["claim"].format(**{k: r.get(k, "?") for k in goal["fields"]})
            user = f"Claim: {claim}\nSource: {r.get('source_url')}\nQuote: \"{r.get('quote')}\""
            if r.get("reasoning"):
                user += f"\nReasoning given: {r.get('reasoning')}"
            messages = [
                {"role": "system", "content": "You judge whether a quoted sentence from a source supports a claim, directly or "
                                              "through the reasoning given, if any. Reply with SUPPORTED or NOT SUPPORTED on the "
                                              "first line, then one sentence saying why."},
                {"role": "user", "content": user},
            ]
            try:
                verdict = generate(self.agent.model, self.agent.tokenizer, messages, tools=None, max_new_tokens=120, enable_thinking=False)
            except Exception as e:
                logger.error(f"Evaluator failed for {label}: {e}")
                self.agent.last_trace["events"].append(f"record_verdict:{label}:ERROR")
                kept.append(r)
                continue
            self.agent._record_turn(f"{goal['key']}_evaluator", 0, verdict)
            supported = verdict.text.strip().upper().startswith("SUPPORTED")
            r["evaluator"] = verdict.text.strip()[:300]
            self.agent.last_trace["events"].append(f"record_verdict:{label}:{'SUPPORTED' if supported else 'NOT_SUPPORTED'}")
            if supported:
                kept.append(r)
        return kept

    def _drop_repeats(self, goal: dict, records: list[dict], filled: dict[str, list[dict]]) -> list[dict]:
        """Each action belongs in one section: a record whose name already
        appears in an earlier section is dropped and logged."""
        seen = {str(r.get(by_sub_label(sub)) or "").strip().lower() for sub, recs in filled.items() for r in recs}
        kept = []
        for r in records:
            name = str(r.get(goal["label"]) or "").strip().lower()
            if name and name in seen:
                self.agent.last_trace["events"].append(f"record_dropped:repeated:{name}")
                continue
            kept.append(r)
        return kept

    # ----- the markdown -----

    @staticmethod
    def _cite(r: dict) -> str:
        text = f"Source: \"{r.get('quote')}\" [{r.get('source_url')}]({r.get('source_url')})"
        if r.get("reasoning"):
            text += f" Reasoning: {r.get('reasoning')}"
        return text

    def _render_records(self, records: list[dict], goal: dict) -> str:
        key, lines = goal["key"], []
        for r in records:
            if key == "goal1_existing_targeted":
                lines.append(f"- **{r.get('therapy')}** ({r.get('status')}). {r.get('indication')} "
                             f"Applies to this patient: {r.get('applies_to_patient')}. {self._cite(r)}")
            elif key == "goal2_supportive_and_precautions":
                lines.append(f"- **{r.get('agent')}**: {r.get('purpose')} "
                             f"Applies to this patient: {r.get('applies_to_patient')}. {self._cite(r)}")
            elif key == "goal3_trials":
                lines.append(f"- **{r.get('nct_id')}**, {r.get('title')} ({r.get('status')}). "
                             f"Eligibility: {r.get('eligibility')}. {self._cite(r)}")
            elif key == "goal4_offlabel_and_repurposing":
                lines.append(f"- **{r.get('compound')}**: {r.get('mechanism')} {r.get('rationale')} "
                             f"Human treatment precedent: {r.get('human_precedent')}. {self._cite(r)}")
            elif key == "goal5_bespoke":
                lines.append(f"- **{r.get('strategy')}**: {r.get('verdict')}. {r.get('basis')} {self._cite(r)}")
            else:
                lines.append(f"- {json.dumps(r)}")
        return "\n".join(lines)

    @staticmethod
    def _outcome_measures(filled: dict[str, list[dict]]) -> str:
        """Tier 3's last subsection, written by code from the Tier 3 records:
        the hypothesised actions ranked by a stated rule, each with the
        measure its own source names and the safety considerations its
        modality carries."""
        order = {"eligible": 0, "likely eligible": 1, "unlikely eligible": 3, "unable to assess": 4, "not eligible": 5}

        def measure(r):
            m = str(r.get("outcome_measure") or "").strip()
            return m if m and m.lower() != "none stated" else "no metric stated by the sources; one must be defined before use"

        hyp = []
        for r in filled.get("Bespoke Therapies", []):
            rank = order.get(str(r.get("verdict", "")).strip().lower(), 4)
            if rank <= 1:
                hyp.append((rank, f"{r.get('strategy')} (bespoke; {r.get('verdict')})",
                            f"efficacy: {measure(r)}; safety: delivery route, dosing and off-target effects assessed per "
                            "the N1C guidelines' preclinical requirements before first use"))
        for r in filled.get("Drug Repurposing", []):
            hyp.append((2, f"{r.get('compound')} (repurposing; no human precedent)",
                        f"efficacy: {measure(r)}; safety: the compound's existing label and dose range with "
                        "age-appropriate monitoring, after testing in a model of the variant"))
        if not hyp:
            return ("No hypothesized action was recorded in Tier 3, so no outcome measures are proposed; established "
                    "and emerging actions are measured by the endpoints of their own labels and trials.")
        hyp.sort(key=lambda h: h[0])
        lines = ["Ranking rule: bespoke strategies the guidelines judge eligible first, then likely eligible, then "
                 "repurposing candidates with a mechanistic rationale; within a rank, in the order recorded.", ""]
        for i, (_, name, measures) in enumerate(hyp, 1):
            lines.append(f"{i}. {name}. Outcome measures: {measures}.")
        return "\n".join(lines)

    def render(self, report_input: ReportInput, filled: dict[str, list[dict]], goals: dict) -> str:
        by_subsection = {sub: g for g in GOALS for sub in g["subsections"]}
        out = [f"# Therapeutic Actionability Report for Patient {report_input.id}", ""]
        for section, subsections in TEMPLATE:
            out.append(f"## {section}")
            if section == "Summary":
                out.append(self._summary(report_input, filled))
            out.append("")
            for sub in subsections:
                out.append(f"### {sub}")
                if sub == "Outcome Measures & Ranking":
                    out.append(self._outcome_measures(filled))
                elif sub in filled:
                    goal = by_subsection[sub]
                    records = filled[sub]
                    if records:
                        out.append(self._render_records(records, goal))
                    else:
                        consulted = goals[goal["key"]]["sources_consulted"]
                        out.append("None found. Sources consulted: " + (", ".join(consulted) if consulted else "none returned a record") + ".")
                else:
                    out.append(NOT_ASSESSED)
                out.append("")
        return "\n".join(out).rstrip() + "\n"

    @staticmethod
    def _summary(report_input: ReportInput, filled: dict[str, list[dict]]) -> str:
        variants = "; ".join(f"{v.gene} {v.transcript}:{v.variant_cdna} ({v.variant_protein}, {v.zygosity})"
                             for v in report_input.patient.genotype)
        lines = [f"Patient: {variants}. {report_input.patient.clinical_context}", ""]
        tier1 = filled.get("Existing Targeted Therapies", [])
        lines.append("Tier 1: " + ("established targeted therapy: " + "; ".join(
            f"{r.get('therapy')} (applies: {r.get('applies_to_patient')})" for r in tier1) + "."
            if tier1 else "no established targeted therapy was found in the sources consulted."))
        avoid = [str(r.get("agent")) for r in filled.get("Contraindications & Precautions", [])]
        if avoid:
            lines.append("Avoid or modify: " + ", ".join(avoid) + ".")
        trials = [r for r in filled.get("Clinical Trial Eligibility", [])
                  if not str(r.get("eligibility", "")).lower().startswith("likely ineligible")]
        offlabel = [str(r.get("compound")) for r in filled.get("Off-label or Expanded-Access Therapies", [])]
        tier2 = []
        if trials:
            tier2.append("recruiting trials to assess: " + ", ".join(f"{r.get('nct_id')} ({r.get('eligibility')})" for r in trials))
        if offlabel:
            tier2.append("off-label therapies with human precedent: " + ", ".join(offlabel))
        lines.append("Tier 2: " + ("; ".join(tier2) + "." if tier2 else "no recruiting trial or off-label therapy with human precedent was found."))
        bespoke = [r for r in filled.get("Bespoke Therapies", []) if str(r.get("verdict", "")).lower() in ("eligible", "likely eligible")]
        repurp = [str(r.get("compound")) for r in filled.get("Drug Repurposing", [])]
        tier3 = []
        if bespoke:
            tier3.append("bespoke strategies judged eligible or likely eligible: " + ", ".join(str(r.get("strategy")) for r in bespoke))
        if repurp:
            tier3.append("repurposing candidates with a mechanistic rationale: " + ", ".join(repurp))
        lines.append("Tier 3: " + ("; ".join(tier3) + "." if tier3 else "no hypothesized action was recorded."))
        # Conditional actions: every record whose applicability depends on
        # information the patient record lacks is restated here.
        conditional = []
        for sub, recs in filled.items():
            for r in recs:
                for field in ("applies_to_patient", "eligibility"):
                    val = str(r.get(field, ""))
                    if val.lower().startswith("conditional"):
                        conditional.append(f"{sub}, {r.get(by_sub_label(sub))}: {val.split(':', 1)[-1].strip()}")
        if conditional:
            lines.append("Conditional actions, depending on information not in the patient record: " + " ".join(sorted(set(conditional))))
        return "\n".join(lines)
