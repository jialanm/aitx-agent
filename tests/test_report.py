"""The report chain, goal 1, with a scripted model and stub tools: records are
parsed, checked against their sources, judged by the evaluator, and rendered
into the template's sections in order."""

from __future__ import annotations

import pytest

import src.agent as agent_module
import src.report as report_module
from src.agent import Agent
from src.model import ModelResponse, ToolCall
from src.report import ReportBuilder, TEMPLATE
from src.schemas import ReportInput
from src.tools.base import BaseTool, EvidenceRecord
from src.variant_validation import ValidatedVariant

CHAPTER_URL = "https://www.ncbi.nlm.nih.gov/books/NBK1283/"
CHAPTER = ("GeneReviews: Primary Hyperoxaluria Type 1 (NBK1283)\nMANAGEMENT: Targeted therapies: (1) Pyridoxine "
           "(vitamin B6) to reduce liver oxalate production in individuals with missense AGXT variants known to be "
           "pyridoxine responsive; (2) RNA interference (RNAi) therapeutics (lumasiran and nedosiran).")
LABEL_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc"
LABEL = ("FDA Drug Label for lumasiran:\nIndications: OXLUMO is indicated for the treatment of primary hyperoxaluria "
         "type 1 (PH1) to lower urinary and plasma oxalate levels in pediatric and adult patients.")


class _GR(BaseTool):
    def __init__(self):
        self.calls = []

    def schema(self):
        return {"type": "function", "function": {"name": "search_genereviews", "description": "chapters",
                "parameters": {"type": "object", "properties": {"gene": {"type": "string"}, "nbk": {"type": "string"}}, "required": ["gene"]}}}

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return CHAPTER, [EvidenceRecord(url=CHAPTER_URL)]


class _FDA(BaseTool):
    def schema(self):
        return {"type": "function", "function": {"name": "search_fda_labels", "description": "labels",
                "parameters": {"type": "object", "properties": {"drug": {"type": "string"}}, "required": ["drug"]}}}

    def execute(self, **kwargs):
        return LABEL, [EvidenceRecord(url=LABEL_URL)]


PATIENT = ReportInput(id="RPT-1", patient={
    "genotype": [{"gene": "AGXT", "transcript": "NM_000030.3", "variant_cdna": "c.33dup",
                  "variant_protein": "p.(Lys12GlnfsTer156)", "zygosity": "homozygous"}],
    "clinical_context": "7 year old with recurrent nephrocalcinosis and chronic kidney disease",
})


def call(name, **args):
    return ModelResponse(text="", tool_calls=[ToolCall(name=name, arguments=args)], raw="<tool_call>", thinking="")


def say(text):
    return ModelResponse(text=text, tool_calls=[], raw=text, thinking="")


@pytest.fixture
def scripted(monkeypatch):
    """Returns a function that installs the scripted turns and builds the agent."""
    monkeypatch.setattr(report_module, "validate_variant",
                        lambda t, c: ValidatedVariant(submitted=f"{t}:{c}", valid=True, transcript_variant=f"{t}:{c}",
                                                      genomic_variant="NC_000002.12:g.240868898dup", gene_symbol="AGXT"))

    def install(turns):
        queue = list(turns)

        def fake_generate(model, tokenizer, messages, tools=None, **kwargs):
            return queue.pop(0)
        monkeypatch.setattr(agent_module, "generate", fake_generate)
        monkeypatch.setattr(report_module, "generate", fake_generate)
        gr = _GR()
        agent = Agent(model=None, tokenizer=None, tools=[gr, _FDA()])
        return ReportBuilder(agent), gr, agent
    return install


RECORDS = (
    '[{"therapy": "lumasiran", "status": "approved", "indication": "PH1 to lower urinary and plasma oxalate", '
    '"applies_to_patient": "yes", "source_url": "%s", '
    '"quote": "OXLUMO is indicated for the treatment of primary hyperoxaluria type 1 (PH1) to lower urinary and plasma oxalate levels in pediatric and adult patients."}, '
    '{"therapy": "pyridoxine", "status": "approved", "indication": "missense variants", "applies_to_patient": "no", '
    '"source_url": "https://example.org/made-up", "quote": "Pyridoxine (vitamin B6) to reduce liver oxalate production"}]'
) % LABEL_URL


def test_goal_one_records_are_checked_judged_and_rendered(scripted):
    builder, gr, agent = scripted([
        call("search_genereviews", nbk="NBK1283"),            # gene is bound by the harness
        call("search_fda_labels", drug="lumasiran"),
        say(RECORDS),
        say("SUPPORTED\nThe label states the indication."),  # evaluator for lumasiran
    ])

    out = builder.build(PATIENT)

    assert gr.calls == [{"nbk": "NBK1283", "gene": "AGXT"}]            # bound argument filled by code
    md = out.report_markdown
    headings = [l for l in md.splitlines() if l.startswith("#")]
    expected = ["# Therapeutic Actionability Report: RPT-1"]
    for section, subs in TEMPLATE:
        expected.append(f"## {section}")
        expected += [f"### {s}" for s in subs]
    assert headings == expected                                          # every section, in order
    assert "**lumasiran** (approved)" in md and LABEL_URL in md
    assert "pyridoxine" not in md.split("### Existing Targeted Therapies")[1].split("###")[0]
    assert "record_dropped:no_source:pyridoxine" in agent.last_trace["events"]
    assert "record_verdict:lumasiran:SUPPORTED" in agent.last_trace["events"]
    assert agent.last_trace["goals"]["goal1_existing_targeted"]["sources_consulted"] == sorted([CHAPTER_URL, LABEL_URL])
    assert "Not yet assessed in this version." in md                   # the unbuilt goals say so


def test_evaluator_not_supported_drops_the_record(scripted):
    builder, _, agent = scripted([
        call("search_fda_labels", drug="lumasiran"),
        say(RECORDS),
        say("NOT SUPPORTED\nThe quote does not mention the variant."),
    ])

    md = builder.build(PATIENT).report_markdown

    assert "**lumasiran**" not in md
    assert "None found. Sources consulted: " + LABEL_URL in md
    assert "record_verdict:lumasiran:NOT_SUPPORTED" in agent.last_trace["events"]


def test_quote_not_in_source_drops_the_record(scripted):
    bad = RECORDS.replace("OXLUMO is indicated for the treatment of primary hyperoxaluria type 1 (PH1)",
                          "OXLUMO cures primary hyperoxaluria type 1 (PH1)")
    builder, _, agent = scripted([call("search_fda_labels", drug="lumasiran"), say(bad)])

    md = builder.build(PATIENT).report_markdown

    assert "**lumasiran**" not in md
    assert "record_dropped:quote_not_in_source:lumasiran" in agent.last_trace["events"]


def test_prose_instead_of_json_gets_one_retry(scripted):
    builder, _, agent = scripted([
        call("search_fda_labels", drug="lumasiran"),
        say("Lumasiran is approved for PH1 per the label."),      # not a JSON array
        say(RECORDS),                                             # the retry, thinking off
        say("SUPPORTED\nStated in the label."),
    ])

    md = builder.build(PATIENT).report_markdown

    assert "records_unparsable:goal1_existing_targeted" in agent.last_trace["events"]
    assert "**lumasiran** (approved)" in md


def test_answering_without_a_lookup_gets_the_evidence_reminder(scripted):
    builder, _, agent = scripted([
        say("[]"),                                                 # no lookup
        call("search_fda_labels", drug="lumasiran"),
        say("[]"),
    ])

    md = builder.build(PATIENT).report_markdown

    assert "evidence_reminder" in agent.last_trace["events"]
    assert "None found." in md
