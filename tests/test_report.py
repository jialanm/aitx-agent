"""The report chain with a scripted model and stub tools: five goals in order,
records parsed, checked against their sources, judged by the evaluator, kept
out of a second section, sorted into subsections, and rendered in the
template's order with the summary and outcome measures written by code."""

from __future__ import annotations

import pytest

import src.agent as agent_module
import src.report as report_module
from src.agent import Agent
from src.model import ModelResponse, ToolCall
from src.report import GOALS, ReportBuilder, TEMPLATE
from src.schemas import ReportInput
from src.tools.base import BaseTool, EvidenceRecord
from src.variant_validation import ValidatedVariant

CHAPTER_URL = "https://www.ncbi.nlm.nih.gov/books/NBK1283/"
CHAPTER = ("GeneReviews: Primary Hyperoxaluria Type 1 (NBK1283)\nMANAGEMENT: Targeted therapies: (1) Pyridoxine "
           "(vitamin B6) to reduce liver oxalate production in individuals with missense AGXT variants known to be "
           "pyridoxine responsive; (2) RNA interference (RNAi) therapeutics (lumasiran and nedosiran). Supportive care: "
           "Goal is to reduce stone formation and preserve kidney function. Agents/circumstances to avoid: "
           "intake of vitamin C exceeding the recommended daily allowance; loop diuretics.")
LABEL_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc"
LABEL = ("FDA Drug Label for lumasiran:\nIndications: OXLUMO is indicated for the treatment of primary hyperoxaluria "
         "type 1 (PH1) to lower urinary and plasma oxalate levels in pediatric and adult patients.")
TRIAL_URL = "https://clinicaltrials.gov/study/NCT00000001"
TRIAL = ("NCT ID: NCT00000001\nTitle: A study of X in PH1\nStatus: RECRUITING\nEligibility criteria:\nInclusion Criteria\n"
         "* Age 2 years and older\nExclusion Criteria\n* eGFR below 30 mL/min/1.73 m2")
N1C_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12120168/"
N1C = ("N1C VARIANT guidelines. Canonical exon skipping: for truncating loss-of-function variants, an ASO skips the "
       "variant's exon to restore the reading frame. An exon is NOT eligible if any of these holds: it is the first "
       "coding exon; it is the last coding exon.")


def _stub(name, params, text, url, calls=None):
    class _T(BaseTool):
        def schema(self):
            return {"type": "function", "function": {"name": name, "description": name,
                    "parameters": {"type": "object", "properties": {p: {"type": "string"} for p in params}, "required": []}}}

        def execute(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            return text, [EvidenceRecord(url=url)]
    return _T()


PATIENT = ReportInput(id="RPT-1", patient={
    "genotype": [{"gene": "AGXT", "transcript": "NM_000030.3", "variant_cdna": "c.33dup",
                  "variant_protein": "p.(Lys12GlnfsTer156)", "zygosity": "homozygous"}],
    "clinical_context": "7 year old with recurrent nephrocalcinosis and chronic kidney disease",
})


def call(name, **args):
    return ModelResponse(text="", tool_calls=[ToolCall(name=name, arguments=args)], raw="<tool_call>", thinking="")


def say(text):
    return ModelResponse(text=text, tool_calls=[], raw=text, thinking="")


def record(**fields):
    import json
    return json.dumps([fields])


@pytest.fixture
def scripted(monkeypatch):
    """Installs the scripted turns and builds the agent with stub tools for every goal."""
    monkeypatch.setattr(report_module, "validate_variant",
                        lambda t, c: ValidatedVariant(submitted=f"{t}:{c}", valid=True, transcript_variant=f"{t}:{c}",
                                                      genomic_variant="NC_000002.12:g.240868898dup", gene_symbol="AGXT"))

    def install(turns):
        queue = list(turns)

        def fake_generate(model, tokenizer, messages, tools=None, **kwargs):
            assert queue, "the script ran out of turns"
            return queue.pop(0)
        monkeypatch.setattr(agent_module, "generate", fake_generate)
        monkeypatch.setattr(report_module, "generate", fake_generate)
        gr_calls, up_calls = [], []
        tools = [
            # step 1 of the chain fetches the consequence; the protein position it prints is bound into UniProt
            _stub("query_ensembl", ["genomic_hgvs", "transcript"],
                  "VEP (Ensembl, GRCh38) for NC_000002.12:g.240868898dup on transcript NM_000030.3:\nFrame (VEP): frameshift_variant; 1 base inserted\n"
                  "Consequence: frameshift_variant\nExon: 1/11\nProtein position: 11\nHGVS protein: NP_000021.1:p.Lys12GlnfsTer156",
                  "https://www.ensembl.org/Homo_sapiens/Location/View?r=2:240868897-240868898"),
            _stub("search_genereviews", ["gene", "nbk"], CHAPTER, CHAPTER_URL, gr_calls),
            _stub("search_fda_labels", ["drug"], LABEL, LABEL_URL),
            _stub("search_pubmed", ["query", "gene"], "PubMed search: x\nTotal results: 0", "https://pubmed.ncbi.nlm.nih.gov/1/"),
            _stub("genetics_reference", ["topic"], "Genetics reference, reading-frame: ...", "https://pubmed.ncbi.nlm.nih.gov/3384440/"),
            _stub("search_clinical_trials", ["query", "nct_id", "status_filter"], TRIAL, TRIAL_URL),
            _stub("search_omim", ["gene"], "OMIM: PH1", "https://www.omim.org/entry/259900"),
            _stub("query_pharos", ["gene"], "PHAROS target AGXT: no approved drugs", "https://pharos.nih.gov/targets/P21549"),
            _stub("n1c_aso_guidelines", ["section"], N1C, N1C_URL),
            _stub("clingen_dosage", ["gene"], "ClinGen dosage AGXT: no curation", "https://search.clinicalgenome.org/kb/genes/HGNC:341"),
            _stub("query_uniprot", ["gene", "protein_position"], "Protein: AGT (P21549)", "https://www.uniprot.org/uniprotkb/P21549/entry", up_calls),
        ]
        agent = Agent(model=None, tokenizer=None, tools=tools)
        return ReportBuilder(agent), agent, gr_calls, up_calls
    return install


LUMASIRAN = record(therapy="lumasiran", status="approved", indication="PH1 to lower urinary and plasma oxalate",
                   applies_to_patient="yes", source_url=LABEL_URL,
                   quote="OXLUMO is indicated for the treatment of primary hyperoxaluria type 1 (PH1) to lower urinary and plasma oxalate levels in pediatric and adult patients.",
                   reasoning="")
SUPPORTIVE = (
    '[{"agent": "hydration", "kind": "supportive", "purpose": "reduce stone formation", "applies_to_patient": "yes", '
    f'"source_url": "{CHAPTER_URL}", "quote": "Goal is to reduce stone formation and preserve kidney function.", "reasoning": ""}}, '
    '{"agent": "loop diuretics", "kind": "avoid", "purpose": "to avoid", "applies_to_patient": "yes", '
    f'"source_url": "{CHAPTER_URL}", "quote": "intake of vitamin C exceeding the recommended daily allowance; loop diuretics", "reasoning": ""}}, '
    '{"agent": "lumasiran", "kind": "supportive", "purpose": "repeat of Tier 1", "applies_to_patient": "yes", '
    f'"source_url": "{CHAPTER_URL}", "quote": "RNA interference (RNAi) therapeutics (lumasiran and nedosiran)", "reasoning": ""}}]'
)
TRIAL_REC = record(nct_id="NCT00000001", title="A study of X in PH1", status="RECRUITING",
                   eligibility="conditional: If eGFR is 30 or above, then eligible; otherwise excluded",
                   source_url=TRIAL_URL, quote="* eGFR below 30 mL/min/1.73 m2", reasoning="the record gives chronic kidney disease but no eGFR")
REPURP = record(compound="compound-x", mechanism="raises AGT activity", human_precedent="no",
                rationale="loss of function needs activity raised", outcome_measure="urinary oxalate excretion",
                source_url="https://pharos.nih.gov/targets/P21549",
                quote="PHAROS target AGXT: no approved drugs", reasoning="no approved drug acts on AGT; a chaperone would raise activity")
BESPOKE = record(strategy="exon skipping", verdict="not eligible", basis="the variant is in the first coding exon",
                 outcome_measure="none stated",
                 source_url=N1C_URL, quote="it is the first coding exon", reasoning="the patient block places c.33dup in exon 1 of 11")

SUPPORTED = say("SUPPORTED\nStated in the source.")


def full_script():
    return [
        call("search_genereviews", nbk="NBK1283"), call("search_fda_labels", drug="lumasiran"), say(LUMASIRAN), SUPPORTED,   # goal 1
        call("search_genereviews", nbk="NBK1283"), say(SUPPORTIVE), SUPPORTED, SUPPORTED, SUPPORTED,                            # goal 2 (3 records)
        call("search_clinical_trials", query="AGXT"), call("search_clinical_trials", nct_id="NCT00000001"), say(TRIAL_REC), SUPPORTED,  # goal 3
        call("query_pharos"), say(REPURP), SUPPORTED,                                                                             # goal 4
        call("n1c_aso_guidelines"), call("query_uniprot"), say(BESPOKE), SUPPORTED,                                              # goal 5
    ]


def test_five_goals_fill_every_section_in_order(scripted):
    builder, agent, gr_calls, up_calls = scripted(full_script())

    md = builder.build(PATIENT).report_markdown

    headings = [l for l in md.splitlines() if l.startswith("#")]
    expected = ["# Therapeutic Actionability Report for Patient RPT-1"]
    for section, subs in TEMPLATE:
        expected.append(f"## {section}")
        expected += [f"### {s}" for s in subs]
    assert headings == expected
    assert "Not yet assessed" not in md
    assert gr_calls[0] == {"nbk": "NBK1283", "gene": "AGXT"}                     # bound by the harness
    assert up_calls == [{"gene": "AGXT", "protein_position": 11}]                 # protein position from VEP, bound
    assert [g["key"] for g in GOALS] == [e.split(":")[1] for e in agent.last_trace["events"] if e.startswith("goal_start:")]


def test_records_are_sorted_into_their_subsections_and_repeats_dropped(scripted):
    builder, agent, _, _ = scripted(full_script())

    md = builder.build(PATIENT).report_markdown

    def section(name):
        return md.split(f"### {name}")[1].split("###")[0]
    assert "**hydration**" in section("Supportive Management") and "loop diuretics" not in section("Supportive Management")
    assert "**loop diuretics**" in section("Contraindications & Precautions")
    assert "lumasiran" not in section("Supportive Management")                   # already in Tier 1
    assert "record_dropped:repeated:lumasiran" in agent.last_trace["events"]
    assert "**compound-x**" in section("Drug Repurposing") and "compound-x" not in section("Off-label or Expanded-Access Therapies")
    assert "**exon skipping**: not eligible." in section("Bespoke Therapies")
    assert "Reasoning: the patient block places c.33dup in exon 1 of 11" in section("Bespoke Therapies")


def test_summary_states_conditional_actions_and_outcome_measures_rank_tier3(scripted):
    builder, _, _, _ = scripted(full_script())

    md = builder.build(PATIENT).report_markdown
    summary = md.split("## Summary")[1].split("## Tier 1")[0]
    outcomes = md.split("### Outcome Measures & Ranking")[1]

    assert "Tier 1: established targeted therapy: lumasiran (applies: yes)." in summary
    assert "Avoid or modify: loop diuretics." in summary
    assert "NCT00000001 (conditional: If eGFR is 30 or above, then eligible; otherwise excluded)" in summary
    assert "Conditional actions, depending on information not in the patient record: Clinical Trial Eligibility, NCT00000001: If eGFR is 30 or above" in summary
    assert "Ranking rule:" in outcomes
    assert "1. compound-x (repurposing; no human precedent). Outcome measures: efficacy: urinary oxalate excretion;" in outcomes
    assert "exon skipping" not in outcomes


def test_source_check_and_evaluator_drop_records(scripted):
    bad_url = LUMASIRAN.replace(LABEL_URL, "https://example.org/made-up")
    builder, agent, _, _ = scripted([
        call("search_fda_labels", drug="lumasiran"), say(bad_url), say("NOT SUPPORTED\nThe quote names the drug only."),  # goal 1: URL replaced, evaluator rejects
        call("search_genereviews"), say(SUPPORTIVE.replace("Goal is to reduce stone formation", "Goal is to cure")),  # record 1 quote wrong
        SUPPORTED, SUPPORTED,                                                              # evaluator: loop diuretics, lumasiran-repeat (then dropped as a repeat? no: tier 1 is empty)
        call("search_clinical_trials", query="AGXT"), say("[]"),
        call("query_pharos"), say("[]"),
        call("n1c_aso_guidelines"), say("[]"),
    ])

    md = builder.build(PATIENT).report_markdown
    events = agent.last_trace["events"]

    # The lumasiran record cited a made-up URL but its quote is verbatim in the label the
    # tool returned, so the URL is replaced and the record goes on to the evaluator.
    assert f"record_url_replaced:lumasiran:https://example.org/made-up->{LABEL_URL}" in events
    assert "record_dropped:quote_not_in_source:hydration" in events
    assert "record_verdict:lumasiran:NOT_SUPPORTED" in events          # the evaluator then rejects it
    assert "None found. Sources consulted: " + LABEL_URL in md
    assert "**loop diuretics**" in md and "**hydration**" not in md
    assert "record_verdict:lumasiran:SUPPORTED" in events               # goal 2's lumasiran record, no longer a repeat since Tier 1 is empty


def test_prose_instead_of_json_gets_one_retry_and_a_failed_goal_does_not_lose_the_others(scripted, monkeypatch):
    builder, agent, _, _ = scripted([
        call("search_fda_labels", drug="lumasiran"), say("Lumasiran is approved for PH1 per the label."), say(LUMASIRAN), SUPPORTED,
        call("search_genereviews"), say("[]"),
        call("search_clinical_trials", query="AGXT"), say("[]"),
        call("query_pharos"), say("[]"),
        call("n1c_aso_guidelines"), say("[]"),
    ])
    original = builder._run_goal

    def failing(goal, *a, **k):
        if goal["key"] == "goal3_trials":
            raise RuntimeError("registry down")
        return original(goal, *a, **k)
    monkeypatch.setattr(builder, "_run_goal", failing)

    md = builder.build(PATIENT).report_markdown

    assert "records_unparsable:goal1_existing_targeted" in agent.last_trace["events"]
    assert "**lumasiran** (approved)" in md
    assert "goal_failed:goal3_trials:registry down" in agent.last_trace["events"]
    assert "### Clinical Trial Eligibility\nNone found." in md
    assert "### Bespoke Therapies\nNone found." in md


def test_multi_fragment_quote_is_verbatim_if_each_fragment_is(scripted):
    frag = record(nct_id="NCT00000001", title="A study of X in PH1", status="RECRUITING",
                  eligibility="conditional: If eGFR is 30 or above, then eligible; otherwise excluded",
                  source_url=TRIAL_URL, quote="* Age 2 years and older ... * eGFR below 30 mL/min/1.73 m2", reasoning="")
    builder, agent, _, _ = scripted([
        call("search_fda_labels", drug="lumasiran"), say("[]"), say("[]"),       # goal 1: [] then the empty-section reminder, [] again
        call("search_genereviews"), say("[]"), say("[]"),
        call("search_clinical_trials", query="AGXT"), call("search_clinical_trials", nct_id="NCT00000001"), say(frag), SUPPORTED,
        call("query_pharos"), say("[]"), say("[]"),
        call("n1c_aso_guidelines"), say("[]"), say("[]"),
    ])

    md = builder.build(PATIENT).report_markdown

    assert "**NCT00000001**" in md
    assert "empty_section_reminder:goal1_existing_targeted:search_genereviews,search_pubmed" in agent.last_trace["events"]


def test_short_list_item_quote_is_accepted(scripted):
    short = SUPPORTIVE.replace('"quote": "intake of vitamin C exceeding the recommended daily allowance; loop diuretics"', '"quote": "loop diuretics"')
    builder, agent, _, _ = scripted([
        call("search_fda_labels", drug="lumasiran"), say(LUMASIRAN), SUPPORTED,
        call("search_genereviews"), say(short), SUPPORTED, SUPPORTED,
        call("search_clinical_trials", query="AGXT"), say("[]"), say("[]"),
        call("query_pharos"), say("[]"), say("[]"),
        call("n1c_aso_guidelines"), say("[]"), say("[]"),
    ])

    md = builder.build(PATIENT).report_markdown

    assert "**loop diuretics**" in md

