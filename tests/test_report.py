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
        # No RxNorm over the network in the unit suite; one test patches it back.
        monkeypatch.setattr(report_module, "normalise_drug_names", lambda name: (name, []))
        monkeypatch.setattr(report_module, "is_drug_name", lambda name: name.lower() in ("lumasiran", "nedosiran"))
        monkeypatch.setattr(report_module, "drug_concept",
                            lambda name: {"rxcui": "1", "current": True, "source": ""}
                            if name.lower() in ("lumasiran", "nedosiran") else None)
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
    # the harness appends the value the patient record lacks
    assert ("NCT00000001 (conditional: If eGFR is 30 or above, then eligible; otherwise excluded. "
            "Not in the patient record: an eGFR measurement.)") in summary
    assert "What would change a decision: an eGFR measurement." in summary
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


def test_quoted_restriction_makes_applies_yes_conditional(scripted):
    # Second live report: nedosiran's label restricts to preserved kidney function (eGFR >= 30);
    # the model wrote "applies: yes" for a patient with chronic kidney disease and no eGFR.
    nedosiran = record(therapy="nedosiran", status="approved", indication="PH1 with preserved kidney function",
                       applies_to_patient="yes", source_url=LABEL_URL, reasoning="",
                       quote="OXLUMO is indicated for the treatment of primary hyperoxaluria type 1 (PH1) to lower urinary and plasma oxalate levels in pediatric and adult patients.")
    # make the stub label carry a kidney-function clause the quote includes
    import src.report as rm
    builder, agent, _, _ = scripted([
        call("search_fda_labels", drug="nedosiran"), say(nedosiran.replace("in pediatric and adult patients.", "in pediatric and adult patients with relatively preserved kidney function, e.g., eGFR >=30 mL/min/1.73 m2.")), SUPPORTED,
        call("search_genereviews"), say("[]"), say("[]"),
        call("search_clinical_trials", query="AGXT"), say("[]"), say("[]"),
        call("query_pharos"), say("[]"), say("[]"),
        call("n1c_aso_guidelines"), say("[]"), say("[]"),
    ])
    # the stub label text must contain the quoted sentence for the source check
    rm_label = LABEL.replace("in pediatric and adult patients.", "in pediatric and adult patients with relatively preserved kidney function, e.g., eGFR >=30 mL/min/1.73 m2.")
    for t in agent.tools.values():
        if t.name == "search_fda_labels":
            t.execute = lambda **kw: (rm_label, [EvidenceRecord(url=LABEL_URL)])

    md = builder.build(PATIENT).report_markdown

    assert "record_made_conditional:nedosiran:kidney function" in agent.last_trace["events"]
    assert "Applies to this patient: conditional: If the patient meets the source's restriction on kidney function" in md
    assert "Conditional actions, depending on information not in the patient record" in md


def test_model_system_evidence_cannot_make_a_bespoke_strategy_eligible(scripted):
    rat = record(strategy="other: gene replacement", verdict="likely eligible",
                 basis="base editing corrected the variant", outcome_measure="urinary oxalate",
                 source_url=N1C_URL, reasoning="the principle applies to small indels",
                 quote="Canonical exon skipping: for truncating loss-of-function variants, an ASO skips the variant's exon to restore the reading frame.")
    builder, agent, _, _ = scripted([
        call("search_fda_labels", drug="lumasiran"), say("[]"), say("[]"),
        call("search_genereviews"), say("[]"), say("[]"),
        call("search_clinical_trials", query="AGXT"), say("[]"), say("[]"),
        call("query_pharos"), say("[]"), say("[]"),
        call("n1c_aso_guidelines"), say(rat.replace("restore the reading frame.", "restore the reading frame in PH1 rats.")), SUPPORTED,
    ])
    for t in agent.tools.values():
        if t.name == "n1c_aso_guidelines":
            t.execute = lambda **kw: (N1C.replace("restore the reading frame.", "restore the reading frame in PH1 rats."), [EvidenceRecord(url=N1C_URL)])

    md = builder.build(PATIENT).report_markdown

    assert "record_verdict_reduced:other: gene replacement:model_system" in agent.last_trace["events"]
    assert "**other: gene replacement**: unable to assess." in md
    assert "No hypothesized action was recorded in Tier 3" in md       # not ranked once reduced


def test_evaluator_instruction_names_the_overreach_cases():
    from src.report import EVALUATOR_INSTRUCTION
    for phrase in ("restriction", "organ function", "model system", "general statement of goals", "NOT SUPPORTED"):
        assert phrase in EVALUATOR_INSTRUCTION


def test_trial_record_from_a_search_summary_is_reminded_then_dropped(scripted):
    """Third live report: three trial records were judged from one-line search
    summaries, the model's own thinking saying it still needed the criteria."""
    from_summary = record(nct_id="NCT00000001", title="A study of X in PH1", status="RECRUITING",
                          eligibility="likely eligible", source_url=TRIAL_URL,
                          quote="Title: A study of X in PH1", reasoning="the patient has PH1")
    builder, agent, _, _ = scripted([
        call("search_fda_labels", drug="lumasiran"), say("[]"), say("[]"),
        call("search_genereviews"), say("[]"), say("[]"),
        call("search_clinical_trials", query="AGXT"), say(from_summary),      # searched, not fetched
        say(from_summary),                                                    # after the fetch reminder: still not fetched
        call("query_pharos"), say("[]"), say("[]"),
        call("n1c_aso_guidelines"), say("[]"), say("[]"),
    ])

    md = builder.build(PATIENT).report_markdown
    events = agent.last_trace["events"]

    assert "fetch_reminder:goal3_trials:NCT00000001" in events
    assert "record_dropped:not_fetched:NCT00000001" in events
    assert "**NCT00000001**" not in md
    notice = next(m["content"] for m in agent.last_trace["messages"] if m["role"] == "system" and "rest on a search summary" in m["content"])
    assert 'Call search_clinical_trials once per item with nct_id set to the identifier' in notice
    assert '{"nct_id": "NCT00000001"}' in notice


def test_trial_record_is_kept_when_the_trial_was_fetched_by_id(scripted):
    fetched = record(nct_id="NCT00000001", title="A study of X in PH1", status="RECRUITING",
                     eligibility="likely eligible", source_url=TRIAL_URL,
                     quote="* Age 2 years and older", reasoning="the patient is 7")
    builder, agent, _, _ = scripted([
        call("search_fda_labels", drug="lumasiran"), say("[]"), say("[]"),
        call("search_genereviews"), say("[]"), say("[]"),
        call("search_clinical_trials", query="AGXT"), call("search_clinical_trials", nct_id="NCT00000001"), say(fetched), SUPPORTED,
        call("query_pharos"), say("[]"), say("[]"),
        call("n1c_aso_guidelines"), say("[]"), say("[]"),
    ])

    md = builder.build(PATIENT).report_markdown

    assert "**NCT00000001**" in md
    assert not any(e.startswith(("fetch_reminder", "record_dropped:not_fetched")) for e in agent.last_trace["events"])



class TestDescriptionCheckOnRecords:
    """A record's reasoning that calls the variant something VEP did not
    report loses that sentence, so the report never prints a description the
    retrieved data contradicts. The claim then stands or falls on its quote."""

    def test_the_contradicted_sentence_is_dropped_from_the_reasoning(self, scripted):
        builder, agent, _, _ = scripted([])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": []}
        builder.vep_terms = ["frameshift_variant"]
        goal = {"label": "therapy", "key": "goal1_existing_targeted"}
        records = [{"therapy": "ataluren",
                    "reasoning": "This variant is a missense change. Read-through agents raise protein output."}]

        kept = builder._check_descriptions(goal, records)

        assert kept[0]["reasoning"] == "Read-through agents raise protein output."
        assert kept[0]["description_dropped"] == "This variant is a missense change."
        assert "record_description:ataluren:missense" in agent.last_trace["events"]

    def test_a_description_vep_reported_is_left_alone(self, scripted):
        builder, agent, _, _ = scripted([])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": []}
        builder.vep_terms = ["frameshift_variant"]
        goal = {"label": "therapy", "key": "goal1_existing_targeted"}
        reasoning = "The patient's variant is a frameshift, so little protein is made."
        records = [{"therapy": "lumasiran", "reasoning": reasoning}]

        kept = builder._check_descriptions(goal, records)

        assert kept[0]["reasoning"] == reasoning
        assert not any(e.startswith("record_description") for e in agent.last_trace["events"])


class TestLiveClaimsOnRecords:
    """A record claiming approval or applicability needs a source about that
    entity in this goal's lookups. The record is kept and the gap is printed,
    since a fact we could not retrieve may still be true."""

    GOAL = {"label": "therapy", "key": "goal1_existing_targeted",
            "fields": {"therapy": "", "status": "", "indication": ""}}
    CHAPTER_CALL = {"tool": "search_genereviews", "arguments": {"gene": "AGXT"},
                    "summary": "GeneReviews: RNA interference therapeutics (lumasiran and nedosiran)"}
    LABEL_CALL = {"tool": "search_fda_labels", "arguments": {"drug": "nedosiran"},
                  "summary": "FDA Drug Label for nedosiran:\n\nIndications: children 2 years of age and older"}

    def _builder(self, scripted):
        builder, agent, _, _ = scripted([])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}
        return builder, agent

    def test_an_approval_claim_with_only_a_chapter_mention_is_marked(self, scripted):
        builder, agent = self._builder(scripted)
        records = [{"therapy": "nedosiran", "status": "approved",
                    "indication": "indicated for primary hyperoxaluria type 1", "reasoning": ""}]

        kept = builder._check_live_claims(self.GOAL, records, [self.CHAPTER_CALL])

        # "approved" and "indicated for" are two live claims about the same drug
        assert kept[0]["unverified"] == ["nedosiran: whether it is approved, authorised or withdrawn",
                                         "nedosiran: whether it applies to this patient"]
        assert any(e.startswith("record_unverified:nedosiran:") for e in agent.last_trace["events"])

    def test_the_mark_is_printed_with_the_citation(self, scripted):
        builder, _ = self._builder(scripted)
        record = {"quote": "RNA interference therapeutics (lumasiran and nedosiran)",
                  "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK1283/",
                  "unverified": ["nedosiran: whether it is approved, authorised or withdrawn"]}

        line = builder._cite(record)

        assert line.endswith("Not verified against a current source: nedosiran: whether it is approved, "
                             "authorised or withdrawn.")

    def test_a_claim_whose_label_this_goal_read_is_not_marked(self, scripted):
        builder, agent = self._builder(scripted)
        records = [{"therapy": "nedosiran", "status": "approved",
                    "indication": "indicated for primary hyperoxaluria type 1", "reasoning": ""}]

        kept = builder._check_live_claims(self.GOAL, records, [self.CHAPTER_CALL, self.LABEL_CALL])

        assert "unverified" not in kept[0]
        assert not any(e.startswith("record_unverified") for e in agent.last_trace["events"])


class TestRejectedRecordKeptConditional:
    """A rejection that rests on a restriction in the record's own quote keeps
    the record: patient 6's v5 report dropped nedosiran, an approved PH1 drug
    whose label limits it by age and kidney function, so the report named
    neither the drug nor the limit."""

    NEDOSIRAN = {
        "therapy": "Nedosiran", "status": "approved",
        "indication": "lower urinary oxalate levels in PH1 with preserved kidney function",
        "applies_to_patient": "yes",
        "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ace9d4bc",
        "quote": ("RIVFLOZA is indicated to lower urinary oxalate levels in children 2 years of age and older "
                  "and adults with primary hyperoxaluria type 1 (PH1) and relatively preserved kidney function, "
                  "e.g., eGFR ≥30 mL/min/1.73 m²."),
        "reasoning": "",
    }
    GOAL = {"label": "therapy", "key": "goal1_existing_targeted",
            "fields": {"therapy": "", "status": "", "indication": "", "applies_to_patient": ""},
            "claim": "{therapy} is {status} for {indication} and applies to this patient: {applies_to_patient}"}

    def test_the_record_stays_and_the_condition_is_stated(self, scripted):
        builder, agent, _, _ = scripted([say("NOT SUPPORTED\nThe quote restricts use by kidney function.")])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}

        kept = builder._evaluate(self.GOAL, [dict(self.NEDOSIRAN)])
        kept = builder._enforce_conditions(self.GOAL, kept)

        assert [r["therapy"] for r in kept] == ["Nedosiran"]
        assert "record_kept_conditional:Nedosiran" in agent.last_trace["events"]
        assert kept[0]["applies_to_patient"].startswith("conditional: If the patient meets the source's restriction")
        assert "kidney function" in kept[0]["applies_to_patient"]

    def test_a_rejection_with_no_restriction_still_drops(self, scripted):
        builder, agent, _, _ = scripted([say("NOT SUPPORTED\nThe quote names the drug only.")])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}
        record = dict(self.NEDOSIRAN, quote="OXLUMO is a HAO1-directed small interfering RNA.")

        kept = builder._evaluate(self.GOAL, [record])

        assert kept == []
        assert "record_verdict:Nedosiran:NOT_SUPPORTED" in agent.last_trace["events"]


class TestMissingMeasurementsAndNextSteps:
    """A condition names the value it needs, and the summary lists what would
    change a decision. Patient 6 is 7 years old with chronic kidney disease
    and no eGFR, so nedosiran's condition turns on a value nobody measured."""

    CONTEXT = "7 year old with recurrent nephrocalcinosis and chronic kidney disease"
    GOAL = TestRejectedRecordKeptConditional.GOAL

    def test_the_condition_names_the_value_the_patient_record_lacks(self, scripted):
        builder, agent, _, _ = scripted([])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}
        builder.clinical_context = self.CONTEXT

        kept = builder._enforce_conditions(self.GOAL, [dict(TestRejectedRecordKeptConditional.NEDOSIRAN)])

        field = kept[0]["applies_to_patient"]
        assert "Not in the patient record: an eGFR measurement." in field
        assert "the patient's age" not in field          # the age is in the clinical context
        assert kept[0]["needs"] == ["an eGFR measurement"]

    def test_the_summary_lists_what_would_change_a_decision(self):
        from src.report import ReportBuilder
        from src.schemas import Patient, ReportInput, Variant
        report_input = ReportInput(id="RPT-00006", patient=Patient(
            genotype=[Variant(gene="AGXT", transcript="NM_000030.3", variant_cdna="c.33dup",
                              variant_protein="p.(Lys12GlnfsTer156)", zygosity="homozygous")],
            clinical_context=self.CONTEXT))
        record = dict(TestRejectedRecordKeptConditional.NEDOSIRAN,
                      applies_to_patient="conditional: ...", needs=["an eGFR measurement"])

        summary = ReportBuilder._summary(report_input, {"Existing Targeted Therapies": [record]})

        assert "What would change a decision: an eGFR measurement." in summary


class TestPatientFactsAndBrandNames:
    """Patient 6's v5 report: a trial record's eligibility rested on "The
    patient is already on Lumasiran", which the patient record nowhere states
    and the evaluator accepted; and nedosiran was rejected because its own
    label says RIVFLOZA."""

    CONTEXT = "7 year old with recurrent nephrocalcinosis and chronic kidney disease"
    GOAL = {"label": "nct_id", "key": "goal3_trials",
            "fields": {"nct_id": "", "title": "", "status": "", "eligibility": ""},
            "claim": "{nct_id} is {status} and the patient is {eligibility}"}

    def test_an_unstated_patient_fact_is_marked_and_printed(self, scripted):
        builder, agent, _, _ = scripted([])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}
        builder.clinical_context = self.CONTEXT
        record = {"nct_id": "NCT06225882", "eligibility": "likely eligible",
                  "reasoning": "The patient is already on Lumasiran and is 7 years old.",
                  "quote": "Inclusion Criteria: patient with PH1 treated with Lumasiran",
                  "source_url": "https://clinicaltrials.gov/study/NCT06225882"}

        kept = builder._check_patient_facts(self.GOAL, [record])

        assert kept[0]["unstated_patient_facts"] == ["Lumasiran"]
        assert "record_patient_fact:NCT06225882:Lumasiran" in agent.last_trace["events"]
        assert builder._cite(kept[0]).endswith("Not stated in the patient record: Lumasiran.")

    def test_a_patient_fact_the_record_states_is_not_marked(self, scripted):
        builder, agent, _, _ = scripted([])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}
        builder.clinical_context = "7 year old with chronic kidney disease, treated with lumasiran since 2024"
        record = {"nct_id": "NCT06225882", "eligibility": "likely eligible",
                  "reasoning": "The patient is already on Lumasiran."}

        kept = builder._check_patient_facts(self.GOAL, [record])

        assert "unstated_patient_facts" not in kept[0]

    def test_the_evaluator_is_given_the_patient_record_and_the_generic_name(self, scripted, monkeypatch):
        import src.report as report_module
        builder, agent, _, _ = scripted([say("SUPPORTED\nThe quote states the indication.")])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}
        builder.clinical_context = self.CONTEXT
        monkeypatch.setattr(report_module, "normalise_drug_names",
                            lambda name: ("nedosiran", [{"from": name, "to": "nedosiran"}]))
        seen = {}

        def capture(model, tokenizer, messages, tools=None, **kwargs):
            seen["prompt"] = messages[-1]["content"]
            return say("SUPPORTED\nThe quote states the indication.")

        monkeypatch.setattr(report_module, "generate", capture)
        goal = TestRejectedRecordKeptConditional.GOAL
        builder._evaluate(goal, [dict(TestRejectedRecordKeptConditional.NEDOSIRAN, quote="RIVFLOZA is indicated for PH1.")])

        assert "RIVFLOZA is nedosiran" in seen["prompt"]
        assert "Patient record: 7 year old" in seen["prompt"]


class TestBracketedInsertions:
    """Patient 6's v6 report lost three real precautions: the chapter lists
    "Loop diuretics" under "Agents/circumstances to avoid", and the model
    quoted "Loop diuretics [should be avoided]", which is not verbatim."""

    CHAPTER = ("GeneReviews: Primary Hyperoxaluria Type 1 (NBK1283)\nAgents/circumstances to avoid: "
               "High doses of nonsteroidal anti-inflammatory drugs or any pharmacologic agent that can "
               "compromise kidney function; Loop diuretics; Intake of vitamin C exceeding the recommended "
               "daily allowance")

    def test_a_quote_with_the_models_insertion_is_still_verbatim(self):
        from src.report import ReportBuilder

        quote = "Loop diuretics [should be avoided]."
        assert ReportBuilder._fragments_in_text(quote, self.CHAPTER)
        assert ReportBuilder.without_insertions(quote) == "Loop diuretics."

    def test_an_invented_quote_is_still_refused(self):
        from src.report import ReportBuilder

        assert not ReportBuilder._fragments_in_text("Loop diuretics are first-line therapy [per guidance].",
                                                    self.CHAPTER)

    def test_the_printed_quote_keeps_only_the_sources_words(self, scripted):
        builder, agent, _, _ = scripted([])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}
        goal = {"label": "agent", "key": "goal2_supportive_and_precautions",
                "fields": {"agent": "", "kind": "", "purpose": ""}}
        call = {"tool": "search_genereviews", "arguments": {"section": "management"},
                "summary": self.CHAPTER, "evidence_urls": ["https://www.ncbi.nlm.nih.gov/books/NBK1283/"]}
        record = {"agent": "Loop diuretics", "kind": "avoid", "purpose": "kidney injury",
                  "quote": "Loop diuretics [should be avoided].",
                  "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK1283/"}

        kept = builder._check_sources(goal, [record], [call])

        assert len(kept) == 1
        assert "[should be avoided]" not in kept[0]["quote"]
        assert "record_quote_trimmed:Loop diuretics" in agent.last_trace["events"]


class TestHeadingAndNegativeRecords:
    """Two findings from patient 6's v7 build."""

    CHAPTER = ("GeneReviews: Primary Hyperoxaluria Type 1 (NBK1283)\n"
               "Agents/circumstances to avoid:\n"
               "Loop diuretics\n"
               "Intake of vitamin C exceeding the recommended daily allowance\n")

    def test_the_evaluator_is_given_the_heading_the_quote_sits_under(self, scripted, monkeypatch):
        """The chapter's heading is what makes "Loop diuretics" a precaution;
        without it the evaluator rejected two real ones."""
        import src.report as report_module
        builder, agent, _, _ = scripted([])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}
        goal = {"label": "agent", "key": "goal2_supportive_and_precautions",
                "fields": {"agent": "", "kind": "", "purpose": ""},
                "claim": "{agent} should be avoided"}
        call = {"tool": "search_genereviews", "arguments": {"section": "management"},
                "summary": self.CHAPTER, "evidence_urls": ["https://www.ncbi.nlm.nih.gov/books/NBK1283/"]}
        record = {"agent": "Loop diuretics", "kind": "avoid", "purpose": "kidney injury",
                  "quote": "Loop diuretics", "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK1283/"}
        kept = builder._check_sources(goal, [record], [call])
        assert kept[0]["quote_heading"] == "Agents/circumstances to avoid:"

        seen = {}

        def capture(model, tokenizer, messages, tools=None, **kwargs):
            seen["prompt"] = messages[-1]["content"]
            return say("SUPPORTED\nThe heading says these are to be avoided.")

        monkeypatch.setattr(report_module, "generate", capture)
        builder._evaluate(goal, kept)

        assert 'under this heading in the source: "Agents/circumstances to avoid:"' in seen["prompt"]

    def test_a_record_that_denies_applicability_is_not_asked_for_a_source(self, scripted):
        """"transcript knockdown: not eligible" claims nothing that a current
        source must confirm, and was given a false caveat."""
        builder, agent, _, _ = scripted([])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}
        builder.clinical_context = "7 year old"
        goal = {"label": "strategy", "key": "goal5_bespoke",
                "fields": {"strategy": "", "verdict": "", "basis": ""}}
        record = {"strategy": "transcript knockdown", "verdict": "not eligible",
                  "basis": "AGXT is haploinsufficient", "reasoning": ""}

        kept = builder._check_live_claims(goal, [record], [])

        assert "unverified" not in kept[0]
        assert not any(e.startswith("record_unverified") for e in agent.last_trace["events"])


def test_a_quote_copied_from_escaped_criteria_text_is_verbatim():
    """ClinicalTrials.gov returns criteria with markdown escapes, so a
    correctly copied quote failed on the backslashes (patient 6, v7)."""
    from src.report import ReportBuilder
    source = ("NCT ID: NCT04580420\nEligibility criteria:\nExclusion Criteria:\n"
              "3. Estimated GFR at Screening \\<30mL/min normalized to 1.73m\\^2 BSA")

    assert ReportBuilder._fragments_in_text(
        "Estimated GFR at Screening <30mL/min normalized to 1.73m\u00b2 BSA", source)
    assert not ReportBuilder._fragments_in_text(
        "Estimated GFR at Screening below 30mL/min per 1.73m\u00b2", source)


class TestHeadingInlineAndStrategyLabels:
    """Two findings from patient 6's v9 build."""

    CHAPTER = ("GeneReviews: Primary Hyperoxaluria Type 1 (NBK1283)\nMANAGEMENT: ... assessments is based on "
               "the individual's kidney function. Agents/circumstances to avoid: Intravascular volume "
               "depletion; high doses of nonsteroidal anti-inflammatory drugs; intake of vitamin C exceeding "
               "the recommended daily allowance; loop diuretics; large intake of foods high in oxalate.")

    def test_an_inline_heading_is_found(self):
        from src.report import ReportBuilder

        assert ReportBuilder._heading_above("Loop diuretics [should be avoided].",
                                            self.CHAPTER) == "Agents/circumstances to avoid:"

    def test_a_nearby_colon_that_says_nothing_is_not_passed_on(self):
        from src.report import ReportBuilder

        assert ReportBuilder._heading_above("assessments is based on the individual's kidney function",
                                            self.CHAPTER) == ""

    def test_a_strategy_label_is_not_asked_for_a_drug_source(self, scripted, monkeypatch):
        import src.report as report_module
        builder, agent, _, _ = scripted([])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}
        builder.clinical_context = "7 year old"
        monkeypatch.setattr(report_module, "is_drug_name", lambda name: False)   # RxNorm: not a drug
        goal = {"label": "strategy", "key": "goal5_bespoke",
                "fields": {"strategy": "", "verdict": "", "basis": ""}}
        record = {"strategy": "gene replacement", "verdict": "unable to assess",
                  "basis": "base editing corrected Agxt in rats", "reasoning": "It is indicated for PH1."}

        kept = builder._check_live_claims(goal, [record], [])

        assert "unverified" not in kept[0]


class TestNotCurrentInRxNorm:
    """Ataluren is in RxNorm as concept 1030912 from the World Health
    Organization's ATC classification, which RxNorm keeps but does not call
    current, so the plain name lookup missed it and the harness treated an
    approved European therapy as not a drug."""

    def test_the_report_says_what_the_register_holds(self, scripted, monkeypatch):
        import src.report as report_module
        builder, agent, _, _ = scripted([])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}
        builder.clinical_context = "boy with Duchenne muscular dystrophy"
        monkeypatch.setattr(report_module, "drug_concept",
                            lambda name: {"rxcui": "1030912", "current": False, "source": "ATC"}
                            if name.lower() == "ataluren" else None)
        goal = {"label": "therapy", "key": "goal1_existing_targeted",
                "fields": {"therapy": "", "status": "", "indication": ""}}
        record = {"therapy": "Ataluren", "status": "approved in the European Union",
                  "indication": "nonsense mutation Duchenne muscular dystrophy", "reasoning": "",
                  "quote": "ataluren slows disease progression", "source_url": "https://pubmed.ncbi.nlm.nih.gov/32851872/"}

        kept = builder._check_live_claims(goal, [record], [])

        assert "record_not_current:Ataluren" in agent.last_trace["events"]
        assert "concept 1030912 from ATC" in kept[0]["registry_note"]
        assert "approximates drugs marketed in the United States" in kept[0]["registry_note"]
        assert kept[0]["registry_note"] in builder._cite(kept[0])

    def test_a_current_drug_gets_no_note(self, scripted):
        builder, agent, _, _ = scripted([])
        agent.last_trace = {"events": [], "tool_calls": [], "messages": [], "model_turns": [], "validation": []}
        builder.clinical_context = "7 year old"
        goal = {"label": "therapy", "key": "goal1_existing_targeted",
                "fields": {"therapy": "", "status": "", "indication": ""}}

        kept = builder._check_live_claims(goal, [{"therapy": "Lumasiran", "status": "approved",
                                                  "indication": "PH1", "reasoning": ""}], [])

        assert "registry_note" not in kept[0]
