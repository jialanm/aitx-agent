"""Tests for the agent module.

Note: Full agent tests require GPU and model loading.
These tests focus on the non-model parts.
"""

import json
import pytest

from src.agent import Agent
from src.model import ToolCall, ModelResponse, parse_response
from src.schemas import TaskInput, TaskOutput
from src.tools.base import BaseTool, EvidenceRecord


class TestParseResponse:
    def test_plain_text(self):
        result = parse_response("Yes")
        assert result.text == "Yes"
        assert result.tool_calls == []
        assert result.thinking == ""

    def test_tool_call(self):
        raw = '<tool_call>{"name": "search_clinvar", "arguments": {"gene": "CFTR", "variant": "c.1521_1523del"}}</tool_call>'
        result = parse_response(raw)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_clinvar"
        assert result.tool_calls[0].arguments["gene"] == "CFTR"

    def test_thinking_block(self):
        raw = "<think>Let me think about this...</think>Yes"
        result = parse_response(raw)
        assert result.text == "Yes"
        assert "think about this" in result.thinking

    def test_thinking_and_tool_call(self):
        raw = '<think>I need to check ClinVar first.</think><tool_call>{"name": "search_clinvar", "arguments": {"gene": "BRCA1", "variant": "c.5266dup"}}</tool_call>'
        result = parse_response(raw)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_clinvar"
        assert "ClinVar" in result.thinking

    def test_malformed_tool_call(self):
        raw = '<tool_call>{"name": "bad_json"</tool_call>Some text'
        result = parse_response(raw)
        assert result.tool_calls == []
        assert "Some text" in result.text

    def test_unclosed_think_block(self):
        """Unclosed <think> block (hit max_new_tokens mid-think) should be stripped."""
        raw = "<think>The model is reasoning about beta-blockers and gets stuck in a loop..."
        result = parse_response(raw)
        assert result.text == ""
        assert "beta-blockers" in result.thinking

    def test_unclosed_think_with_text_before(self):
        """Text before an unclosed think block should be preserved."""
        raw = "beta-blockers<think>Let me reconsider this reasoning because..."
        result = parse_response(raw)
        assert result.text == "beta-blockers"
        assert "reconsider" in result.thinking

    def test_closed_then_unclosed_think(self):
        """Closed think block followed by unclosed one."""
        raw = "<think>First thought</think>Answer<think>Second thought gets cut off"
        result = parse_response(raw)
        assert result.text == "Answer"
        assert "First thought" in result.thinking
        assert "Second thought" in result.thinking


class TestSchemas:
    def test_task_input_parsing(self):
        data = {
            "id": "AITX-00001",
            "patient": {
                "genotype": [
                    {
                        "gene": "CFTR",
                        "transcript": "NM_000492.4",
                        "variant_cdna": "c.1521_1523del",
                        "variant_protein": "p.Phe508del",
                        "zygosity": "homozygous",
                    }
                ],
                "clinical_context": "A 15-year-old male with cystic fibrosis.",
            },
            "question": {
                "category": "Established_Targeted",
                "answer_format": "binary",
                "prompt": "Is this eligible?",
                "date_submitted": "2024-12-10",
            },
        }
        task = TaskInput(**data)
        assert task.id == "AITX-00001"
        assert len(task.patient.genotype) == 1
        assert task.patient.genotype[0].gene == "CFTR"

    def test_task_output_serialization(self):
        output = TaskOutput(
            id="AITX-00001",
            response="Yes",
            evidence=[],
        )
        data = json.loads(output.model_dump_json())
        assert data["id"] == "AITX-00001"
        assert data["response"] == "Yes"
        assert data["evidence"] == []


class TestAnswerValidation:
    """Test the _is_valid_answer static method."""

    def test_valid_binary_yes(self):
        assert Agent._is_valid_answer("Yes", "binary") is True

    def test_valid_binary_no(self):
        assert Agent._is_valid_answer("No", "binary") is True

    def test_invalid_binary(self):
        assert Agent._is_valid_answer("Maybe", "binary") is False

    def test_empty_binary(self):
        assert Agent._is_valid_answer("", "binary") is False

    def test_valid_numeric(self):
        assert Agent._is_valid_answer("42", "numeric_match") is True

    def test_invalid_numeric(self):
        assert Agent._is_valid_answer("not a number", "numeric_match") is False

    def test_valid_string(self):
        assert Agent._is_valid_answer("sirolimus", "string_match") is True

    def test_empty_string(self):
        assert Agent._is_valid_answer("", "string_match") is False

    def test_whitespace_only(self):
        assert Agent._is_valid_answer("   ", "string_match") is False

    DMD_OPTS = ["Golodirsen", "Viltolarsen", "Eteplirsen", "Casimersen", "Ataluren", "None"]

    def test_multiple_choice_option_is_valid(self):
        assert Agent._is_valid_answer("Ataluren", "multiple_choice", self.DMD_OPTS) is True

    def test_multiple_choice_none_option_is_valid(self):
        assert Agent._is_valid_answer("None", "multiple_choice", self.DMD_OPTS) is True

    def test_multiple_choice_non_option_is_invalid(self):
        # A drug that is real but not offered must trigger the retry, not pass.
        assert Agent._is_valid_answer("Nusinersen", "multiple_choice", self.DMD_OPTS) is False

    def test_multiple_choice_without_verified_options_accepts_text(self):
        assert Agent._is_valid_answer("anything", "multiple_choice", []) is True


class TestOptionExtraction:
    """The model call is stubbed; these test the parsing and verification around it."""

    NF1 = "In which functional domain does this variant occur? Answer choices: CSRD, TBD, GRD, Sec14-PH, HLR, NLS, SBR."

    @staticmethod
    def _stub(monkeypatch, text):
        import src.agent as agent_module
        monkeypatch.setattr(
            agent_module, "generate",
            lambda *a, **k: ModelResponse(text=text, tool_calls=[], raw=text, thinking=""),
        )
        return Agent(model=None, tokenizer=None, tools=[])

    def test_verified_list_is_returned(self, monkeypatch):
        agent = self._stub(monkeypatch, '["CSRD", "TBD", "GRD", "Sec14-PH", "HLR", "NLS", "SBR"]')
        assert agent._extract_options(self.NF1) == ["CSRD", "TBD", "GRD", "Sec14-PH", "HLR", "NLS", "SBR"]

    def test_hallucinated_option_is_rejected_and_logged(self, monkeypatch, caplog):
        agent = self._stub(monkeypatch, '["CSRD", "TBD", "Nusinersen"]')
        with caplog.at_level("WARNING", logger="src.agent"):
            assert agent._extract_options(self.NF1) == []
        assert "rejected" in caplog.text and "Nusinersen" in caplog.text

    def test_unparseable_output_is_rejected_and_logged(self, monkeypatch, caplog):
        agent = self._stub(monkeypatch, "The choices are CSRD, TBD and GRD.")
        with caplog.at_level("WARNING", logger="src.agent"):
            assert agent._extract_options(self.NF1) == []
        assert "unparseable" in caplog.text

    def test_model_error_yields_no_options(self, monkeypatch):
        import src.agent as agent_module
        def boom(*a, **k):
            raise RuntimeError("cuda")
        monkeypatch.setattr(agent_module, "generate", boom)
        assert Agent(model=None, tokenizer=None, tools=[])._extract_options(self.NF1) == []



def loop_events(trace):
    """Trace events except the per-citation ones added by the justification step."""
    return [e for e in trace["events"] if not e.startswith("citation_unsupported")]


class TestTrace:
    """run() keeps its working notes on last_trace; the model is scripted."""

    TASK = {
        "id": "AITX-TEST",
        "patient": {
            "genotype": [{"gene": "CFTR", "transcript": "NM_000492.4",
                          "variant_cdna": "c.1521_1523del",
                          "variant_protein": "p.Phe508del", "zygosity": "homozygous"}],
            "clinical_context": "A 15-year-old with cystic fibrosis.",
        },
        "question": {"category": "Established_Targeted", "answer_format": "binary",
                     "prompt": "Is this variant eligible?", "date_submitted": "2024-12-10"},
    }

    class _Tool(BaseTool):
        def schema(self):
            return {"type": "function", "function": {"name": "search_clinvar", "parameters": {}}}

        def execute(self, **kwargs):
            return "one record", [EvidenceRecord(url="https://www.ncbi.nlm.nih.gov/clinvar/variation/1/")]

    @staticmethod
    def _scripted(monkeypatch, turns):
        """generate() returns the scripted turns in order, then plain 'Yes'."""
        import src.agent as agent_module
        from src.variant_validation import ValidatedVariant
        monkeypatch.setattr(
            agent_module, "validate_variant",
            lambda t, c: ValidatedVariant(submitted=f"{t}:{c}", valid=True, transcript_variant=f"{t}:{c}",
                                          genomic_variant="NC_000007.14:g.117559592_117559594del", gene_symbol="CFTR"),
        )
        queue = list(turns)

        def fake_generate(*a, **k):
            if queue:
                return queue.pop(0)
            return ModelResponse(text="Yes", tool_calls=[], raw="Yes", thinking="")

        monkeypatch.setattr(agent_module, "generate", fake_generate)

    def test_model_tool_call_is_recorded_with_arguments_and_evidence(self, monkeypatch):
        call = ToolCall(name="search_clinvar", arguments={"gene": "CFTR", "variant": "c.1521_1523del"})
        self._scripted(monkeypatch, [
            ModelResponse(text="", tool_calls=[call], raw="<tool_call>...</tool_call>", thinking=""),
        ])
        agent = Agent(model=None, tokenizer=None, tools=[self._Tool()])

        output = agent.run(TaskInput(**self.TASK))

        assert output.response == "Yes"
        trace = agent.last_trace
        assert trace["id"] == "AITX-TEST"
        assert loop_events(trace) == []
        assert len(trace["tool_calls"]) == 1
        recorded = trace["tool_calls"][0]
        assert recorded["tool"] == "search_clinvar"
        assert recorded["arguments"] == {"gene": "CFTR", "variant": "c.1521_1523del"}
        assert recorded["origin"] == "model"
        assert recorded["evidence_urls"] == ["https://www.ncbi.nlm.nih.gov/clinvar/variation/1/"]
        assert recorded["summary"] == "one record"
        assert recorded["seconds"] >= 0
        assert trace["messages"][0]["role"] == "system"
        assert any(m["role"] == "user" and "one record" in m["content"] for m in trace["messages"])

    def test_answer_without_lookup_gets_one_reminder_and_the_model_writes_the_call(self, monkeypatch):
        # Turn 1: the model answers without a tool. Turn 2, after the reminder: it calls
        # ClinVar with its own arguments. Turn 3: it answers from the result.
        call = ToolCall(name="search_clinvar", arguments={"gene": "CFTR", "variant": "c.1521_1523del"})
        self._scripted(monkeypatch, [
            ModelResponse(text="Yes", tool_calls=[], raw="Yes", thinking=""),
            ModelResponse(text="", tool_calls=[call], raw="<tool_call>", thinking=""),
        ])
        agent = Agent(model=None, tokenizer=None, tools=[self._Tool()])

        output = agent.run(TaskInput(**self.TASK))

        assert output.response == "Yes"
        assert loop_events(agent.last_trace) == ["evidence_reminder"]
        assert [c["origin"] for c in agent.last_trace["tool_calls"]] == ["model"]
        assert agent.last_trace["tool_calls"][0]["arguments"] == {"gene": "CFTR", "variant": "c.1521_1523del"}
        reminder = next(m for m in agent.last_trace["messages"] if m["role"] == "user" and "looking anything up" in m["content"])
        assert "recommended first tool for this question is search_clinvar" in reminder["content"]

    def test_reminder_ignored_is_recorded_and_the_answer_accepted(self, monkeypatch):
        self._scripted(monkeypatch, [
            ModelResponse(text="No", tool_calls=[], raw="No", thinking=""),
            ModelResponse(text="No", tool_calls=[], raw="No", thinking=""),
        ])
        agent = Agent(model=None, tokenizer=None, tools=[self._Tool()])

        output = agent.run(TaskInput(**self.TASK))

        assert output.response == "No"
        assert loop_events(agent.last_trace) == ["evidence_reminder", "reminder_ignored"]
        assert agent.last_trace["tool_calls"] == []

    def test_validation_result_reaches_trace_and_prompt(self, monkeypatch):
        self._scripted(monkeypatch, [
            ModelResponse(text="", tool_calls=[ToolCall(name="search_clinvar", arguments={})], raw="", thinking=""),
        ])
        agent = Agent(model=None, tokenizer=None, tools=[self._Tool()])

        agent.run(TaskInput(**self.TASK))

        validation = agent.last_trace["validation"]
        assert [v["transcript_variant"] for v in validation] == ["NM_000492.4:c.1521_1523del"]
        system_prompt = agent.last_trace["messages"][0]["content"]
        assert "Validated (VariantValidator, GRCh38): NM_000492.4:c.1521_1523del" in system_prompt
        assert "Genomic position: NC_000007.14:g.117559592_117559594del" in system_prompt


class TestEmptyResultNotice:
    """A lookup with no evidence gets one notice naming the next untried tool; the model writes the next call."""

    class _Empty(BaseTool):
        def schema(self):
            return {"type": "function", "function": {"name": "search_genereviews", "parameters": {}}}

        def execute(self, **kwargs):
            return "GeneReviews has no chapter indexed to gene AGXT.", []

    class _Found(BaseTool):
        def schema(self):
            return {"type": "function", "function": {"name": "search_clinvar", "parameters": {}}}

        def execute(self, **kwargs):
            return "one record", [EvidenceRecord(url="https://www.ncbi.nlm.nih.gov/clinvar/variation/1/")]

    class _Menu(BaseTool):
        """A reply with no evidence links that is nevertheless an answer to choose from."""
        def schema(self):
            return {"type": "function", "function": {"name": "search_genereviews", "parameters": {}}}

        def execute(self, **kwargs):
            return ("GeneReviews chapters indexed to COL1A1 (3); call again with the accession that matches the patient's phenotype:\n"
                    "- NBK1244: Classic Ehlers-Danlos Syndrome (PMID 20301422)"), []

    def test_a_menu_reply_does_not_trigger_the_notice(self, monkeypatch):
        import src.agent as agent_module
        from src.variant_validation import ValidatedVariant
        monkeypatch.setattr(agent_module, "validate_variant",
                            lambda t, c: ValidatedVariant(submitted=f"{t}:{c}", valid=True, transcript_variant=f"{t}:{c}",
                                                          genomic_variant="NC_000007.14:g.117559592_117559594del", gene_symbol="CFTR"))
        turns = [
            ModelResponse(text="", tool_calls=[ToolCall(name="search_genereviews", arguments={"gene": "COL1A1"})], raw="<tool_call>", thinking=""),
            ModelResponse(text="Yes", tool_calls=[], raw="Yes", thinking=""),
            ModelResponse(text="Yes", tool_calls=[], raw="Yes", thinking=""),
        ]
        monkeypatch.setattr(agent_module, "generate", lambda *a, **k: turns.pop(0))
        agent = Agent(model=None, tokenizer=None, tools=[self._Menu()])

        agent.run(TaskInput(**TestTrace.TASK))

        assert "empty_result_notice" not in agent.last_trace["events"]
        assert not any("returned no evidence" in m["content"] for m in agent.last_trace["messages"] if m["role"] == "user")

    def test_notice_names_the_next_untried_tool_and_model_writes_the_call(self, monkeypatch):
        import src.agent as agent_module
        from src.variant_validation import ValidatedVariant
        monkeypatch.setattr(agent_module, "validate_variant",
                            lambda t, c: ValidatedVariant(submitted=f"{t}:{c}", valid=True, transcript_variant=f"{t}:{c}",
                                                          genomic_variant="NC_000007.14:g.117559592_117559594del", gene_symbol="CFTR"))
        turns = [
            ModelResponse(text="", tool_calls=[ToolCall(name="search_genereviews", arguments={"gene": "CFTR"})], raw="<tool_call>", thinking=""),
            ModelResponse(text="", tool_calls=[ToolCall(name="search_clinvar", arguments={"gene": "CFTR", "variant": "c.1521_1523del"})], raw="<tool_call>", thinking=""),
            ModelResponse(text="Yes", tool_calls=[], raw="Yes", thinking=""),
        ]
        monkeypatch.setattr(agent_module, "generate", lambda *a, **k: turns.pop(0))
        agent = Agent(model=None, tokenizer=None, tools=[self._Empty(), self._Found()])

        output = agent.run(TaskInput(**TestTrace.TASK))  # Established_Targeted: genereviews, clinvar, ...

        assert output.response == "Yes"
        assert loop_events(agent.last_trace) == ["empty_result_notice"]
        assert [(c["tool"], c["origin"]) for c in agent.last_trace["tool_calls"]] == [("search_genereviews", "model"), ("search_clinvar", "model")]
        notice = next(m for m in agent.last_trace["messages"] if m["role"] == "user" and "returned no evidence" in m["content"])
        assert "next recommended tool for this question is search_clinvar" in notice["content"]


class TestModelTurnsInTrace:
    """Every model turn is kept, including a turn-1 answer displaced by a forced lookup."""

    def test_displaced_first_answer_and_final_answer_are_recorded(self, monkeypatch):
        import src.agent as agent_module
        from src.variant_validation import ValidatedVariant
        monkeypatch.setattr(agent_module, "validate_variant",
                            lambda t, c: ValidatedVariant(submitted=f"{t}:{c}", valid=True, transcript_variant=f"{t}:{c}",
                                                          genomic_variant="NC_000002.12:g.240871433G>A", gene_symbol="AGXT"))
        turns = [
            ModelResponse(text="No", tool_calls=[], raw="No", thinking="I know this one."),   # answered without a lookup
            ModelResponse(text="Yes", tool_calls=[], raw="Yes", thinking="Still no lookup."),  # after the reminder, still none
        ]
        monkeypatch.setattr(agent_module, "generate", lambda *a, **k: turns.pop(0))
        agent = Agent(model=None, tokenizer=None, tools=[TestTrace._Tool()])

        output = agent.run(TaskInput(**TestTrace.TASK))

        recorded = agent.last_trace["model_turns"]
        assert [(m["kind"], m["iteration"], m["text"]) for m in recorded] == [("loop", 1, "No"), ("loop", 2, "Yes")]
        assert recorded[0]["thinking"] == "I know this one."
        assert loop_events(agent.last_trace) == ["evidence_reminder", "reminder_ignored"]
        assert agent.last_trace["final_text"] == "Yes"
        assert agent.last_trace["answer"] == output.response == "Yes"


class TestFabricatedIdentifierCheck:
    """An identifier the model answers must occur in some tool result; otherwise it is removed and recorded."""

    TASK = {
        "id": "AITX-00009",
        "patient": {
            "genotype": [{"gene": "SLC35A2", "transcript": "NM_005660.3", "variant_cdna": "c.3G>A",
                          "variant_protein": "p.Met1Ile", "zygosity": "heterozygous"}],
            "clinical_context": "Patient with SLC35A2-CDG who is experiencing seizures and global developmental delay",
        },
        "question": {"category": "Clinical_Trials", "answer_format": "string_match",
                     "prompt": "What clinical trial developing a new therapeutic for this condition is recruiting? Return a clinical trials ID",
                     "date_submitted": "2025-08-01"},
    }

    class _Trials(BaseTool):
        def __init__(self, summary, urls):
            self._summary, self._urls = summary, urls

        def schema(self):
            return {"type": "function", "function": {"name": "search_clinical_trials", "parameters": {}}}

        def execute(self, **kwargs):
            return self._summary, [EvidenceRecord(url=u) for u in self._urls]

    def _run(self, monkeypatch, tool, answer):
        import src.agent as agent_module
        from src.variant_validation import ValidatedVariant
        monkeypatch.setattr(agent_module, "validate_variant",
                            lambda t, c: ValidatedVariant(submitted=f"{t}:{c}", valid=True, transcript_variant=f"{t}:{c}",
                                                          genomic_variant="NC_000023.11:g.48911634C>T", gene_symbol="SLC35A2"))
        turns = [
            ModelResponse(text="", tool_calls=[ToolCall(name="search_clinical_trials", arguments={"query": "SLC35A2"})], raw="<tool_call>", thinking=""),
            ModelResponse(text=answer, tool_calls=[], raw=answer, thinking=""),
        ]
        monkeypatch.setattr(agent_module, "generate", lambda *a, **k: turns.pop(0))
        agent = Agent(model=None, tokenizer=None, tools=[tool])
        return agent.run(TaskInput(**self.TASK)), agent.last_trace

    def test_trial_id_absent_from_every_result_becomes_none(self, monkeypatch):
        tool = self._Trials("No clinical trials found for query: SLC35A2. Search URL: https://clinicaltrials.gov/search?term=SLC35A2",
                            ["https://clinicaltrials.gov/search?term=SLC35A2"])

        output, trace = self._run(monkeypatch, tool, "NCT05678901")

        assert output.response == "None"
        assert loop_events(trace) == ["fabricated_identifier:NCT05678901"]

    def test_trial_id_present_in_a_result_is_kept(self, monkeypatch):
        tool = self._Trials("Found 1 clinical trial.\nNCT ID: NCT05402384\nStatus: RECRUITING", ["https://clinicaltrials.gov/study/NCT05402384"])

        output, trace = self._run(monkeypatch, tool, "NCT05402384")

        assert output.response == "NCT05402384"
        assert loop_events(trace) == []

    def test_only_the_absent_id_is_removed_from_a_list(self, monkeypatch):
        tool = self._Trials("NCT ID: NCT05402384", ["https://clinicaltrials.gov/study/NCT05402384"])

        output, trace = self._run(monkeypatch, tool, "NCT05402384,NCT05678901")

        assert output.response == "NCT05402384"
        assert loop_events(trace) == ["fabricated_identifier:NCT05678901"]


class TestGroundedJustifications:
    """A citation's justification must quote its own retrieved text; invented quotes are replaced and recorded."""

    SOURCE = ("GeneReviews: Primary Hyperoxaluria Type 1 (NBK1283)\nMANAGEMENT: Targeted therapies: (1) Pyridoxine "
              "(vitamin B6) to reduce liver oxalate production in individuals with missense AGXT variants known to be "
              "pyridoxine responsive; (2) RNA interference (RNAi) therapeutics (lumasiran and nedosiran).")

    class _GR(BaseTool):
        def schema(self):
            return {"type": "function", "function": {"name": "search_genereviews", "parameters": {}}}

        def execute(self, **kwargs):
            return TestGroundedJustifications.SOURCE, [EvidenceRecord(url="https://www.ncbi.nlm.nih.gov/books/NBK1283/")]

    def _run(self, monkeypatch, justification_reply):
        import src.agent as agent_module
        from src.variant_validation import ValidatedVariant
        monkeypatch.setattr(agent_module, "validate_variant",
                            lambda t, c: ValidatedVariant(submitted=f"{t}:{c}", valid=True, transcript_variant=f"{t}:{c}",
                                                          genomic_variant="NC_000002.12:g.240871433G>A", gene_symbol="AGXT"))
        turns = [
            ModelResponse(text="", tool_calls=[ToolCall(name="search_genereviews", arguments={"gene": "AGXT"})], raw="<tool_call>", thinking=""),
            ModelResponse(text="Pyridoxine", tool_calls=[], raw="Pyridoxine", thinking=""),
            ModelResponse(text=justification_reply, tool_calls=[], raw=justification_reply, thinking=""),
        ]
        monkeypatch.setattr(agent_module, "generate", lambda *a, **k: turns.pop(0))
        monkeypatch.setattr(agent_module, "normalise_drug_names", lambda a: (a, []))  # no RxNorm network in tests
        task = dict(TestFabricatedIdentifierCheck.TASK)
        task["id"] = "AITX-00004"
        task["patient"] = {"genotype": [{"gene": "AGXT", "transcript": "NM_000030.3", "variant_cdna": "c.508G>A",
                                         "variant_protein": "p.(Gly170Arg)", "zygosity": "homozygous"}],
                           "clinical_context": "Recurrent nephrocalcinosis and chronic kidney disease"}
        task["question"] = {"category": "Established_Targeted", "answer_format": "string_match",
                            "prompt": "What targeted, small molecule therapy is available for this patient? Provide the generic name or None.",
                            "date_submitted": "2025-08-01"}
        agent = Agent(model=None, tokenizer=None, tools=[self._GR()])
        return agent.run(TaskInput(**task)), agent.last_trace

    def test_verbatim_quote_from_the_source_is_kept(self, monkeypatch):
        reply = ('1. QUOTE: "Pyridoxine (vitamin B6) to reduce liver oxalate production in individuals with missense AGXT '
                 'variants known to be pyridoxine responsive"\n   WHY: The patient has a missense AGXT variant.')

        output, trace = self._run(monkeypatch, reply)

        assert output.evidence[0].source == "https://www.ncbi.nlm.nih.gov/books/NBK1283/"
        assert output.evidence[0].justification.startswith('"Pyridoxine (vitamin B6) to reduce liver oxalate production')
        assert output.evidence[0].justification.endswith("The patient has a missense AGXT variant.")
        assert not any(e.startswith("citation_unsupported") for e in trace["events"])
        assert trace["model_turns"][-1]["kind"] == "justifications"

    def test_invented_quote_is_replaced_and_recorded(self, monkeypatch):
        reply = ('1. QUOTE: "Pyridoxine is first-line therapy for all AGXT patients regardless of genotype"\n'
                 '   WHY: It supports the answer.')

        output, trace = self._run(monkeypatch, reply)

        assert output.evidence[0].justification == "This record was retrieved and consulted; its returned text does not state the answer."
        assert "citation_unsupported:https://www.ncbi.nlm.nih.gov/books/NBK1283/" in trace["events"]

    def test_not_supported_is_honoured(self, monkeypatch):
        output, trace = self._run(monkeypatch, "1. QUOTE: NOT SUPPORTED\n   WHY: nothing relevant")

        assert output.evidence[0].justification == "This record was retrieved and consulted; its returned text does not state the answer."
        assert "citation_unsupported:https://www.ncbi.nlm.nih.gov/books/NBK1283/" in trace["events"]


class TestAdapterBoilerplateIsNotGrounding:
    """A citation that quotes the adapter's own 'nothing found' line is unsupported (run 3, AITX-00010)."""

    class _EmptyTrials(BaseTool):
        def schema(self):
            return {"type": "function", "function": {"name": "search_clinical_trials", "parameters": {}}}

        def execute(self, **kwargs):
            return ("No clinical trials found for query: SLC35A2 seizures. Search URL: https://clinicaltrials.gov/search?term=SLC35A2",
                    [EvidenceRecord(url="https://clinicaltrials.gov/search?term=SLC35A2")])

    def test_quoted_nothing_found_line_is_marked_unsupported(self, monkeypatch):
        import src.agent as agent_module
        from src.variant_validation import ValidatedVariant
        monkeypatch.setattr(agent_module, "validate_variant",
                            lambda t, c: ValidatedVariant(submitted=f"{t}:{c}", valid=True, transcript_variant=f"{t}:{c}",
                                                          genomic_variant="NC_000023.11:g.48911634C>T", gene_symbol="SLC35A2"))
        reply = '1. QUOTE: "No clinical trials found for query: SLC35A2 seizures."\n   WHY: No trials means not eligible.'
        turns = [
            ModelResponse(text="", tool_calls=[ToolCall(name="search_clinical_trials", arguments={"query": "SLC35A2"})], raw="<tool_call>", thinking=""),
            ModelResponse(text="No", tool_calls=[], raw="No", thinking=""),
            ModelResponse(text=reply, tool_calls=[], raw=reply, thinking=""),
        ]
        monkeypatch.setattr(agent_module, "generate", lambda *a, **k: turns.pop(0))
        task = dict(TestFabricatedIdentifierCheck.TASK)
        task["question"] = dict(task["question"], answer_format="binary", prompt="Is this patient eligible for clinical trial NCT05402384? Answer yes or no.")
        agent = Agent(model=None, tokenizer=None, tools=[self._EmptyTrials()])

        output = agent.run(TaskInput(**task))

        assert output.response == "No"
        assert output.evidence[0].justification == "This record was retrieved and consulted; its returned text does not state the answer."
        assert "citation_unsupported:https://clinicaltrials.gov/search?term=SLC35A2" in agent.last_trace["events"]

    @pytest.mark.parametrize("quote, boilerplate", [
        ("No clinical trials found for query: KCNT1", True),
        ("VEP returned nothing for NC_000002.12:g.240871433G>A: HTTP 500", True),
        ("GeneReviews has no chapter indexed to gene SLC35A2.", True),
        ("ClinVar API error: 429 Too Many Requests", True),
        ("No ClinVar record found for the patient's exact variant NM_004006.3:c.7544_9286del", True),
        ("Targeted therapies: (1) Pyridoxine (vitamin B6) to reduce liver oxalate production", False),
        ("NCT ID: NCT05402384", False),
    ])
    def test_boilerplate_patterns(self, quote, boilerplate):
        assert Agent._is_adapter_boilerplate(quote) is boilerplate
