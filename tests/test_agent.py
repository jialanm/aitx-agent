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
from src.tools.genereviews import GeneReviewsTool


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


class TestExtractCondition:
    """Test the _extract_condition static method."""

    def test_diagnosed_with(self):
        ctx = "A 14-year-old male diagnosed with Long QT Syndrome type 1 (LQT1)."
        result = Agent._extract_condition(ctx)
        assert "Long QT Syndrome" in result

    def test_diagnosis_of(self):
        ctx = "A 15-year-old male with a diagnosis of cystic fibrosis presents with chronic symptoms."
        result = Agent._extract_condition(ctx)
        assert "cystic fibrosis" in result

    def test_fallback_preamble_strip(self):
        ctx = "A 42-year-old female with triple-negative breast cancer."
        result = Agent._extract_condition(ctx)
        assert "breast cancer" in result

    def test_empty_context(self):
        result = Agent._extract_condition("")
        assert result == ""


class TestGeneReviewsTitleRelevance:
    """Test GeneReviews title relevance scoring."""

    def test_gene_in_title_scores_high(self):
        # Title with gene name should score higher
        score_lqts = GeneReviewsTool._title_relevance_score(
            "KCNQ1 Long QT Syndrome Overview", "KCNQ1"
        )
        score_bws = GeneReviewsTool._title_relevance_score(
            "Beckwith-Wiedemann Syndrome", "KCNQ1"
        )
        assert score_lqts > score_bws

    def test_overview_bonus(self):
        score_overview = GeneReviewsTool._title_relevance_score(
            "Long QT Syndrome Overview", "KCNQ1"
        )
        score_plain = GeneReviewsTool._title_relevance_score(
            "Some Other Syndrome", "KCNQ1"
        )
        assert score_overview > score_plain

    def test_unrelated_penalized(self):
        score = GeneReviewsTool._title_relevance_score(
            "Beckwith-Wiedemann Syndrome", "KCNQ1"
        )
        assert score < 0


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
        assert trace["events"] == []
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

    def test_forced_first_tool_is_recorded_as_an_event(self, monkeypatch):
        # Model answers without calling a tool, so the agent forces one.
        self._scripted(monkeypatch, [
            ModelResponse(text="Yes", tool_calls=[], raw="Yes", thinking=""),
        ])
        agent = Agent(model=None, tokenizer=None, tools=[self._Tool()])

        agent.run(TaskInput(**self.TASK))

        assert "forced_first_tool" in agent.last_trace["events"]
        assert agent.last_trace["tool_calls"][0]["origin"] == "forced"
