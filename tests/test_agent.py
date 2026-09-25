"""Tests for the agent module.

Note: Full agent tests require GPU and model loading.
These tests focus on the non-model parts.
"""

import json
import pytest

from src.agent import Agent
from src.model import ToolCall, ModelResponse, parse_response
from src.schemas import TaskInput, TaskOutput
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
