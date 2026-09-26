"""The eval writes one tool trace per question beside the results file."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import src.agent as agent_module
from src.model import ModelResponse, ToolCall
from src.tools.base import BaseTool, EvidenceRecord

# scripts/ is not a package; load eval.py as a module by path.
_spec = importlib.util.spec_from_file_location(
    "eval_script", Path(__file__).parent.parent / "scripts" / "eval.py"
)
eval_script = importlib.util.module_from_spec(_spec)
sys.modules["eval_script"] = eval_script
_spec.loader.exec_module(eval_script)


class _Tool(BaseTool):
    """Stands in for every real tool so nothing reaches the network."""

    def __init__(self, name: str = "search_clinvar"):
        self._name = name

    def schema(self):
        return {"type": "function", "function": {"name": self._name, "parameters": {}}}

    def execute(self, **kwargs):
        return "one record", [EvidenceRecord(url="https://www.ncbi.nlm.nih.gov/clinvar/variation/1/")]


QUESTION = {
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


@pytest.mark.parametrize("output_path, expected", [
    ("data/eval_results_baseline.json", "data/eval_trace_baseline.json"),
    ("data/eval_results.json", "data/eval_trace.json"),
    ("runs/stats.json", "runs/stats_trace.json"),
])
def test_trace_path_follows_results_path(output_path, expected):
    assert eval_script.trace_path_for(output_path) == expected


def test_eval_writes_one_trace_per_question(tmp_path, monkeypatch):
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([QUESTION]))
    answers = tmp_path / "a.json"
    answers.write_text(json.dumps([{"id": "AITX-TEST", "answer_expected": "Yes"}]))
    results = tmp_path / "eval_results_test.json"
    outputs = tmp_path / "eval_outputs_test.json"

    # No model: generate() is scripted to call one tool, then answer.
    monkeypatch.setattr(eval_script, "load_model", lambda quantize=False: (None, None))
    call = ToolCall(name="search_clinvar", arguments={"gene": "CFTR", "variant": "c.1521_1523del"})
    queue = [ModelResponse(text="", tool_calls=[call], raw="<tool_call>", thinking="")]
    monkeypatch.setattr(
        agent_module, "generate",
        lambda *a, **k: queue.pop(0) if queue else ModelResponse(text="Yes", tool_calls=[], raw="Yes", thinking=""),
    )
    # Every tool class becomes a stand-in; the first keeps the real tool
    # name so the scripted call reaches it.
    stand_ins = {"ClinVarTool": "search_clinvar", "EnsemblTool": "query_ensembl",
                 "GeneReviewsTool": "search_genereviews", "ClinicalTrialsTool": "search_clinical_trials",
                 "PubMedTool": "search_pubmed", "UniProtTool": "query_uniprot",
                 "PharmGKBTool": "search_pharmgkb", "OMIMTool": "search_omim", "OpenFDATool": "search_openfda"}
    for cls, tool_name in stand_ins.items():
        monkeypatch.setattr(eval_script, cls, lambda n=tool_name: _Tool(n))
    monkeypatch.setattr(eval_script, "run_metadata", lambda *a, **k: {"git_commit": "test"})

    eval_script.run_evaluation(str(questions), str(answers), False, str(results), str(outputs))

    trace_file = tmp_path / "eval_trace_test.json"
    assert trace_file.exists()
    written = json.loads(trace_file.read_text())
    assert written["metadata"] == {"git_commit": "test"}
    assert [t["id"] for t in written["traces"]] == ["AITX-TEST"]
    recorded = written["traces"][0]["tool_calls"]
    assert [c["tool"] for c in recorded] == ["search_clinvar"]
    assert recorded[0]["origin"] == "model"
    assert written["traces"][0]["messages"][0]["role"] == "system"
