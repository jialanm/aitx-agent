"""compare_runs counts first-tool, answer and correctness changes between two runs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("compare_runs", Path(__file__).parent.parent / "scripts" / "compare_runs.py")
compare_runs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(compare_runs)


def _write(tmp_path, name, variant, rows):
    results = {"metadata": {"prompt_variant": variant}, "results": [{"id": i, "response": a, "correct": c} for i, a, c, _ in rows]}
    traces = {"traces": [{"id": i, "tool_calls": [{"tool": t, "origin": "model"}] if t else []} for i, _, _, t in rows]}
    (tmp_path / f"eval_results_{name}.json").write_text(json.dumps(results))
    (tmp_path / f"eval_trace_{name}.json").write_text(json.dumps(traces))
    return str(tmp_path / f"eval_results_{name}.json")


def test_counts_each_kind_of_change(tmp_path):
    a = _write(tmp_path, "a", "none", [("Q1", "Yes", True, "search_clinvar"), ("Q2", "None", False, "search_pubmed"), ("Q3", "66", True, "query_ensembl")])
    b = _write(tmp_path, "b", "reword", [("Q1", "Yes", True, "search_genereviews"), ("Q2", "Ataluren", True, "search_pubmed"), ("Q3", "66", True, "query_ensembl")])

    c = compare_runs.compare(compare_runs.load(a), compare_runs.load(b))

    assert (c["n"], c["first_tool_changed"], c["answer_changed"], c["correctness_flipped"]) == (3, 1, 1, 1)
