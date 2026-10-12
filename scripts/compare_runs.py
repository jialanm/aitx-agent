"""Compare two eval runs question by question: first tool, answer, and correctness.

Usage: uv run python scripts/compare_runs.py data/eval_results_run3.json data/eval_results_run4.json

Reads each run's results file and the trace file beside it
(eval_results_X.json -> eval_trace_X.json). Prints, per question, the first
tool the model called and the answer in each run, and counts how many
questions changed first tool, changed answer, and flipped between right and
wrong. These are the prompt-sensitivity numbers when the two runs differ
only in a harmless prompt property.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(results_path: str) -> dict[str, dict]:
    results = json.load(open(results_path))
    trace_path = Path(results_path).with_name(Path(results_path).name.replace("eval_results", "eval_trace"))
    traces = {t["id"]: t for t in json.load(open(trace_path))["traces"]}
    out = {}
    for r in results["results"]:
        t = traces.get(r["id"], {})
        model_calls = [c["tool"] for c in t.get("tool_calls", []) if c.get("origin") == "model"]
        out[r["id"]] = {
            "first_tool": model_calls[0] if model_calls else "(none)",
            "answer": str(r.get("response")),
            "correct": r.get("correct"),
            "variant": results["metadata"].get("prompt_variant", "none"),
        }
    return out


def compare(a: dict[str, dict], b: dict[str, dict]) -> dict:
    ids = [i for i in a if i in b]
    rows, tool_changed, answer_changed, flipped = [], 0, 0, 0
    for i in ids:
        x, y = a[i], b[i]
        tc = x["first_tool"] != y["first_tool"]
        ac = x["answer"].strip().lower() != y["answer"].strip().lower()
        fl = x["correct"] != y["correct"]
        tool_changed += tc; answer_changed += ac; flipped += fl
        rows.append((i, x["first_tool"], y["first_tool"], x["answer"][:22], y["answer"][:22], x["correct"], y["correct"]))
    return {"n": len(ids), "first_tool_changed": tool_changed, "answer_changed": answer_changed, "correctness_flipped": flipped, "rows": rows}


def main(path_a: str, path_b: str) -> None:
    a, b = load(path_a), load(path_b)
    c = compare(a, b)
    print(f"{'id':11} {'first tool A':22} {'first tool B':22} {'answer A':22} {'answer B':22} A  B")
    for i, ta, tb, aa, ab, ca, cb in c["rows"]:
        mark = "" if ta == tb else " *"
        print(f"{i:11} {ta.replace('search_','').replace('query_',''):22} {tb.replace('search_','').replace('query_',''):22} {aa:22} {ab:22} {str(ca)[0]}  {str(cb)[0]}{mark}")
    print(f"\n{c['n']} questions: first tool changed {c['first_tool_changed']}, answer changed {c['answer_changed']}, "
          f"correctness flipped {c['correctness_flipped']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
