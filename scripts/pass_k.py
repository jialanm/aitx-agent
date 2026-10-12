"""pass^k over repeated eval runs of one configuration.

Usage: uv run python scripts/pass_k.py data/eval_results_rep1.json data/eval_results_rep2.json ...

For each question with c correct answers out of k runs, the unbiased
estimate of pass^j is C(c, j) / C(k, j); pass^j is that averaged over
questions (Chen et al. 2021 for pass@k; tau-bench 2024 for pass^k). pass^1 is
ordinary accuracy; pass^k is the share of questions answered correctly in
every run. Prints the curve, the per-question success counts, and, when
the runs' traces are beside the results, which questions changed their
first tool between runs.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path


def load_runs(paths: list[str]) -> tuple[list[dict], dict[str, list[bool]], dict[str, list[str]]]:
    metas, correct, first_tools = [], {}, {}
    for p in paths:
        results = json.load(open(p))
        metas.append(results["metadata"])
        trace_path = Path(p).with_name(Path(p).name.replace("eval_results", "eval_trace"))
        traces = {t["id"]: t for t in json.load(open(trace_path))["traces"]} if trace_path.exists() else {}
        for r in results["results"]:
            correct.setdefault(r["id"], []).append(bool(r.get("correct")))
            calls = [c["tool"] for c in traces.get(r["id"], {}).get("tool_calls", []) if c.get("origin") == "model"]
            first_tools.setdefault(r["id"], []).append(calls[0] if calls else "(none)")
    return metas, correct, first_tools


def pass_k_curve(correct: dict[str, list[bool]]) -> list[float]:
    k = min(len(v) for v in correct.values())
    curve = []
    for j in range(1, k + 1):
        vals = [comb(sum(v[:k]), j) / comb(k, j) for v in correct.values()]
        curve.append(sum(vals) / len(vals))
    return curve


def main(paths: list[str]) -> None:
    metas, correct, first_tools = load_runs(paths)
    k = min(len(v) for v in correct.values())
    commits = {m.get("git_commit") for m in metas}
    if len(commits) > 1:
        print(f"WARNING: runs are from different commits: {commits}")
    print(f"{len(paths)} runs of {len(correct)} questions, commit(s) {commits}, model {metas[0].get('model_id')}, {metas[0].get('precision')}, {metas[0].get('decoding')}\n")
    curve = pass_k_curve(correct)
    print("pass^j, j = 1.." + str(k) + ":  " + "  ".join(f"{c:.3f}" for c in curve))
    print(f"pass^1 {curve[0]:.3f} = mean accuracy; pass^{k} {curve[-1]:.3f} = share correct in every run\n")
    print(f"{'id':11} {'correct/k':9} {'first tools across runs'}")
    for qid in correct:
        c = sum(correct[qid][:k])
        tools = first_tools.get(qid, [])
        mark = " *" if len(set(tools)) > 1 else ""
        print(f"{qid:11} {c}/{k:<7} " + " ".join(t.replace("search_", "").replace("query_", "") for t in tools) + mark)
    unstable = [q for q, v in correct.items() if 0 < sum(v[:k]) < k]
    changed = [q for q, t in first_tools.items() if len(set(t)) > 1]
    print(f"\nquestions sometimes right, sometimes wrong: {len(unstable)} {unstable}")
    print(f"questions whose first tool changed between runs: {len(changed)} {changed}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
