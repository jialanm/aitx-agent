"""Does the model's own description of the variant match what VEP returned?

An offline audit over a recorded trace: no model, no GPU. It reads each loop
turn's thinking and text, keeps the sentences that make a claim about this
patient's variant, and compares the consequence word used against the terms
the Ensembl block reported.

Two cases are not contradictions and are excluded:
  - a frameshift creates a premature stop downstream, so "premature stop"
    beside VEP's frameshift_variant is correct, not a mismatch;
  - a negated sentence ("this variant is not a nonsense mutation", "inframe,
    meaning no frameshift") states the opposite of the word it contains;
  - "nonsense-mediated decay" names the mechanism, not the variant's class;
  - a hypothetical ("if the deletion leads to a frameshift") asserts nothing.

Usage: uv run python scripts/audit_consequence_claims.py [--trace PATH ...]
"""

from __future__ import annotations

import argparse
import json
import re

WORDS = [
    (r"\bframe[- ]?shift(?:ing|ed)?\b", "frameshift_variant"),
    (r"\bnonsense\b(?!-mediated|\s+mediated)", "stop_gained"),   # not the decay mechanism's name
    (r"\bpremature (?:stop|termination)\b", "stop_gained"),
    (r"\bmissense\b", "missense_variant"),
    (r"\bin[- ]?frame\b", "inframe"),
    (r"\bstart (?:codon )?(?:lost|loss)\b", "start_lost"),
]
ABOUT_THE_VARIANT = re.compile(
    r"\b(?:this|the|patient'?s?) (?:variant|mutation|deletion|duplication|insertion|change)\b"
    r"|\bthe patient has\b", re.I)
HYPOTHETICAL = re.compile(r"\b(?:if|whether|would|could|might|unless|suppose)\b", re.I)
NEGATED = re.compile(r"\b(?:not|no|isn'?t|aren'?t|rather than|unlike|instead of)\b", re.I)


def consequences(block: str) -> list[str]:
    terms: list[str] = []
    for line in re.findall(r"^Consequence[^:]*:\s*(.+)$", block or "", re.M):
        terms += [t.strip() for t in line.split(",") if t.strip()]
    return list(dict.fromkeys(terms))


def mismatch(text: str, terms: list[str]) -> tuple[str, str] | None:
    """(word used, sentence) for a claim about this variant that VEP's terms contradict."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text or ""):
        if not ABOUT_THE_VARIANT.search(sentence) or NEGATED.search(sentence) or HYPOTHETICAL.search(sentence):
            continue
        for pattern, claimed in WORDS:
            m = re.search(pattern, sentence, re.I)
            if not m:
                continue
            if claimed == "inframe":
                if not any(t.startswith("inframe_") for t in terms):
                    return m.group(0), sentence.strip()
            elif claimed == "stop_gained" and "frameshift_variant" in terms and "nonsense" not in m.group(0).lower():
                continue          # a frameshift does create a premature stop
            elif claimed not in terms:
                return m.group(0), sentence.strip()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", nargs="+", default=["data/eval_trace_32b_run5.json", "data/eval_trace_32b_run6.json"])
    args = ap.parse_args()
    for path in args.trace:
        traces = json.load(open(path))["traces"]
        hits = 0
        print(f"=== {path}")
        for r in traces:
            vep = [c for c in r["tool_calls"] if c["tool"] == "query_ensembl"]
            terms = consequences(vep[0].get("summary", "")) if vep else []
            if not terms:
                continue
            for i, m in enumerate(r["model_turns"]):
                if m.get("kind") != "loop":
                    continue
                found = mismatch(f"{m.get('thinking','')}\n{m.get('text','')}", terms)
                if found:
                    word, sentence = found
                    hits += 1
                    print(f"  {r['id']} turn {i}: wrote {word!r}, VEP reported {', '.join(terms)}")
                    print(f"      {sentence[:150]}")
                    print(f"      answer: {r.get('answer')}")
                    break
        print(f"  questions with a contradicted description: {hits} of {len(traces)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
