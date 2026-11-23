"""Problem 2 dev search: score prompted trial-eligibility judgements on the TREC pairs.

One configuration = an input format x a verbalizer x a number of shots. The
model is asked once per pair, thinking off, greedy, and its first words are
read as the label. Metrics: accuracy, macro-F1, and the false-eligible rate
(pairs judged eligible whose gold label is excluded or not relevant), which
is the safety-relevant error for a trial judge. Every run writes its
predictions and metrics with the model, precision, decoding and commit.

Usage, on the GPU machine:
    uv run --group hw12 python hw12/prompt_search.py --split dev \
        --model Qwen/Qwen3-32B-FP8 --precision fp8 \
        --formats full criteria split --verbalizers words digits --shots 0 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DECODING_PRESETS, PRECISIONS, generate, load_model, set_decoding  # noqa: E402

LABELS = {0: "not relevant", 1: "excluded", 2: "eligible"}

# --- input formats: what the model reads about the trial -------------------

def fmt_full(pair: dict) -> str:
    return pair["trial_text"]


def fmt_criteria(pair: dict) -> str:
    """Title plus the criteria block only; no conditions, interventions or population line."""
    lines = pair["trial_text"].split("\n")
    title = lines[0]
    idx = next((i for i, l in enumerate(lines) if l.startswith("Eligibility criteria:")), len(lines))
    return "\n".join([title] + lines[idx:])


def fmt_split(pair: dict) -> str:
    """Full text, with the criteria block divided into numbered inclusion and exclusion items."""
    lines = pair["trial_text"].split("\n")
    idx = next((i for i, l in enumerate(lines) if l.startswith("Eligibility criteria:")), len(lines))
    head, crit = lines[:idx], "\n".join(lines[idx + 1:])
    low = crit.lower()
    cut = low.find("exclusion criteria")
    incl, excl = (crit, "") if cut < 0 else (crit[:cut], crit[cut:])

    def items(block: str, drop: str) -> list[str]:
        out = []
        for raw in block.split("\n"):
            s = raw.strip().lstrip("*-• ").strip()
            if s and drop not in s.lower() and len(s) > 3:
                out.append(s)
        return out

    inc_items = items(incl, "inclusion criteria")
    exc_items = items(excl, "exclusion criteria")
    body = ["Inclusion criteria:"] + [f"I{i}. {s}" for i, s in enumerate(inc_items, 1)] or ["Inclusion criteria: not stated"]
    body += ["Exclusion criteria:"] + ([f"E{i}. {s}" for i, s in enumerate(exc_items, 1)] or ["none listed"])
    return "\n".join(head + body)


FORMATS = {"full": fmt_full, "criteria": fmt_criteria, "split": fmt_split}

# --- verbalizers: how the label is asked for and read back ------------------

VERBALIZERS = {
    "words": {"options": "not relevant, excluded, or eligible", "answers": {0: "not relevant", 1: "excluded", 2: "eligible"}},
    "digits": {"options": "0 (not relevant), 1 (excluded), or 2 (eligible)", "answers": {0: "0", 1: "1", 2: "2"}},
    "letters": {"options": "A (not relevant), B (excluded), or C (eligible)", "answers": {0: "A", 1: "B", 2: "C"}},
}

DEFINITIONS = (
    "Definitions: 'eligible' means the patient meets the inclusion criteria and no exclusion "
    "criterion applies; 'excluded' means the patient would be a candidate for this trial's "
    "condition but at least one exclusion criterion applies or an inclusion criterion is not "
    "met; 'not relevant' means the trial is for a different condition or population."
)


def build_messages(pair: dict, fmt: str, verb: str, shots: list[dict]) -> list[dict]:
    v = VERBALIZERS[verb]
    system = (
        "You judge whether a patient qualifies for a clinical trial from the patient's note "
        f"and the trial's criteria. {DEFINITIONS} Answer with exactly one of: {v['options']}. "
        "Output only the answer."
    )
    msgs = [{"role": "system", "content": system}]
    for s in shots:
        msgs.append({"role": "user", "content": f"Patient:\n{s['patient_text']}\n\nTrial:\n{FORMATS[fmt](s)}\n\nAnswer:"})
        msgs.append({"role": "assistant", "content": v["answers"][s["label"]]})
    msgs.append({"role": "user", "content": f"Patient:\n{pair['patient_text']}\n\nTrial:\n{FORMATS[fmt](pair)}\n\nAnswer:"})
    return msgs


def parse_label(text: str, verb: str) -> int | None:
    """The label the model's first words name, or None if unreadable."""
    t = text.strip().strip('"\'.').lower()
    answers = VERBALIZERS[verb]["answers"]
    for label, word in sorted(answers.items(), key=lambda kv: -len(kv[1])):
        if t.startswith(word.lower()):
            return label
    # Any verbalizer's words are accepted as a fallback, longest first, so
    # "not relevant" is not read as a miss for "relevant".
    for label, word in sorted(LABELS.items(), key=lambda kv: -len(kv[1])):
        if t.startswith(word):
            return label
    return None


def metrics(golds: list[int], preds: list[int | None]) -> dict:
    n = len(golds)
    acc = sum(g == p for g, p in zip(golds, preds)) / n if n else 0.0
    f1s = []
    for c in (0, 1, 2):
        tp = sum(1 for g, p in zip(golds, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(golds, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(golds, preds) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    non_eligible = [(g, p) for g, p in zip(golds, preds) if g != 2]
    false_eligible = sum(1 for g, p in non_eligible if p == 2) / len(non_eligible) if non_eligible else 0.0
    confusion = {f"gold{g}->pred{p}": sum(1 for gg, pp in zip(golds, preds) if gg == g and pp == p) for g in (0, 1, 2) for p in (0, 1, 2, None)}
    return {"n": n, "accuracy": round(acc, 4), "macro_f1": round(sum(f1s) / 3, 4), "f1_per_class": [round(f, 4) for f in f1s],
            "false_eligible_rate": round(false_eligible, 4), "unparsed": sum(p is None for p in preds),
            "confusion": {k: v for k, v in confusion.items() if v}}


def load_pairs(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if "label" in r]  # the first line is the file's provenance header


def pick_shots(train: list[dict], k: int, seed: int) -> list[dict]:
    """k examples from the train split, one label at a time in turn, fixed order."""
    if k == 0:
        return []
    import random
    rng = random.Random(seed)
    by_label = {c: [p for p in train if p["label"] == c] for c in (0, 1, 2)}
    for c in by_label:
        rng.shuffle(by_label[c])
    shots = []
    i = 0
    while len(shots) < k:
        c = (2, 1, 0)[i % 3]
        if by_label[c]:
            shots.append(by_label[c].pop())
        i += 1
    return shots


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    ap.add_argument("--data", default="data/hw12")
    ap.add_argument("--model", default="Qwen/Qwen3-32B-FP8")
    ap.add_argument("--precision", default="fp8", choices=sorted(PRECISIONS))
    ap.add_argument("--decoding", default="greedy", choices=sorted(DECODING_PRESETS))
    ap.add_argument("--formats", nargs="+", default=["full", "criteria", "split"], choices=sorted(FORMATS))
    ap.add_argument("--verbalizers", nargs="+", default=["words"], choices=sorted(VERBALIZERS))
    ap.add_argument("--shots", nargs="+", type=int, default=[0])
    ap.add_argument("--limit", type=int, default=None, help="score only the first N pairs (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--out", default="data/hw12/search")
    args = ap.parse_args()

    pairs = load_pairs(Path(args.data) / f"trec_pairs_{args.split}.jsonl")
    train = load_pairs(Path(args.data) / "trec_pairs_train.jsonl")
    if args.limit:
        pairs = pairs[: args.limit]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    decoding = set_decoding(args.decoding)
    print(f"Loading {args.model} ({args.precision})...", flush=True)
    model, tokenizer = load_model(args.model, precision=args.precision)

    summary = []
    for fmt in args.formats:
        for verb in args.verbalizers:
            for k in args.shots:
                name = f"{args.split}_{fmt}_{verb}_{k}shot"
                shots = pick_shots(train, k, seed=11967)
                golds, preds, rows = [], [], []
                t0 = time.time()
                for i, pair in enumerate(pairs, 1):
                    resp = generate(model, tokenizer, build_messages(pair, fmt, verb, shots), tools=None,
                                    max_new_tokens=args.max_new_tokens, enable_thinking=False)
                    pred = parse_label(resp.text, verb)
                    golds.append(pair["label"]); preds.append(pred)
                    rows.append({"topic_id": pair["topic_id"], "nct_id": pair["nct_id"], "gold": pair["label"],
                                 "pred": pred, "raw": resp.text[:80]})
                    if i % 20 == 0 or i == len(pairs):
                        print(f"  {name}: {i}/{len(pairs)} acc so far {sum(g == p for g, p in zip(golds, preds)) / i:.3f}", flush=True)
                m = metrics(golds, preds)
                m.update({"config": name, "format": fmt, "verbalizer": verb, "shots": k, "seconds": round(time.time() - t0, 1),
                          "model": args.model, "precision": PRECISIONS[args.precision], "decoding": decoding,
                          "commit": commit, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                (out_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
                (out_dir / f"{name}.metrics.json").write_text(json.dumps(m, indent=1) + "\n")
                summary.append(m)
                print(f"{name}: acc {m['accuracy']} | macro-F1 {m['macro_f1']} | false-eligible {m['false_eligible_rate']} "
                      f"| unparsed {m['unparsed']} | {m['seconds']}s", flush=True)

    summary.sort(key=lambda m: (-m["macro_f1"], m["false_eligible_rate"]))
    (out_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\nranked by macro-F1, then false-eligible rate:")
    for m in summary:
        print(f"  {m['config']:32} acc {m['accuracy']:.3f}  F1 {m['macro_f1']:.3f}  false-eligible {m['false_eligible_rate']:.3f}")


if __name__ == "__main__":
    main()
