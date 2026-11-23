"""Build balanced (patient, trial) pairs from TREC Clinical Trials 2021-2022 for Homework 12.

Source: Hugging Face dataset tirandazdylan/trec-ct-matching (configs topics,
qrels, trials), a ClinicalTrials.gov snapshot of 2021-04-27 with physician
relevance judgments. Trial records are US Government data; ClinicalTrials.gov
asks that the source and the processing date be stated wherever they are
used, and each output file carries both in its first line.

Patients stay disjoint across splits: the dataset's own split column assigns
each of the 125 cases to train (82), validation (13) or test (30). Within a
split, pairs are drawn per label with a fixed seed and a cap per patient, so
the three labels are balanced and no single case dominates a label.

Labels: 0 not relevant, 1 excluded (meets inclusion but an exclusion
applies), 2 eligible.

Usage:
    uv run --group hw12 python hw12/build_trec_pairs.py [--out data/hw12]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from datasets import load_dataset

DATASET = "tirandazdylan/trec-ct-matching"
SNAPSHOT_DATE = "2021-04-27"
SEED = 11967
LABEL_WORDS = {0: "not relevant", 1: "excluded", 2: "eligible"}
# Pairs per label per split, and the most pairs one patient may supply per label.
SIZES = {"train": 300, "validation": 60, "test": 100}
PER_PATIENT_CAP = {"train": 15, "validation": 15, "test": 10}
OUT_NAMES = {"train": "train", "validation": "dev", "test": "test"}


def trial_text(t: dict) -> str:
    """The trial as the judge will read it: identity, population, and the full criteria."""
    lines = [
        f"Title: {t.get('brief_title') or t.get('official_title') or ''}",
        f"Conditions: {', '.join(t.get('conditions') or []) or 'not stated'}",
        f"Interventions: {', '.join(t.get('interventions') or []) or 'not stated'}",
        f"Sex: {t.get('gender') or 'not stated'} | Minimum age: {t.get('minimum_age') or 'not stated'} | "
        f"Maximum age: {t.get('maximum_age') or 'not stated'} | Healthy volunteers: {t.get('healthy_volunteers') or 'not stated'}",
        "Eligibility criteria:",
        (t.get("eligibility_criteria") or "").strip() or "not stated",
    ]
    return "\n".join(lines)


def sample_split(qrels: list[dict], split_topics: set[str], n_per_label: int, cap: int, rng: random.Random) -> list[dict]:
    """Balanced pairs for one split, capped per patient per label."""
    by_label: dict[int, list[dict]] = defaultdict(list)
    for q in qrels:
        if q["topic_id"] in split_topics:
            by_label[int(q["judgment"])].append(q)
    chosen: list[dict] = []
    for label in (0, 1, 2):
        pool = by_label[label][:]
        rng.shuffle(pool)
        per_patient: Counter = Counter()
        picked = []
        for q in pool:
            if per_patient[q["topic_id"]] >= cap:
                continue
            picked.append(q)
            per_patient[q["topic_id"]] += 1
            if len(picked) == n_per_label:
                break
        if len(picked) < n_per_label:
            raise SystemExit(f"only {len(picked)} pairs available for label {label} under the cap; lower the cap or the size")
        chosen.extend(picked)
    rng.shuffle(chosen)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/hw12", help="output directory")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    topics = {t["topic_id"]: t for t in load_dataset(DATASET, "topics", split="train")}
    qrels = list(load_dataset(DATASET, "qrels", split="train"))
    trials = {t["nct_id"]: t for t in load_dataset(DATASET, "trials", split="train")}
    print(f"loaded {len(topics)} patients, {len(qrels)} judgments, {len(trials)} trials")

    rng = random.Random(SEED)
    summary = {}
    for split, n in SIZES.items():
        split_topics = {tid for tid, t in topics.items() if t["split"] == split}
        pairs = sample_split(qrels, split_topics, n, PER_PATIENT_CAP[split], rng)
        path = out_dir / f"trec_pairs_{OUT_NAMES[split]}.jsonl"
        header = {
            "_source": DATASET, "_clinicaltrials_snapshot": SNAPSHOT_DATE, "_built": date.today().isoformat(),
            "_seed": SEED, "_split": split, "_patients": len(split_topics), "_pairs": len(pairs),
            "_per_label": n, "_per_patient_cap": PER_PATIENT_CAP[split],
        }
        missing = 0
        with path.open("w") as f:
            f.write(json.dumps(header) + "\n")
            for q in pairs:
                t = trials.get(q["nct_id"])
                if t is None:
                    missing += 1
                    continue
                f.write(json.dumps({
                    "topic_id": q["topic_id"],
                    "patient_text": topics[q["topic_id"]]["text"].strip(),
                    "nct_id": q["nct_id"],
                    "trial_text": trial_text(t),
                    "label": int(q["judgment"]),
                    "label_word": LABEL_WORDS[int(q["judgment"])],
                }) + "\n")
        labels = Counter(int(q["judgment"]) for q in pairs)
        patients = Counter(q["topic_id"] for q in pairs)
        summary[OUT_NAMES[split]] = {"pairs": len(pairs) - missing, "labels": dict(labels), "patients_used": len(patients),
                                     "max_pairs_from_one_patient": max(patients.values()), "trials_missing": missing}
        print(f"{OUT_NAMES[split]:5} {len(pairs) - missing} pairs | labels {dict(sorted(labels.items()))} | "
              f"{len(patients)} of {len(split_topics)} patients | max per patient {max(patients.values())} | missing trials {missing} -> {path}")
    (out_dir / "trec_pairs_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
