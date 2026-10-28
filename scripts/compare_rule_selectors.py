"""Which rules should a question get, and which selector picks them?

No model and no GPU: the Ensembl blocks for all sixteen validator questions
are already in a recorded trace, so a rule selector can be scored on its own.

Three selectors are compared against a hand label of the rules each question
needs to be answered correctly:

  trigger  the reverted step 1c (commit c12a7b7): five fixed phrases searched
           in the Ensembl text, up to three rules attached.
  clause   each rule's own "Applies when" sentence, read as a test over the
           fields the Ensembl block prints.
  scoped   the clause test, plus a question check: a rule about the variant's
           functional consequence is attached only where the answer depends
           on that consequence, not to a count or a trial-identifier question.

Usage: uv run python scripts/compare_rule_selectors.py [--trace PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RULES = ["reading-frame", "splicing", "exon-skipping", "nonsense-mediated-decay",
         "read-through", "mechanism-direction"]

# Hand label, written 2026-10-28 by reading each question beside its Ensembl
# block: the rules whose content a correct answer rests on. A rule that is
# background for the variant but not needed for the answer is not labelled,
# because attaching it has a measured cost (32B run 5).
LABELS = {
    "AITX-00001": (["reading-frame", "splicing", "exon-skipping"],
                   "Partly deleted exon 63, so the whole-exon count 1744 is the one that matters; it is not a "
                   "multiple of three, and exon 51 restores the frame, which is what makes eteplirsen the answer."),
    "AITX-00002": (["read-through"],
                   "Two bases replaced by two, stop_gained with no length change, so the stop comes from a "
                   "substitution and ataluren's mechanism applies."),
    "AITX-00003": ([], "A missense variant is neither a skippable deletion nor a premature stop; the consequence "
                       "alone rules out every option."),
    "AITX-00004": ([], "The answer is a therapy named in GeneReviews for pyridoxine-responsive missense variants."),
    "AITX-00005": ([], "GeneReviews restricts pyridoxine to missense variants; this is a frameshift, so the "
                       "non-match settles it."),
    "AITX-00006": ([], "The answer is the two approved RNAi drugs, from a chapter and their labels."),
    "AITX-00007": ([], "An approval age from a GeneReviews chapter."),
    "AITX-00008": ([], "Two medications from a GeneReviews chapter."),
    "AITX-00009": ([], "A trial identifier and its recruitment status."),
    "AITX-00010": ([], "Eligibility against one trial's criteria."),
    "AITX-00011": ([], "Trials for this gene and condition."),
    "AITX-00012": (["mechanism-direction"],
                   "A premature stop is a loss of function, so of memantine, L-serine and radiprodil the two "
                   "antagonists act in the wrong direction."),
    "AITX-00013": ([], "A count of codons printed in the exon line."),
    "AITX-00014": ([], "A percentage printed in the exon line."),
    "AITX-00015": (["nonsense-mediated-decay"],
                   "The frameshift lies in the last exon, 37 of 37, which is exactly the decay rule's escape case."),
    "AITX-00016": ([], "A protein domain from UniProt and InterPro."),
}

# Step 1c as it was: a phrase searched in the Ensembl text, up to three rules.
TRIGGERS = [("Splice sites removed", ["splicing", "exon-skipping"]),
            ("frameshift_variant", ["reading-frame", "nonsense-mediated-decay"]),
            ("stop_gained", ["nonsense-mediated-decay", "read-through"]),
            ("inframe_deletion", ["reading-frame"]),
            ("start_lost", ["reading-frame"])]
TRIGGER_CAP = 3


def trigger_select(block: str, question: dict) -> list[str]:
    out: list[str] = []
    for phrase, rules in TRIGGERS:
        if phrase in block:
            out += [r for r in rules if r not in out]
    return out[:TRIGGER_CAP]


def fields(block: str) -> dict:
    """The values the Ensembl block prints, as data for the clause tests."""
    cons = re.search(r"^Consequence[^:]*:\s*(.+)$", block, re.M)
    whole = re.search(r"If exons [\d\-]+ were lost whole: (\d+) coding bases, ([^\n]*)", block)
    partial = re.search(r"Exon \d+: (\d+) of (\d+) coding bases deleted", block)
    exon = re.search(r"^Exon(?:s)?: ([\d\-]+)/(\d+)", block, re.M)
    neighbour = re.search(r"together with one neighbouring exon: ([^\n]*)", block)
    frame = re.search(r"^Frame[^:]*:\s*(.+)$", block, re.M)
    return {
        "consequences": [c.strip() for c in (cons.group(1) if cons else "").split(",") if c.strip()],
        "frame_line": frame.group(1) if frame else "",
        "counts_bases": bool(re.search(r"\d+ (?:coding )?bases?,? (?:a multiple|one more|[a-z ]*)?(?:deleted|inserted)", frame.group(1))) if frame else False,
        "whole_exon_count": int(whole.group(1)) if whole else None,
        "whole_exon_remark": whole.group(2) if whole else "",
        "partial_exon": bool(partial),
        "exon_index": exon.group(1) if exon else "",
        "exon_total": int(exon.group(2)) if exon else None,
        "neighbour_line": neighbour.group(1) if neighbour else "",
    }


def clause_select(block: str, question: dict) -> list[str]:
    """Each rule's own 'Applies when' sentence, as a test over those fields."""
    f = fields(block)
    cons = f["consequences"]
    out = []
    # "the Frame line gives a count of coding bases deleted or inserted"
    if f["counts_bases"]:
        out.append("reading-frame")
    # "a deletion covers most but not all of an exon's coding bases and VEP
    #  reports splice acceptor or donor sites among the consequences"
    if f["partial_exon"] and any(c.startswith(("splice_acceptor", "splice_donor")) for c in cons):
        out.append("splicing")
    # "a whole-exon loss that is not a multiple of three and the neighbouring-exon
    #  line names an exon whose removal gives a multiple of three"
    not_multiple = bool(re.search(r"(?:one|two) (?:more|fewer) than a multiple of three|not a multiple of three",
                                  f["whole_exon_remark"]))
    if f["whole_exon_count"] is not None and not_multiple and "a multiple of three" in f["neighbour_line"]:
        out.append("exon-skipping")
    # "the consequence is stop_gained or frameshift_variant"
    if any(c in ("stop_gained", "frameshift_variant") for c in cons):
        out.append("nonsense-mediated-decay")
    # "the consequence is stop_gained with no change in length"
    if "stop_gained" in cons and "no change in length" in f["frame_line"]:
        out.append("read-through")
    # "the answer is a choice between agents acting on the variant's protein"
    if question["answer_format"] == "multiple_choice" and drug_options(question["prompt"]):
        out.append("mechanism-direction")
    return out


def drug_options(prompt: str) -> bool:
    """True when the multiple-choice options are drug names (RxNorm decides)."""
    from src.drug_names import is_drug_name
    tail = prompt.split(":")[-1]
    tail = re.sub(r"\bor\b", ",", tail)
    cands = [o.strip(" .?\n") for o in tail.split(",")]
    cands = [c for c in cands if 4 <= len(c) <= 40 and c.lower() not in ("none", "n/a")]
    return any(is_drug_name(c) is True for c in cands[:6])


# A rule about the variant's functional consequence is attached only where the
# answer depends on that consequence. A codon count or a trial identifier does
# not, and run 5 attached rules to five such questions.
# A count printed in the exon line and a trial identifier do not depend on the
# variant's mechanism; run 5 attached rules to five such questions.
NON_CONSEQUENCE_CATEGORIES = {"Clinical_Trials"}


def scoped_select(block: str, question: dict) -> list[str]:
    if question["category"] in NON_CONSEQUENCE_CATEGORIES:
        return []
    if question["answer_format"] == "numeric_match":
        return []
    return clause_select(block, question)


SELECTORS = {"trigger": trigger_select, "clause": clause_select, "scoped": scoped_select}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", default="data/eval_trace_32b_run6.json")
    ap.add_argument("--questions", default="data/phase1_questions.json")
    ap.add_argument("--out", default=None, help="Write the markdown table here as well as to stdout")
    args = ap.parse_args()

    questions = {q["id"]: q for q in json.load(open(args.questions))}
    traces = json.load(open(args.trace))["traces"]
    blocks = {}
    for r in traces:
        vep = [c for c in r["tool_calls"] if c["tool"] == "query_ensembl"]
        blocks[r["id"]] = vep[0].get("summary", "") if vep else ""

    rows, totals = [], {name: {"hit": 0, "miss": 0, "extra": 0} for name in SELECTORS}
    per_rule = {name: {r: {"hit": 0, "miss": 0, "extra": 0} for r in RULES} for name in SELECTORS}
    for qid in sorted(blocks):
        q = dict(questions[qid]["question"])
        want = set(LABELS[qid][0])
        row = {"id": qid, "want": sorted(want)}
        for name, fn in SELECTORS.items():
            got = set(fn(blocks[qid], q))
            row[name] = sorted(got)
            for r in want & got:
                totals[name]["hit"] += 1; per_rule[name][r]["hit"] += 1
            for r in want - got:
                totals[name]["miss"] += 1; per_rule[name][r]["miss"] += 1
            for r in got - want:
                totals[name]["extra"] += 1; per_rule[name][r]["extra"] += 1
        rows.append(row)

    out = ["# Rule selector comparison", "",
           f"Blocks from `{args.trace}`; hand label in `scripts/compare_rule_selectors.py`.",
           "No model was run. Hit: a needed rule attached. Miss: a needed rule withheld.",
           "Extra: a rule attached that the answer does not rest on.", "",
           "| Question | Needed | trigger | clause | scoped |", "|---|---|---|---|---|"]
    for row in rows:
        cells = [", ".join(row[k]) or "-" for k in ("want", "trigger", "clause", "scoped")]
        out.append(f"| {row['id']} | " + " | ".join(cells) + " |")
    out += ["", "| Selector | Needed rules attached | Withheld | Extra |", "|---|---|---|---|"]
    for name in SELECTORS:
        t = totals[name]
        out.append(f"| {name} | {t['hit']} of {t['hit'] + t['miss']} | {t['miss']} | {t['extra']} |")
    out += ["", "Per rule, needed and attached:", "", "| Rule | trigger | clause | scoped |", "|---|---|---|---|"]
    for r in RULES:
        cells = []
        for name in SELECTORS:
            p = per_rule[name][r]
            cells.append(f"{p['hit']}/{p['hit'] + p['miss']} (+{p['extra']})")
        out.append(f"| {r} | " + " | ".join(cells) + " |")
    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
