"""Does a claim about this patient's variant match what VEP returned?

The model describes the variant in its own words, and those words can
contradict the retrieved consequence. In both 32B runs of AITX-00002 it wrote
"The patient's variant is a frameshifting insertion/deletion" where VEP
reported stop_gained with no change in length: two bases replaced by two, so
no frame shift. That run answered correctly, which is the point of checking
it here rather than only on the eval set. On a report the derivation is the
product, and an unsound description reaches the reader even when the
conclusion happens to hold.

What is not a contradiction, and why:
  - a frameshift does create a premature stop downstream, so "premature stop"
    beside VEP's frameshift_variant is correct;
  - "nonsense-mediated decay" names the mechanism, not the variant's class;
  - a negated sentence ("not a nonsense mutation", "inframe, meaning no
    frameshift") states the opposite of the word it contains;
  - a hypothetical ("if the deletion leads to a frameshift") asserts nothing;
  - a sentence that does not refer to this variant is a general statement
    about a rule or a drug, not a claim about the patient.
"""

from __future__ import annotations

import re

# A word the model may use for the variant, and the VEP term it claims.
WORDS = [
    (r"\bframe[- ]?shift(?:ing|ed)?\b", "frameshift_variant"),
    (r"\bnonsense\b(?!-mediated|\s+mediated)", "stop_gained"),
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


def vep_consequences(blocks: list[str]) -> list[str]:
    """VEP's consequence terms, read from the step-1 block's own lines."""
    terms: list[str] = []
    for text in blocks:
        for line in re.findall(r"^Consequence[^:]*:\s*(.+)$", text or "", re.M):
            terms += [t.strip() for t in line.split(",") if t.strip()]
    return list(dict.fromkeys(terms))


def contradiction(text: str, terms: list[str]) -> tuple[str, str] | None:
    """(the word used, the sentence) for a claim about this variant that VEP's
    terms contradict; None when every such claim matches a reported term."""
    if not terms:
        return None
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


def notice(word: str, terms: list[str]) -> str:
    """What the harness tells the model, in values rather than conclusions."""
    return (
        f'You described this variant as "{word}". Ensembl VEP reported these consequences for the patient\'s '
        f"transcript: {', '.join(terms)}. The term you used is not among them. Settle the description against "
        "the values in the variant block before using any rule that turns on it, and say which value you rely on."
    )
