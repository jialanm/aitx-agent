"""Claims whose truth is a database state, and whether we hold that database.

The model's knowledge is wanted: mechanism, direction of effect, the frame and
decay rules, drug classes, and the inference steps that link them are stable
and it brings them well. What its memory cannot carry is anything that changed
after its cutoff or differs by country, and both 32B regressions of 2026-10-28
were exactly that: "that means it's not FDA-approved in the US for DMD" on
AITX-00002, and "nedosiran was approved in 2023" on AITX-00006, where the
label had in fact moved from age 9 to age 2 before the question's own date.

So this module does not judge reasoning. It asks one question per live claim:
is there a source about this entity, of the kind that settles this category,
among the things retrieved for this patient? A claim with no such source is
reported so the harness can ask for the lookup once, and is never deleted on
the strength of its absence.

The subject of a source is taken from the recorded call's arguments, which the
harness or the model wrote as structured values, not from the summary's prose;
keying on wording is what made the rule triggers miss AITX-00001.
"""

from __future__ import annotations

import re

# category -> trigger phrases, and per entity kind the tools that settle it.
CATEGORIES: list[tuple[str, list[str], dict[str, list[str]]]] = [
    ("status", [r"\bapprov(?:ed|al)\b", r"\bauthoris?ed\b", r"\bwithdrawn\b",
                r"\binvestigational\b", r"\bon the market\b", r"\bmarketed\b"],
     # Label first, then PubMed: a drug approved only abroad has no United
     # States label, which openFDA's empty reply says in as many words.
     {"drug": ["search_fda_labels", "search_pubmed", "search_genereviews", "search_pharmgkb"]}),
    ("applicability", [r"\bindicated for\b", r"\beligible\b", r"\bamenable\b", r"\bqualifies\b",
                       r"\bapplies to (?:this|the) patient\b", r"\bcontraindicated\b"],
     {"drug": ["search_fda_labels", "search_genereviews"],
      "trial": ["search_clinical_trials"]}),
    ("trial_state", [r"\brecruiting\b", r"\bterminated\b", r"\bwithdrawn\b", r"\bcompleted\b",
                     r"\bphase\s*(?:1|2|3|4|i{1,3}|iv)\b"],
     {"trial": ["search_clinical_trials"]}),
    ("classification", [r"\bpathogenic\b", r"\buncertain significance\b", r"\bVUS\b", r"\bbenign\b"],
     {"variant": ["search_clinvar"]}),
]

# The argument a call carries for each kind of subject.
SUBJECT_ARGS = {
    "drug": ("drug",),
    "trial": ("nct_id",),
    "variant": ("variant_cdna", "genomic_hgvs", "variant"),
    "gene": ("gene",),
}
TRIAL_ID = re.compile(r"\bNCT\d{8}\b")
# A name the model introduces that no option, answer or lookup mentions: a
# capitalised word, a parenthesised one, or the word after "drug called".
# AITX-00006's thinking offered "a drug called pegalysine (Oxepduo) approved
# in the US", which the candidate list could not see.
# The first pattern is the model calling something a drug in as many words,
# which stands on its own; the others are shapes that need RxNorm to confirm.
NAMED_AS_DRUG = re.compile(r"\b(?:drugs?|agents?|therap(?:y|ies))\s+(?:called|named)\s+([A-Za-z][A-Za-z-]{3,})", re.I)
SHAPED_LIKE_A_NAME = [
    re.compile(r"\(([A-Z][A-Za-z-]{3,})\)"),
    re.compile(r"\b([A-Z][a-z]{3,}(?:mab|nib|sen|tide|ast|vir|pril|statin|zumab))\b"),
]
# Words that look like names but are not drugs; RxNorm would reject them and
# the call is not worth making.
NOT_A_NAME = re.compile(r"^(?:The|This|That|These|Those|There|When|While|With|What|Which|However|Because|Since|"
                        r"Therefore|Patient|Variant|Exon|Gene|Protein|Status|Phase|FDA|EMA|PubMed|GeneReviews|"
                        r"ClinVar|Ensembl|Indications|Approved|Trial|Study|United|States|Europe)$", re.I)
# An adapter's own "nothing found" reply is not a source about its subject.
NOTHING_FOUND = re.compile(r"^\s*(?:no |none |0 )|not found|returned no|no .{0,40}found", re.I)


def sources_held(tool_calls: list[dict]) -> dict[tuple[str, str], set[str]]:
    """(kind, entity) -> the tools that returned something about it.

    Trial identifiers are also read out of the summary, because a trial search
    by condition returns records the call's arguments do not name, and an
    identifier is a token with a fixed shape rather than a turn of phrase.
    """
    held: dict[tuple[str, str], set[str]] = {}
    for call in tool_calls:
        summary = call.get("summary") or ""
        if not summary or NOTHING_FOUND.match(summary.strip()):
            continue
        tool = call.get("tool", "")
        args = call.get("arguments") or {}
        for kind, names in SUBJECT_ARGS.items():
            for arg in names:
                value = args.get(arg)
                if isinstance(value, str) and value.strip():
                    held.setdefault((kind, value.strip().lower()), set()).add(tool)
        for nct in TRIAL_ID.findall(summary):
            held.setdefault(("trial", nct.lower()), set()).add(tool)
    return held


def attempted(tool_calls: list[dict]) -> dict[tuple[str, str], set[str]]:
    """(kind, entity) -> every tool already called about it, result or not, so
    a notice names a lookup that has not been tried (AITX-00002: the label
    search for ataluren had already returned nothing, and PubMed was the
    lookup that settled it)."""
    tried: dict[tuple[str, str], set[str]] = {}
    for call in tool_calls:
        args = call.get("arguments") or {}
        for kind, names in SUBJECT_ARGS.items():
            for arg in names:
                value = args.get(arg)
                if isinstance(value, str) and value.strip():
                    tried.setdefault((kind, value.strip().lower()), set()).add(call.get("tool", ""))
    return tried


def entities_in(sentence: str, drugs: list[str], gene: str = "") -> list[tuple[str, str]]:
    """(kind, name) for the entities a sentence names, from a known candidate
    list rather than free-text extraction: the question's options and the
    answer's own terms for drugs, a trial identifier by its shape."""
    found: list[tuple[str, str]] = []
    low = sentence.lower()
    for drug in drugs:
        if drug and drug.strip() and drug.strip().lower() in low:
            found.append(("drug", drug.strip().lower()))
    for nct in TRIAL_ID.findall(sentence):
        found.append(("trial", nct.lower()))
    if gene and re.search(r"\bthis variant\b|\bthe variant\b", sentence, re.I):
        found.append(("variant", gene.strip().lower()))
    return list(dict.fromkeys(found))


def introduced_names(sentence: str) -> list[tuple[str, bool]]:
    """(name, the model called it a drug) for names this sentence introduces.

    A name the model itself calls a drug needs no confirmation: the claim is
    the model's either way, and if the name does not exist the lookup is the
    thing that shows it. A name that merely has the shape of one is confirmed
    with RxNorm first, so an ordinary capitalised word is never challenged.
    """
    out: list[tuple[str, bool]] = []
    seen: set[str] = set()

    def add(name: str, explicit: bool) -> None:
        name = name.strip()
        if len(name) < 4 or NOT_A_NAME.match(name) or name.lower() in seen:
            return
        seen.add(name.lower())
        out.append((name, explicit))

    for name in NAMED_AS_DRUG.findall(sentence):
        add(name, True)
    for pattern in SHAPED_LIKE_A_NAME:
        for name in pattern.findall(sentence):
            add(name, False)
    return out


def confirmed_drugs(sentence: str, confirm, cache: dict[str, bool | None] | None = None,
                    limit: int = 3) -> list[str]:
    """Of the names a sentence introduces, those RxNorm knows as drugs.

    A name the model itself calls a drug is taken as one without asking. For
    the rest, RxNorm deciding keeps the check from challenging any capitalised
    word, and its unknown answer yields no candidate, since a notice on a
    guess is worse than a claim left unchecked.

    RxNorm's silence is never read as the drug not existing: it has no concept
    for ataluren, which is approved in the European Union, and none for
    pegalysine, which the model asserted was approved in the United States.
    Both cases want the same thing, a lookup.
    """
    found: list[str] = []
    for name, explicit in introduced_names(sentence)[:limit]:
        key = name.lower()
        if explicit:                               # the model called it a drug
            found.append(key)
            continue
        if cache is not None and key in cache:
            verdict = cache[key]
        else:
            try:
                verdict = confirm(name)
            except Exception:                      # a lookup failure is an unknown, not a drug
                verdict = None
            if cache is not None:
                cache[key] = verdict
        if verdict is True:
            found.append(key)
    return found


def unsourced(text: str, drugs: list[str], held: dict[tuple[str, str], set[str]],
              gene: str = "", tried: dict[tuple[str, str], set[str]] | None = None,
              confirm=None, cache: dict[str, bool | None] | None = None
              ) -> list[tuple[str, str, str, str]]:
    """(entity, kind, category, tool to call) for each live claim in the text
    that no retrieved source of the right kind covers. One entry per entity
    and category, in the order the claims appear."""
    out: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    # The entity is often named one sentence before the claim about it
    # ("The FDA label search for Ataluren didn't return any results. That
    # means it's not FDA-approved in the US"), so the last entity named
    # carries forward until another is named.
    carried: list[tuple[str, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text or ""):
        here = entities_in(sentence, drugs, gene)
        hits = [(category, patterns, by_kind) for category, patterns, by_kind in CATEGORIES
                if any(re.search(p, sentence, re.I) for p in patterns)]
        # A live claim about a name the harness has never seen: ask RxNorm
        # whether that name is a drug, and only then treat it as an entity.
        if hits and not here and confirm is not None:
            here = [("drug", name) for name in confirmed_drugs(sentence, confirm, cache)]
        if here:
            carried = here
        for category, patterns, by_kind in hits:
            for kind, name in (here or carried):
                tools = by_kind.get(kind)
                if not tools or (name, category) in seen:
                    continue
                if held.get((kind, name), set()) & set(tools):
                    continue
                seen.add((name, category))
                already = (tried or {}).get((kind, name), set())
                next_tool = next((t for t in tools if t not in already), tools[0])
                out.append((name, kind, category, next_tool))
    return out


def answered(claims: list[tuple[str, str, str, str]],
             held: dict[tuple[str, str], set[str]]) -> list[tuple[str, str, str, str]]:
    """The claims that now have a source of the right kind."""
    by_category = {category: by_kind for category, _patterns, by_kind in CATEGORIES}
    out = []
    for claim in claims:
        name, kind, category, _tool = claim
        tools = by_category.get(category, {}).get(kind, [])
        if held.get((kind, name), set()) & set(tools):
            out.append(claim)
    return out


def all_tried(claims: list[tuple[str, str, str, str]],
              tried: dict[tuple[str, str], set[str]]) -> bool:
    """True when the tool named for each claim has been called, whatever it
    returned. The notice goes once this holds: it says to call a tool, and
    nothing in the conversation marks an instruction done, so AITX-00003 of
    run 6 read it as outstanding and made the same five calls on four turns.
    An empty reply is reported separately, never left for the model to infer
    from a notice still standing."""
    for name, kind, _category, tool in claims:
        if tool not in tried.get((kind, name), set()):
            return False
    return bool(claims)


def empty_note(unanswered: list[tuple[str, str, str, str]],
               tried: dict[tuple[str, str], set[str]]) -> str:
    """One line for the lookups that came back with nothing, naming what is
    left to try. Silence from a source is not evidence about the claim."""
    parts = []
    by_category = {category: by_kind for category, _patterns, by_kind in CATEGORIES}
    for name, kind, category, _tool in unanswered:
        done = tried.get((kind, name), set())
        rest = [t for t in by_category.get(category, {}).get(kind, []) if t not in done]
        left = f"; {rest[0]} has not been tried" if rest else ""
        parts.append(f'"{name}" ({WORDING[category]}): the lookups returned nothing{left}')
    return ("The lookups you made did not settle everything: " + "; ".join(parts) +
            ". A source holding nothing about a claim says nothing for or against it, so report the "
            "claim as not verified against a current source rather than as false.")


WORDING = {
    "status": "whether it is approved, authorised or withdrawn",
    "applicability": "whether it applies to this patient",
    "trial_state": "what state the trial is in",
    "classification": "how the variant is classified",
}


def notice(claims: list[tuple[str, str, str, str]]) -> str:
    """What the harness says. It names the lookup and leaves the conclusion
    open: a fact we cannot retrieve may still be true, and the claim stands
    marked rather than removed."""
    parts = []
    for name, _kind, category, tool in claims:
        parts.append(f'"{name}" ({WORDING[category]}), which {tool} returns')
    listed = "; ".join(parts)
    return (
        "Your answer rests on facts that change over time and that nothing retrieved for this patient "
        f"states: {listed}. These are not things to settle from memory, because approvals, labels, trial "
        "states and classifications change and differ by country. Call the named tool for each, then answer. "
        "If a lookup returns nothing, say the fact is not verified against a current source rather than "
        "treating it as false."
    )
