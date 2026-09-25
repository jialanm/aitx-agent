"""Answer normalization for exact-match scoring."""

from __future__ import annotations

import json
import re


def normalize_answer(
    raw_answer: str,
    answer_format: str,
    prompt: str = "",
    options: list[str] | None = None,
) -> str:
    """Normalize model output to match expected answer format.

    Args:
        raw_answer: Raw text from the model.
        answer_format: One of binary, multiple_choice, numeric_match, string_match.
        prompt: The original question prompt (used to map lettered answers).
        options: Verified answer options for multiple_choice; empty means the
            answer is left as written.

    Returns:
        Normalized answer string.
    """
    # Strip any <think> blocks that leaked through (safety net)
    cleaned = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL)
    cleaned = re.sub(r"<think>(?:(?!</think>).)*$", "", cleaned, flags=re.DOTALL)

    # Strip whitespace and common wrapping
    cleaned = cleaned.strip().strip('"').strip("'").strip(".")
    cleaned = cleaned.strip()

    if answer_format == "binary":
        return _normalize_binary(cleaned)
    elif answer_format == "multiple_choice":
        return _normalize_multiple_choice(cleaned, prompt, options or [])
    elif answer_format == "numeric_match":
        return _normalize_numeric(cleaned)
    elif answer_format == "string_match":
        return _normalize_string(cleaned)
    else:
        return cleaned


def _normalize_binary(text: str) -> str:
    """Extract Yes/No from model output."""
    lower = text.lower().strip()

    # Direct match
    if lower in ("yes", "no"):
        return lower.capitalize()

    # Look for yes/no at the start of the response
    if lower.startswith("yes"):
        return "Yes"
    if lower.startswith("no"):
        return "No"

    # Search for yes/no as whole words
    yes_match = re.search(r"\byes\b", lower)
    no_match = re.search(r"\bno\b", lower)

    if yes_match and not no_match:
        return "Yes"
    if no_match and not yes_match:
        return "No"

    # If both found, take the first one
    if yes_match and no_match:
        return "Yes" if yes_match.start() < no_match.start() else "No"

    # Last resort: look at the last line
    lines = text.strip().split("\n")
    last_line = lines[-1].lower().strip()
    if "yes" in last_line:
        return "Yes"
    if "no" in last_line:
        return "No"

    return text.strip()


_LETTERED_OPTIONS = re.compile(r"\(([A-Z])\)\s*(.+?)(?=\s*\([A-Z]\)|$)")


def parse_option_json(text: str) -> list[str] | None:
    """Read a JSON array of strings out of model output, or None if absent.

    Tolerates a surrounding code fence or a sentence before the array, since
    small models add those even when told not to. Anything that is not a
    flat list of non-empty strings is treated as unparseable.
    """
    m = re.search(r"\[.*?\]", text, flags=re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not all(isinstance(x, str) and x.strip() for x in data):
        return None
    return [x.strip() for x in data]


def _find_in_prompt(option: str, prompt: str) -> re.Match | None:
    """Locate the option in the prompt, or None if it is not there.

    Match is case-insensitive and bounded so "GRD" does not match "upgrade".
    Hyphens count as word characters so "L-serine" is one token.
    """
    return re.search(r"(?<![\w-])" + re.escape(option) + r"(?![\w-])", prompt, flags=re.I)


# Text allowed between two options in a list: optional comma, optional or/and.
_LIST_GAP = re.compile(r"^\s*(?:,\s*)?(?:(?:or|and)\s+)?$", re.I)
# A short fragment after the last option that looks like one more list item.
_TAIL_ITEM = re.compile(r"^\s*(?:,\s*(?:or|and)?|or|and)\s+([^,.?;:]+?)\s*(?:[,.?;:]|$)", re.I)
# Text before the first option that implies an earlier list item was skipped.
_HEAD_SEP = re.compile(r"(?:,|\bor|\band)\s*$", re.I)


def _completeness_reason(matches: list[re.Match], prompt: str) -> str | None:
    """Use the prompt's list structure to detect a dropped option.

    Options in a question are written as one contiguous list. Every gap
    between consecutive verified options must be only a separator; a
    leftover fragment is a dropped option. The text just after the last
    option and just before the first are checked the same way.
    """
    ordered = sorted(matches, key=lambda m: m.start())
    for a, b in zip(ordered, ordered[1:]):
        gap = prompt[a.end():b.start()]
        if not _LIST_GAP.match(gap):
            return f"possible dropped option between {a.group(0)!r} and {b.group(0)!r}: {gap.strip(' ,')!r}"

    tail = _TAIL_ITEM.match(prompt[ordered[-1].end():])
    if tail and len(tail.group(1).split()) <= 4:
        return f"possible dropped option after {ordered[-1].group(0)!r}: {tail.group(1)!r}"

    if _HEAD_SEP.search(prompt[: ordered[0].start()]):
        return f"possible dropped option before {ordered[0].group(0)!r}"

    return None


def verify_options(candidates: list[str], prompt: str) -> tuple[list[str], str | None]:
    """Accept a model-proposed option list only if the prompt backs every item.

    Rules: each item appears verbatim in the prompt (case-insensitive, word
    bounded); at least two items; no duplicates; no item contained in another;
    and the prompt's list structure shows no item the model left out.
    Returns (options as spelled in the prompt, None) on success or
    ([], reason) on rejection. The caller logs the reason.
    """
    if len(candidates) < 2:
        return [], f"fewer than two options: {candidates}"

    matches = []
    for c in candidates:
        m = _find_in_prompt(c, prompt)
        if m is None:
            return [], f"option not in prompt: {c!r}"
        matches.append(m)
    found = [m.group(0) for m in matches]

    lowered = [f.lower() for f in found]
    if len(set(lowered)) != len(lowered):
        return [], f"duplicate options: {found}"
    for i, a in enumerate(lowered):
        for j, b in enumerate(lowered):
            if i != j and a in b:
                return [], f"option {found[i]!r} is contained in {found[j]!r}"

    reason = _completeness_reason(matches, prompt)
    if reason:
        return [], reason

    return found, None


def _normalize_multiple_choice(text: str, prompt: str, options: list[str]) -> str:
    """Map model output onto one of the verified options, in the prompt's spelling.

    Falls through to the raw text when nothing matches: an unmatched answer
    should score as wrong and be visible in the eval output, not be repaired.
    """
    cleaned = text.strip()
    if not options:
        return cleaned

    lower = cleaned.lower()

    for opt in options:
        if lower == opt.lower():
            return opt

    # A bare letter answer only means something when the prompt lettered its
    # options; map it through the "(A) text" pattern in the prompt.
    m = re.match(r"^\(?([a-z])\)?[.):]?(?:\s|$)", lower)
    if m:
        for letter, opt_text in _LETTERED_OPTIONS.findall(prompt):
            if letter == m.group(1).upper() and opt_text.strip() in options:
                return opt_text.strip()

    # Option mentioned inside a longer answer: take the earliest mention,
    # preferring the longer option when two start at the same place.
    hits = []
    for opt in options:
        m = re.search(r"(?<![\w-])" + re.escape(opt.lower()) + r"(?![\w-])", lower)
        if m:
            hits.append((m.start(), -len(opt), opt))
    if hits:
        return min(hits)[2]

    return cleaned


def _normalize_numeric(text: str) -> str:
    """Extract numeric value from model output."""
    # Look for numbers (integer or float)
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        num = numbers[0]
        # Return integer format if whole number
        try:
            f = float(num)
            if f == int(f):
                return str(int(f))
            return num
        except ValueError:
            return num

    return text.strip()


def _normalize_string(text: str) -> str:
    """Normalize string match answers."""
    cleaned = text.strip().strip('"').strip("'").rstrip(".")

    # Handle GOF/LOF normalization
    lower = cleaned.lower()
    if lower in ("gof", "gain-of-function", "gain of function"):
        return "GOF"
    if lower in ("lof", "loss-of-function", "loss of function"):
        return "LOF"

    # Remove common preambles
    preamble_patterns = [
        r"^(?:the\s+)?(?:answer\s+is|response\s+is|it\s+is)\s*[:\s]*",
        r"^(?:based\s+on\s+.*?,?\s*)",
    ]
    for pat in preamble_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    # Remove trailing periods and quotes
    cleaned = cleaned.strip().strip('"').strip("'").rstrip(".")

    return cleaned
