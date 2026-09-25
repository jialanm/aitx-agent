"""Answer normalization for exact-match scoring."""

from __future__ import annotations

import re


def normalize_answer(raw_answer: str, answer_format: str, prompt: str = "") -> str:
    """Normalize model output to match expected answer format.

    Args:
        raw_answer: Raw text from the model.
        answer_format: One of binary, multiple_choice, numeric_match, string_match.
        prompt: The original question prompt (needed for multiple_choice).

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
        return _normalize_multiple_choice(cleaned, prompt)
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
_LEADIN_LIST = re.compile(
    r"(?:answer\s+choices|choices|options)\s*:\s*(.+?)\s*[.?]?\s*$", re.I | re.S
)
# A trailing comma list whose last item is introduced by "or": "X, Y, or Z".
_OR_LIST = re.compile(r"((?:[^,:?]+,\s*)+(?:or\s+)?[^,?]+?)\s*[?.]?\s*$", re.S)
# When the list is not introduced by a colon, the first item still carries the
# question stem ("...treatment with Memantine"); cut at the last such word.
_STEM_WORDS = re.compile(r"\b(?:with|to|of|between|among|following)\b\s*", re.I)


def _split_list_items(text: str) -> list[str]:
    items = [i.strip() for i in text.split(",")]
    items = [re.sub(r"^(?:or|and)\s+", "", i, flags=re.I).strip(" .?\"'") for i in items]
    return [i for i in items if i]


def extract_options(prompt: str) -> list[str]:
    """Pull the answer options out of a multiple-choice prompt.

    Handles the three shapes seen in the challenge set: a colon-introduced
    list ending in "or X", a bare list after "Answer choices:", and an
    un-introduced list ("...with Memantine, L-serine, or Radiprodil"), plus
    the lettered "(A) text (B) text" form. Returns [] when no list is found,
    so the caller can leave the answer untouched rather than guess.
    """
    lettered = _LETTERED_OPTIONS.findall(prompt)
    if lettered:
        return [text.strip() for _, text in lettered]

    m = _LEADIN_LIST.search(prompt)
    if m:
        return _split_list_items(m.group(1))

    m = _OR_LIST.search(prompt)
    if m and re.search(r"\bor\b", m.group(1)):
        items = _split_list_items(m.group(1))
        if ":" not in prompt[: m.start(1)]:
            items[0] = _STEM_WORDS.split(items[0])[-1].strip()
        return items

    return []


def _normalize_multiple_choice(text: str, prompt: str) -> str:
    """Map model output onto one of the prompt's options, in the prompt's spelling.

    Falls through to the raw text when nothing matches: an unmatched answer
    should score as wrong and be visible in the eval output, not be repaired.
    """
    options = extract_options(prompt)
    cleaned = text.strip()
    if not options:
        return cleaned

    lower = cleaned.lower()

    for opt in options:
        if lower == opt.lower():
            return opt

    # Letter answers only mean something when the prompt lettered its options.
    lettered = _LETTERED_OPTIONS.findall(prompt)
    if lettered:
        m = re.match(r"^\(?([a-z])\)?[.):]?(?:\s|$)", lower)
        if m:
            for letter, opt_text in lettered:
                if letter == m.group(1).upper():
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
