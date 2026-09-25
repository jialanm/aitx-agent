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


def _normalize_multiple_choice(text: str, prompt: str) -> str:
    """Match model output to one of the provided options."""
    # Extract options from prompt: (A) text (B) text ...
    options = re.findall(r"\(([A-Z])\)\s*(.+?)(?=\s*\([A-Z]\)|$)", prompt)

    if not options:
        return text.strip()

    lower = text.lower().strip()

    # Check for letter match: "(A)", "A)", "A.", just "A"
    letter_match = re.match(r"^\(?([a-z])\)?[\.\):]?\s*", lower)
    if letter_match:
        letter = letter_match.group(1).upper()
        for opt_letter, opt_text in options:
            if opt_letter == letter:
                return opt_text.strip()

    # Check for exact option text match
    for opt_letter, opt_text in options:
        opt_clean = opt_text.strip()
        if opt_clean.lower() in lower:
            return opt_clean

    # Fuzzy: check if the model output contains key words from an option
    best_match = None
    best_score = 0
    for opt_letter, opt_text in options:
        opt_clean = opt_text.strip()
        opt_words = set(opt_clean.lower().split())
        text_words = set(lower.split())
        overlap = len(opt_words & text_words)
        if overlap > best_score:
            best_score = overlap
            best_match = opt_clean

    if best_match and best_score > 0:
        return best_match

    return text.strip()


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
