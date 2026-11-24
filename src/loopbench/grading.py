import re

_NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_FINAL_PATTERNS = (
    re.compile(
        r"\\boxed\{\s*(?:\\(?:mathbf|textbf|mathrm)\{\s*)?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\}?"
        r"\s*(?:\^?\{?\\circ\}?|°|\\text\{[^}]*\}|\s*[A-Za-z%°]*)?\s*\}"
    ),
    re.compile(
        r"(?:final\s+answer|computed\s+answer|answer)\s*\*{0,2}\s*(?:is|:|=)\s*\*{0,2}"
        r"\s*(?:\\\(|\\\[|\$)*\s*(?:\\boxed\{\s*)?(?:\\(?:mathbf|textbf|mathrm)\{\s*)?"
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        re.IGNORECASE,
    ),
)
_TRAILING_NOTE_RE = re.compile(
    r"(?:(?:\n+|\s{2,}|\.\s+)(?:note|aside|remark)\s*:.*|"
    r"(?:\n+|\.\s+)(?:if\s+you\s+(?:meant|mean|are\s+asking)|in\s+case\s+you\s+meant)\b.*)+$",
    re.IGNORECASE | re.S,
)
_TRAILING_NOTE_PAREN_RE = re.compile(
    r"\s*\(\s*(?:note|aside|remark|if\s+you\s+(?:meant|mean))\b[\s\S]*\)\s*$",
    re.IGNORECASE,
)
_TRAILING_PAREN_RE = re.compile(r"(?:\s*\([^()]*\)\s*)+$")


def _strip_trailing_asides(text: str) -> str:
    cleaned = _TRAILING_NOTE_PAREN_RE.sub("", text.strip())
    cleaned = _TRAILING_NOTE_RE.sub("", cleaned.strip())
    cleaned = _TRAILING_PAREN_RE.sub("", cleaned).strip()
    return cleaned or text


def extract_number(text) -> float | None:
    if not isinstance(text, str) or not text.strip():
        return None
    for pat in _FINAL_PATTERNS:
        found = pat.findall(text)
        if found:
            return float(found[-1].replace(",", ""))
    primary = _strip_trailing_asides(text)
    matches = _NUMBER_RE.findall(primary)
    if not matches:
        matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def grade(text, expected, tolerance: float = 1e-6, finish_reason: str = "stop") -> dict:
    if finish_reason == "error":
        return {
            "correct": False,
            "parse_status": "backend_error",
            "outcome": "backend_error",
            "predicted": None,
        }
    if finish_reason == "length":
        return {
            "correct": False,
            "parse_status": "truncated",
            "outcome": "truncated",
            "predicted": extract_number(text),
        }
    actual = extract_number(text)
    if actual is None:
        return {
            "correct": False,
            "parse_status": "parse_failure",
            "outcome": "parse_failure",
            "predicted": None,
        }
    is_correct = abs(actual - float(expected)) <= tolerance
    return {
        "correct": is_correct,
        "parse_status": "parsed",
        "outcome": "correct" if is_correct else "wrong_answer",
        "predicted": actual,
    }
