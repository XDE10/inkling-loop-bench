import re

_NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_FINAL_PATTERNS = (
    re.compile(r"\\boxed\{\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\}"),
    re.compile(
        r"(?:final\s+answer|computed\s+answer|answer)\s*(?:is|:|=)\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        re.IGNORECASE,
    ),
)


def extract_number(text) -> float | None:
    if not isinstance(text, str) or not text.strip():
        return None
    for pat in _FINAL_PATTERNS:
        found = pat.findall(text)
        if found:
            return float(found[-1].replace(",", ""))
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
