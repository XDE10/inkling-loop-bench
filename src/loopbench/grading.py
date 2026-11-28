import re

# Single numeric token (supports integer/decimal, comma grouping, scientific notation)
_NUM_CORE = r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_FRAC_CORE = rf"[-+]?\s*(?:\\(?:d|t)?frac\{{\s*({_NUM_CORE})\s*\}}\{{\s*({_NUM_CORE})\s*\}}|({_NUM_CORE})\s*/\s*({_NUM_CORE})(?!\s*/))"
_SCI_TIMES_RE = re.compile(
    rf"({_NUM_CORE})\s*(?:\\times|×|\*)\s*10\^\{{\s*([-+]?\d+)\s*\}}|({_NUM_CORE})\s*(?:\\times|×|\*)\s*10\^([-+]?\d+)"
)
_VAL_EXPR = rf"(?:[-+]?\s*\\(?:d|t)?frac\{{\s*{_NUM_CORE}\s*\}}\{{\s*{_NUM_CORE}\s*\}}|{_NUM_CORE}\s*(?:\\times|×|\*)\s*10\^(?:\{{\s*[-+]?\d+\s*\}}|[-+]?\d+)|{_NUM_CORE}\s*/\s*{_NUM_CORE}(?!\s*/)|{_NUM_CORE})"

_NUMBER_RE = re.compile(rf"(?<![A-Za-z_])({_VAL_EXPR})")
_BOXED_RE = re.compile(
    rf"\\boxed\{{\s*(?:\\(?:mathbf|textbf|mathrm|text)\{{\s*)?({_VAL_EXPR})\s*\}}?"
    r"\s*(?:\^?\{?\\circ\}?|°|\\text\{[^}]*\}|\s*[A-Za-z%°]*)?\s*\}"
)
_ANSWER_RE = re.compile(
    rf"(?:final\s+answer|computed\s+answer|answer)\s*\**(?:\s*(?:is|:|=)\s*)\**"
    rf"\s*(?:\\\(|\\\[|\$)*\s*(?:\\boxed\{{\s*)?(?:\\(?:mathbf|textbf|mathrm|text)\{{\s*)?"
    rf"({_VAL_EXPR})",
    re.IGNORECASE,
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
_LATEX_COMMA_RE = re.compile(r"(?<=\d)\{,\}(?=\d)")


def _normalize_text(text: str) -> str:
    # Normalize Unicode minus/dashes and LaTeX thousands separators `{,}`
    s = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = _LATEX_COMMA_RE.sub(",", s)
    return s


def _parse_numeric_expr(expr: str) -> float | None:
    s = _normalize_text(expr).strip()
    if not s:
        return None
    # Scientific notation with \times 10^k
    m_sci = _SCI_TIMES_RE.fullmatch(s)
    if m_sci:
        base = float((m_sci.group(1) or m_sci.group(3)).replace(",", ""))
        exp = int(m_sci.group(2) or m_sci.group(4))
        return base * (10.0 ** exp)
    # LaTeX \frac{a}{b} or plain a/b
    m_frac = re.fullmatch(_FRAC_CORE, s)
    if m_frac:
        sign = -1.0 if s.lstrip().startswith("-") and not (m_frac.group(1) or m_frac.group(3) or "").lstrip().startswith("-") else 1.0
        num_s = m_frac.group(1) if m_frac.group(1) is not None else m_frac.group(3)
        den_s = m_frac.group(2) if m_frac.group(2) is not None else m_frac.group(4)
        den = float(den_s.replace(",", ""))
        if abs(den) < 1e-15:
            return None
        val = float(num_s.replace(",", "")) / den
        return -abs(val) if sign < 0 and val > 0 else val
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _strip_trailing_asides(text: str) -> str:
    cleaned = _TRAILING_NOTE_PAREN_RE.sub("", text.strip())
    cleaned = _TRAILING_NOTE_RE.sub("", cleaned.strip())
    cleaned = _TRAILING_PAREN_RE.sub("", cleaned).strip()
    return cleaned or text


def extract_number(text) -> float | None:
    if not isinstance(text, str) or not text.strip():
        return None
    norm = _normalize_text(text)
    for pat in (_BOXED_RE, _ANSWER_RE):
        found = pat.findall(norm)
        if found:
            val = _parse_numeric_expr(found[-1])
            if val is not None:
                return val
    primary = _strip_trailing_asides(norm)
    matches = _NUMBER_RE.findall(primary)
    if not matches:
        matches = _NUMBER_RE.findall(norm)
    for raw in reversed(matches):
        val = _parse_numeric_expr(raw)
        if val is not None:
            return val
    return None


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
