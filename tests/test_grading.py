from loopbench.grading import extract_number, grade


def test_grading_outcomes():
    res_ok = grade("Step 1: 10 + 32. Final Answer: 42", 42)
    assert res_ok["correct"] is True
    assert res_ok["outcome"] == "correct"

    res_wrong = grade("answer 41", 42)
    assert res_wrong["correct"] is False
    assert res_wrong["outcome"] == "wrong_answer"

    res_parse = grade("answer: unknown", 42)
    assert res_parse["parse_status"] == "parse_failure"
    assert res_parse["outcome"] == "parse_failure"

    res_trunc = grade("Let us compute: angle A is 42", 42, finish_reason="length")
    assert res_trunc["correct"] is False
    assert res_trunc["outcome"] == "truncated"

    res_err = grade("", 42, finish_reason="error")
    assert res_err["correct"] is False
    assert res_err["outcome"] == "backend_error"


def test_extract_number_formats_and_trailing_asides():
    assert extract_number("x=-1.25") == -1.25
    assert extract_number(r"Therefore \boxed{1,250}.") == 1250.0
    assert extract_number(r"**Answer: \(\boxed{35^\circ}\)**") == 35.0
    assert extract_number("Intermediate 99; Final answer is 55 degrees.") == 55.0

    # Real Phase 2 TML-d24-varloop-research outputs with helpful trailing parenthetical / Note asides:
    s1 = "The marked interior angle at vertex **C** is **48°**.  \n(If you meant the exterior angle at **B**, it would be **98°**.)"
    assert grade(s1, 48.0)["correct"] is True

    s2 = "The marked angle **∠ADB** is **80°** as shown in the figure.\n\nNote: the red unknown angle at **B** would be **35°**."
    assert grade(s2, 80.0)["correct"] is True

    s3 = "The marked angle **ADB** is **80°**.\n\nIf you meant the red unknown angle at **B** (∠DBC), it is **35°**."
    assert grade(s3, 80.0)["correct"] is True

    s4 = "The marked interior angle at vertex **C** is **48°**.\n\n(Note: the exterior angle at B would be \\(50^\\circ + 48^\\circ = 98^\\circ\\).)"
    assert grade(s4, 48.0)["correct"] is True

    assert extract_number(r"Result: \boxed{65\text{ units}} (where max was 90).") == 65.0
    assert extract_number(r"Result: \boxed{35^\circ} in triangle CBD (with C = 45^\circ).") == 35.0
    assert extract_number(r"**Answer**: \(40^\circ\)") == 40.0

    # Fractions, scientific notation, LaTeX comma grouping, Unicode minus
    assert abs(extract_number(r"\boxed{\frac{3}{4}}") - 0.75) < 1e-9
    assert abs(extract_number("Answer: 3/4") - 0.75) < 1e-9
    assert abs(extract_number(r"\boxed{-\frac{5}{2}}") - (-2.5)) < 1e-9
    assert abs(extract_number("1.5e3") - 1500.0) < 1e-9
    assert abs(extract_number(r"Final answer is 2.5 \times 10^3") - 2500.0) < 1e-9
    assert abs(extract_number(r"\boxed{12{,}500}") - 12500.0) < 1e-9
    assert abs(extract_number("The value is \u22124.") - (-4.0)) < 1e-9
