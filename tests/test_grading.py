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


def test_extract_number_formats():
    assert extract_number("x=-1.25") == -1.25
    assert extract_number(r"Therefore \boxed{1,250}.") == 1250.0
    assert extract_number("Intermediate 99; Final answer is 55 degrees.") == 55.0
