from loopbench.grading import grade,extract_number
def test_grading():
    assert grade("answer 42",42)["correct"]
    assert not grade("answer 41",42)["correct"]
    assert grade("answer: unknown",42)["parse_status"]=="parse_failure"
    assert extract_number("x=-1.25") == -1.25
