from loopbench.backends.base import Backend
from loopbench.data.generate import generate_dataset
from loopbench.metrics import summarize
from loopbench.report import MOCK_STAMP, build_report
from loopbench.runner import run
from loopbench.storage import load_rows

CFG = {
    "model_id": "mock://test",
    "model_revision": "mock-v1",
    "backend": "mock",
    "loop_num_passes": 3,
    "reasoning_effort": "medium",
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 128,
    "stop": [],
    "image_size": [768, 768],
    "color_mode": "RGB",
    "render_version": "matplotlib-3.8.4",
    "concurrency": 1,
    "seed": 11,
}


class MixedOutcomeBackend(Backend):
    def __init__(self):
        self.calls = 0

    def generate(self, question, config):
        idx = self.calls
        self.calls += 1
        mode = idx % 5
        if mode == 0:
            return {"text": f"Final answer: {question['answer']:g}", "finish_reason": "stop", "e2e_ms": 10.0, "is_mock": True}
        if mode == 1:
            return {"text": f"Final answer: {question['answer'] + 99:g}", "finish_reason": "stop", "e2e_ms": 12.0, "is_mock": True}
        if mode == 2:
            return {"text": "I cannot determine a numeric result.", "finish_reason": "stop", "e2e_ms": 14.0, "is_mock": True}
        if mode == 3:
            return {"text": f"Thinking {question['answer']:g}...", "finish_reason": "length", "e2e_ms": 20.0, "is_mock": True}
        raise RuntimeError("Simulated device timeout")


def test_five_way_error_accounting_and_report(tmp_path):
    ds = tmp_path / "ds"
    generate_dataset(ds, seed=11, n_per_type=6)  # 24 questions
    jsonl = tmp_path / "results.jsonl"
    run(CFG, ds, jsonl, backend=MixedOutcomeBackend())
    rows = load_rows(jsonl)
    s = summarize(rows)
    assert s["n"] == 24
    assert (
        s["correct"]
        + s["wrong_answers"]
        + s["parse_failures"]
        + s["truncated"]
        + s["backend_errors"]
        == 24
    )
    assert s["correct"] == 5
    assert s["wrong_answers"] == 5
    assert s["parse_failures"] == 5
    assert s["truncated"] == 5
    assert s["backend_errors"] == 4
    assert s["mock"] is True

    rep_dir = tmp_path / "report"
    rep = build_report(jsonl, rep_dir)
    assert rep["label"] == MOCK_STAMP
    assert MOCK_STAMP in (rep_dir / "report.md").read_text(encoding="utf-8")
    assert (rep_dir / "pareto_mock.png").is_file()


def test_leakage_control_image_modes(tmp_path):
    ds = tmp_path / "ds"
    generate_dataset(ds, seed=11, n_per_type=6)

    blank_jsonl = tmp_path / "blank.jsonl"
    run({**CFG, "image_mode": "blank"}, ds, blank_jsonl)
    s_blank = summarize(load_rows(blank_jsonl))
    assert s_blank["accuracy"] == 0.0
    assert s_blank["parse_failures"] == 24

    mismatch_jsonl = tmp_path / "mismatch.jsonl"
    run({**CFG, "image_mode": "mismatched"}, ds, mismatch_jsonl)
    s_mis = summarize(load_rows(mismatch_jsonl))
    assert s_mis["accuracy"] == 0.0
    assert s_mis["wrong_answers"] == 24
