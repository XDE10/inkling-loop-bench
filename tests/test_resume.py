import pytest
from loopbench.data.generate import generate_dataset
from loopbench.runner import run
from loopbench.storage import append_jsonl, load_completed, load_rows

BASE_CFG = {
    "model_id": "mock",
    "model_revision": "v1",
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
    "render_version": "test",
    "concurrency": 1,
    "seed": 7,
}


def test_resume(tmp_path):
    d = tmp_path / "data"
    generate_dataset(d, seed=7)
    out = tmp_path / "results.jsonl"
    assert run(BASE_CFG, d, out)["new_rows"] == 48
    assert run(BASE_CFG, d, out)["new_rows"] == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 48


def test_trailing_partial_and_nul_fault_recovery(tmp_path):
    p = tmp_path / "r.jsonl"
    append_jsonl(p, {"config_id": "c", "question_id": "q1"})
    with open(p, "ab") as f:
        f.write(b'{"config_id":"c","question_id":')
    assert load_completed(p) == {("c", "q1")}
    assert p.read_bytes().endswith(b"\n")

    # Fault injection: terminated corrupt/zero-filled tail line from sudden crash
    with open(p, "ab") as f:
        f.write(b'\x00\x00{"config_id":"c"\n')
    assert load_completed(p) == {("c", "q1")}
    assert len(load_rows(p)) == 1


def test_mid_file_corruption_raises(tmp_path):
    p = tmp_path / "corrupt.jsonl"
    append_jsonl(p, {"config_id": "c", "question_id": "q1"})
    with open(p, "ab") as f:
        f.write(b'{"corrupt_middle_line": true}\n')
    append_jsonl(p, {"config_id": "c", "question_id": "q2"})
    with pytest.raises(ValueError, match="Corrupt JSONL at line 2"):
        load_completed(p)
