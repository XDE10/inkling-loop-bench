import json
from pathlib import Path
from loopbench.data.generate import generate_dataset
from loopbench.data.validate import validate_dataset


def test_smoke48_determinism_and_templates(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_dataset(a, seed=42)
    generate_dataset(b, seed=42)
    summary = validate_dataset(a, expected_images=24, expected_questions=48)
    assert summary["questions"] == 48
    assert summary["images"] == 24
    assert summary["templates"] == 12

    ja = json.loads((a / "questions.json").read_text(encoding="utf-8"))
    jb = json.loads((b / "questions.json").read_text(encoding="utf-8"))
    assert ja == jb

    subtypes = {q["metadata"]["subtype"] for q in ja["questions"]}
    assert len(subtypes) == 12
    for q in ja["questions"]:
        assert not Path(q["image_path"]).is_absolute()
        assert q["image_path"].startswith("images/")


def test_variable_scale_dataset_validation(tmp_path):
    d = tmp_path / "mini12"
    generate_dataset(d, seed=99, n_per_type=6)
    res = validate_dataset(d, expected_images=12, expected_questions=24)
    assert res["images"] == 12
    assert res["questions"] == 24
