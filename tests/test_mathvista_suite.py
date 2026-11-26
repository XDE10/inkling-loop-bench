import json
from pathlib import Path
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from loopbench.config import load_config
from loopbench.data.generate import pixel_sha256
from loopbench.data.mathvista import grade_mathvista_numeric, validate_mathvista_holdout
from loopbench.storage import append_jsonl
from run_phase7_mathvista import build_mathvista_report


def test_mathvista_numeric_grader():
    cfg = load_config(ROOT / "configs/phase7_mathvista.yaml")
    assert cfg["backend"] == "transformers"
    assert cfg["image_size"] == "native"
    assert grade_mathvista_numeric(30.0, 30.0, answer_type="integer", precision=None) is True
    assert grade_mathvista_numeric(30.1, 30.0, answer_type="integer", precision=None) is False
    assert grade_mathvista_numeric(130.92, 130.9, answer_type="float", precision=1) is True
    assert grade_mathvista_numeric(130.9, 130.9, answer_type="float", precision=1) is True
    assert grade_mathvista_numeric(131.1, 130.9, answer_type="float", precision=1) is False
    assert grade_mathvista_numeric(0.734, 0.73, answer_type="float", precision=2) is True
    assert grade_mathvista_numeric(None, 10.0, answer_type="integer") is False


def test_mathvista_validation_and_mock_report(tmp_path: Path):
    ds_dir = tmp_path / "ds"
    (ds_dir / "images").mkdir(parents=True)
    questions = []
    for idx, src in enumerate(("GEOS", "FunctionQA", "IQTest", "TabMWP")):
        img_id = f"mathvista-{idx:04d}"
        rel = f"images/{img_id}.png"
        full = ds_dir / rel
        Image.new("RGB", (256, 256), (20 * idx, 40, 60)).save(full)
        questions.append({
            "question_id": f"{img_id}-q1",
            "image_id": img_id,
            "template_id": f"mathvista-{src.lower()}",
            "modality": "geometry" if src == "GEOS" else "math_visual",
            "task_type": "geometry_problem_solving",
            "prompt": "Find x.",
            "answer": float(10 + idx),
            "unit": "numeric",
            "image_path": rel,
            "metadata": {
                "source": src,
                "answer_type": "integer",
                "precision": None,
                "pixel_sha256": pixel_sha256(full),
            },
        })
    qjson = ds_dir / "questions.json"
    qjson.write_text(
        json.dumps({
            "schema_version": 2,
            "profile": "holdout_mathvista_hard",
            "source_dataset": "AI4Math/MathVista",
            "source_revision": "2b6ad69445fbb5695c9b165475e8decdbeb97747",
            "by_source": {"GEOS": 1, "FunctionQA": 1, "IQTest": 1, "TabMWP": 1},
            "image_count": 4,
            "question_count": 4,
            "questions": questions,
        }),
        encoding="utf-8",
    )
    valid = validate_mathvista_holdout(ds_dir, expected_questions=4)
    assert valid["valid"] is True

    # Write mock JSONL traces for low_loop1 and low_loop3
    traces_dir = tmp_path / "traces"
    p_l1 = traces_dir / "mathvista83_low_loop1.jsonl"
    p_l3 = traces_dir / "mathvista83_low_loop3.jsonl"
    for idx, q in enumerate(questions):
        append_jsonl(p_l1, {
            "config_id": "c1",
            "question_id": q["question_id"],
            "image_id": q["image_id"],
            "expected": q["answer"],
            "predicted": q["answer"] if idx % 2 == 0 else q["answer"] + 5,
            "finish_reason": "stop",
            "outcome": "correct" if idx % 2 == 0 else "wrong_answer",
            "correct": idx % 2 == 0,
            "e2e_ms": 10000.0,
            "ttft_ms": 100.0,
            "tpot_ms": 35.0,
            "reasoning_tokens": 200,
            "peak_vram_mb": 50600.0,
            "is_mock": True,
        })
        append_jsonl(p_l3, {
            "config_id": "c3",
            "question_id": q["question_id"],
            "image_id": q["image_id"],
            "expected": q["answer"],
            "predicted": q["answer"] if idx < 3 else q["answer"] + 5,
            "finish_reason": "stop",
            "outcome": "correct" if idx < 3 else "wrong_answer",
            "correct": idx < 3,
            "e2e_ms": 22000.0,
            "ttft_ms": 200.0,
            "tpot_ms": 80.0,
            "reasoning_tokens": 180,
            "peak_vram_mb": 50750.0,
            "is_mock": True,
        })

    rep_dir = tmp_path / "report"
    out = build_mathvista_report(
        cell_jsonl_map={("low", 1): p_l1, ("low", 3): p_l3},
        questions_json=qjson,
        out_dir=rep_dir,
    )
    assert out["mock"] is True
    assert out["cell_summaries"]["low_loop1"]["accuracy"] == 0.5
    assert out["cell_summaries"]["low_loop3"]["accuracy"] == 0.75
    assert (rep_dir / "mathvista_summary.json").is_file()
    assert (rep_dir / "mathvista_grid.csv").is_file()
    assert (rep_dir / "pareto_mathvista_hard_mock.png").is_file()
