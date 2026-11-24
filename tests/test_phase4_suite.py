import json
import shutil
from pathlib import Path

from loopbench.backends.base import Backend
from loopbench.data.generate import generate_dataset
from loopbench.data.validate import validate_dataset
from loopbench.metrics import evaluate_synth320_extension_rule
from loopbench.report import MOCK_STAMP, build_phase4_report
from loopbench.runner import run
from loopbench.storage import load_rows


class _SynthMockBackend(Backend):
    def generate(self, question, config):
        passes = int(config["loop_num_passes"])
        layers = 24 + 12 * (passes - 1)
        is_hard = question.get("metadata", {}).get("difficulty") == "hard"
        if not is_hard:
            ok = True
        else:
            ok = passes >= 3 or question["task_type"] == "direct"
        ans = float(question["answer"]) if ok else float(question["answer"]) + 15.0
        return {
            "text": f"Final answer: {ans:g}",
            "finish_reason": "stop",
            "ttft_ms": 115.0 * (layers / 24.0),
            "tpot_ms": 34.0 * (layers / 24.0),
            "e2e_ms": (6500.0 if is_hard else 4500.0) * (layers / 24.0),
            "peak_vram_mb": 50670.0 + 55.0 * (passes - 1),
            "reasoning_tokens": 340 if is_hard else 210,
            "answer_tokens": 25,
            "visual_tokens": 400,
            "effective_layers_verified": layers,
            "is_mock": True,
        }


def test_synth240_full_generation_validation_and_prefix_invariance(tmp_path):
    ds240 = tmp_path / "synth240"
    ds320 = tmp_path / "synth320"

    generate_dataset(ds240, seed=20261124, n_per_type=60, profile="synth", seed_mode="per_item")
    res240 = validate_dataset(ds240, expected_images=120, expected_questions=240)
    assert res240["valid"] is True
    assert res240["images"] == 120
    assert res240["questions"] == 240
    assert res240["templates"] == 24
    assert res240["modalities"] == {"chart": 120, "geometry": 120}
    assert res240["difficulties"] == {"standard": 120, "hard": 120}

    # Generate prefix-extended subset (e.g. n_per_type=16 vs n_per_type=12) or full synth-320 check on n=16 vs 12
    # Wait: let's test prefix invariance directly on ds240 vs ds320!
    generate_dataset(ds320, seed=20261124, n_per_type=80, profile="synth", seed_mode="per_item")
    res320 = validate_dataset(ds320, expected_images=160, expected_questions=320)
    assert res320["valid"] is True
    assert res320["images"] == 160
    assert res320["questions"] == 320
    assert res320["difficulties"] == {"standard": 160, "hard": 160}

    m240 = {q["question_id"]: q for q in json.loads((ds240 / "questions.json").read_text(encoding="utf-8"))["questions"]}
    m320 = {q["question_id"]: q for q in json.loads((ds320 / "questions.json").read_text(encoding="utf-8"))["questions"]}
    assert set(m240).issubset(set(m320))
    for qid, q240 in m240.items():
        assert m320[qid] == q240


def test_phase4_prefix_resume_and_report_builder(tmp_path):
    ds24 = tmp_path / "ds24"
    ds32 = tmp_path / "ds32"
    generate_dataset(ds24, seed=20261124, n_per_type=6, profile="synth", seed_mode="per_item")
    generate_dataset(ds32, seed=20261124, n_per_type=8, profile="synth", seed_mode="per_item")

    base_cfg = {
        "model_id": "mock://phase4",
        "model_revision": "mock-v4",
        "backend": "mock",
        "loop_num_passes": 1,
        "reasoning_effort": "medium",
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 2048,
        "stop": [],
        "image_size": [768, 768],
        "color_mode": "RGB",
        "render_version": "matplotlib-3.8.4",
        "concurrency": 1,
        "seed": 20261124,
    }
    backend = _SynthMockBackend()
    pass_map: dict[int, Path] = {}
    for p in (1, 2, 3, 5):
        jp24 = tmp_path / f"synth24_loop{p}.jsonl"
        r_init = run({**base_cfg, "loop_num_passes": p}, ds24, jp24, backend=backend)
        assert r_init["new_rows"] == 24

        jp32 = tmp_path / f"synth32_loop{p}.jsonl"
        shutil.copy2(jp24, jp32)
        r_ext = run({**base_cfg, "loop_num_passes": p}, ds32, jp32, backend=backend)
        assert r_ext["new_rows"] == 8
        assert len(load_rows(jp32)) == 32
        pass_map[p] = jp32

    rep_dir = tmp_path / "report"
    payload = build_phase4_report(
        pass_jsonl_map=pass_map,
        out_dir=rep_dir,
        scale_tag="synth32",
        cache_stats={"loads": 1, "hits": 127},
    )
    assert payload["mock"] is True
    assert payload["label"] == MOCK_STAMP
    assert payload["pass_summaries"]["1"]["by_difficulty"]["standard"] == 1.0
    assert payload["pass_summaries"]["1"]["by_difficulty"]["hard"] == 0.5
    assert payload["pass_summaries"]["3"]["by_difficulty"]["hard"] == 1.0
    assert (rep_dir / "phase4_summary.json").is_file()
    assert (rep_dir / "phase4_pass_curve.csv").is_file()
    assert (rep_dir / "phase4_difficulty_breakdown.csv").is_file()
    assert (rep_dir / "pareto_synth32_mock.png").is_file()
    assert (rep_dir / "report.md").is_file()


def test_predeclared_synth320_uncertainty_rule():
    decisive = {
        "loop3_minus_loop1": {"diff_accuracy": 0.08, "diff_cluster_bootstrap_95ci": [0.04, 0.12]},
        "loop5_minus_loop1": {"diff_accuracy": 0.075, "diff_cluster_bootstrap_95ci": [0.035, 0.115]},
        "loop5_minus_loop3": {"diff_accuracy": -0.005, "diff_cluster_bootstrap_95ci": [-0.03, 0.02]},
    }
    res_ok = evaluate_synth320_extension_rule(decisive)
    assert res_ok["trigger_synth320"] is False
    assert res_ok["reasons"] == []

    ambiguous = {
        "loop3_minus_loop1": {"diff_accuracy": 0.032, "diff_cluster_bootstrap_95ci": [-0.012, 0.076]},
        "loop5_minus_loop1": {"diff_accuracy": 0.010, "diff_cluster_bootstrap_95ci": [-0.020, 0.040]},
        "loop5_minus_loop3": {"diff_accuracy": -0.022, "diff_cluster_bootstrap_95ci": [-0.050, 0.006]},
    }
    res_trig = evaluate_synth320_extension_rule(ambiguous)
    assert res_trig["trigger_synth320"] is True
    assert len(res_trig["reasons"]) == 1

    wide_ci = {
        "loop3_minus_loop1": {"diff_accuracy": 0.07, "diff_cluster_bootstrap_95ci": [0.01, 0.13]},
    }
    res_wide = evaluate_synth320_extension_rule(wide_ci)
    assert res_wide["trigger_synth320"] is True
    assert "exceeds 5.0 pp" in res_wide["reasons"][0]
