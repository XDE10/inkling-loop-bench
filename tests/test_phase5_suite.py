import io
import json
from pathlib import Path
from PIL import Image

from loopbench.backends.base import Backend
from loopbench.data.generate import generate_dataset
from loopbench.data.holdout import grade_chartqa_relaxed, validate_holdout_dataset
from loopbench.oracle import analyze_offline_oracle
from loopbench.runner import run
from loopbench.storage import atomic_write_text, load_rows


class _EffortAndHoldoutMockBackend(Backend):
    def generate(self, question, config):
        passes = int(config["loop_num_passes"])
        effort = str(config["reasoning_effort"])
        layers = 24 + 12 * (passes - 1)
        is_hard = question.get("metadata", {}).get("difficulty") == "hard"
        is_multi = question.get("task_type") == "multi_step"

        # Simulate realistic latent-looping x explicit-effort trade-off:
        # At 'none'/'low', deeper passes rescue multi-step accuracy at low token cost.
        if effort == "none":
            ok = (not is_multi) or (passes >= 3 and not is_hard) or (passes >= 5)
            r_tok = 0
        elif effort == "low":
            ok = (not is_hard) or (passes >= 3)
            r_tok = 55
        elif effort == "medium":
            ok = True
            r_tok = 260
        else:
            ok = True
            r_tok = 520

        ans = float(question["answer"]) if ok else float(question["answer"]) + 11.0
        e2e_ms = (120.0 + (r_tok + 20) * 34.0) * (layers / 24.0)
        return {
            "text": f"Final answer: {ans:g}",
            "finish_reason": "stop",
            "ttft_ms": 115.0 * (layers / 24.0),
            "tpot_ms": 34.0 * (layers / 24.0),
            "e2e_ms": e2e_ms,
            "peak_vram_mb": 50670.0 + 56.0 * (passes - 1),
            "reasoning_tokens": r_tok,
            "answer_tokens": 20,
            "visual_tokens": 420 if config.get("image_size") == "native" else 400,
            "effective_layers_verified": layers,
            "is_mock": True,
        }


def _make_mini_holdout_dir(root: Path) -> Path:
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    from loopbench.data.generate import pixel_sha256

    qs = []
    for i in range(4):
        img_id = f"chartqa-{i:03d}"
        rel = f"images/{img_id}.png"
        full = root / rel
        Image.new("RGB", (640 + i * 20, 480), (240, 240, 250)).save(full)
        sha = pixel_sha256(full)
        for slot, val in (("q1", 25.0 + i), ("q2", 0.40 + 0.05 * i)):
            qs.append({
                "question_id": f"{img_id}-{slot}",
                "image_id": img_id,
                "template_id": "chartqa-human-numeric",
                "modality": "chart",
                "task_type": "holdout_human",
                "prompt": f"What is metric {slot}? Answer with a single number.",
                "answer": val,
                "unit": "chart_value",
                "image_path": rel,
                "metadata": {"pixel_sha256": sha, "difficulty": "real_world"},
            })
    manifest = {
        "schema_version": 2,
        "profile": "holdout_chartqa",
        "source_dataset": "HuggingFaceM4/ChartQA",
        "source_revision": "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5",
        "image_count": 4,
        "question_count": 8,
        "questions": qs,
    }
    atomic_write_text(root / "questions.json", json.dumps(manifest, indent=2) + "\n")
    return root


def test_chartqa_relaxed_grader_and_holdout_validation(tmp_path):
    assert grade_chartqa_relaxed(103.0, 100.0) is True
    assert grade_chartqa_relaxed(106.0, 100.0) is False
    assert grade_chartqa_relaxed(0.004, 0.0) is True
    assert grade_chartqa_relaxed(None, 10.0) is False

    hd = _make_mini_holdout_dir(tmp_path / "mini_chartqa")
    valid = validate_holdout_dataset(hd, expected_images=4, expected_questions=8)
    assert valid["valid"] is True
    assert valid["images"] == 4
    assert valid["questions"] == 8


def test_phase5_separate_reports_and_offline_oracle(tmp_path):
    from loopbench.report import MOCK_STAMP, build_phase5_reports

    ds_eff = tmp_path / "ds_eff"
    generate_dataset(ds_eff, seed=20261124, n_per_type=4, profile="synth", seed_mode="per_item")
    ds_hold = _make_mini_holdout_dir(tmp_path / "ds_hold")

    eff_cfg = {
        "model_id": "mock://p5",
        "model_revision": "mock-p5",
        "backend": "mock",
        "loop_num_passes": 1,
        "reasoning_effort": "none",
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
        "stop": [],
        "image_size": [768, 768],
        "color_mode": "RGB",
        "render_version": "matplotlib-3.8.4",
        "concurrency": 1,
        "seed": 20261124,
    }
    hold_cfg = {**eff_cfg, "image_size": "native", "reasoning_effort": "medium"}
    backend = _EffortAndHoldoutMockBackend()

    effort_map = {}
    oracle_input = {}
    for eff in ("none", "low", "medium", "high"):
        for p in (1, 3, 5):
            jl = tmp_path / f"eff_{eff}_loop{p}.jsonl"
            run({**eff_cfg, "reasoning_effort": eff, "loop_num_passes": p}, ds_eff, jl, backend=backend)
            effort_map[(eff, p)] = jl
            oracle_input[f"{eff}_loop{p}"] = load_rows(jl)

    oracle_res = analyze_offline_oracle(oracle_input)
    assert oracle_res["router_oracle"]["accuracy"] == 1.0
    assert oracle_res["router_oracle"]["mean_speedup_vs_best_fixed"] > 1.5

    holdout_map = {}
    for p in (1, 2, 3, 5):
        jl = tmp_path / f"hold_loop{p}.jsonl"
        run({**hold_cfg, "loop_num_passes": p}, ds_hold, jl, backend=backend)
        holdout_map[p] = jl

    rep_out = tmp_path / "report_p5"
    res = build_phase5_reports(
        effort_jsonl_map=effort_map,
        holdout_jsonl_map=holdout_map,
        out_dir=rep_out,
        effort_tag="effort16",
        holdout_tag="chartqa8",
        cache_stats={"loads": 1, "hits": 15},
    )
    assert res["effort_matrix"]["mock"] is True
    assert res["effort_matrix"]["label"] == MOCK_STAMP
    assert res["holdout"]["mock"] is True
    assert (rep_out / "effort_matrix" / "effort_summary.json").is_file()
    assert (rep_out / "effort_matrix" / "effort_grid.csv").is_file()
    assert (rep_out / "effort_matrix" / "pareto_effort_matrix_mock.png").is_file()
    assert (rep_out / "holdout_chartqa8" / "holdout_summary.json").is_file()
    assert (rep_out / "holdout_chartqa8" / "holdout_pass_curve.csv").is_file()
    assert (rep_out / "holdout_chartqa8" / "pareto_holdout_chartqa8_mock.png").is_file()
