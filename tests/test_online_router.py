import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from loopbench.online_router import (
    evaluate_binary_policy,
    evaluate_online_routing_pair,
    evaluate_zero_shot_transfer,
    extract_observable_features,
    make_group_folds,
)
from loopbench.storage import append_jsonl
from run_phase7_online_router import run_phase7_online_router_suite


def _make_synthetic_pair(n_images: int = 12):
    base_rows = []
    esc_rows = []
    prompts = {}
    for i in range(n_images):
        is_geom = i % 2 == 0
        img_id = f"{'geometry' if is_geom else 'chart'}-{i:03d}"
        tpl = f"{'geometry-v1' if is_geom else 'chart-v1'}-{i % 3}"
        for task in ("direct", "multi_step"):
            qid = f"{img_id}-{task}"
            prompt = (
                "Find the angle at vertex A in degrees." if (is_geom and task == "direct")
                else "Solve for the unknown angle marked '?'." if is_geom
                else "Read the bar value for Q2." if task == "direct"
                else "Compute the total sum across all bars."
            )
            prompts[qid] = prompt
            # Make geometry/direct fail at base and succeed at escalated
            base_ok = not (is_geom and task == "direct")
            esc_ok = True
            base_rows.append({
                "config_id": "base",
                "question_id": qid,
                "image_id": img_id,
                "template_id": tpl,
                "modality": "geometry" if is_geom else "chart",
                "task_type": task,
                "outcome": "correct" if base_ok else "wrong_answer",
                "correct": base_ok,
                "finish_reason": "stop",
                "text": "The answer is 45. (Note: other angle is 60)" if not base_ok else "45",
                "reasoning_text": "Wait, let me double-check?" if not base_ok else "Direct read 45.",
                "reasoning_tokens": 750 if not base_ok else 150,
                "answer_tokens": 90 if not base_ok else 12,
                "visual_tokens": 400,
                "input_tokens": 430,
                "e2e_ms": 10_000.0,
                "is_mock": True,
            })
            esc_rows.append({
                "config_id": "esc",
                "question_id": qid,
                "image_id": img_id,
                "template_id": tpl,
                "modality": "geometry" if is_geom else "chart",
                "task_type": task,
                "outcome": "correct" if esc_ok else "wrong_answer",
                "correct": esc_ok,
                "finish_reason": "stop",
                "text": "45",
                "reasoning_text": "Clean check 45.",
                "reasoning_tokens": 350,
                "answer_tokens": 12,
                "visual_tokens": 400,
                "input_tokens": 430,
                "e2e_ms": 30_000.0,
                "is_mock": True,
            })
    return base_rows, esc_rows, prompts


def test_feature_extraction_and_grouped_folds():
    base_rows, esc_rows, prompts = _make_synthetic_pair(10)
    f = extract_observable_features(base_rows[0], prompts[base_rows[0]["question_id"]])
    assert f["is_geom_prompt"] == 1.0
    assert f["has_aside_in_answer"] == 1.0
    assert f["multi_answer_numbers"] == 1.0
    assert f["hedge_count"] >= 2.0

    # Paired image questions must ALWAYS land in the same CV fold
    groups = [r["image_id"] for r in base_rows]
    folds, k = make_group_folds(groups, n_folds=5)
    assert k == 5
    by_img = {}
    for g, fl in zip(groups, folds):
        by_img.setdefault(g, set()).add(fl)
    assert all(len(s) == 1 for s in by_img.values())

    # Cascade latency must be cumulative (base + esc) while pregen is replacement (esc only)
    dec = [True] + [False] * (len(base_rows) - 1)
    cas = evaluate_binary_policy(base_rows, esc_rows, dec, mode="cascade", n_boot=100)
    pre = evaluate_binary_policy(base_rows, esc_rows, dec, mode="pregen", n_boot=100)
    assert cas["total_gpu_s"] > pre["total_gpu_s"]
    assert abs((cas["total_gpu_s"] - pre["total_gpu_s"]) - 10.0) < 1e-6


def test_online_router_cv_and_zero_shot_suite(tmp_path: Path):
    base_rows, esc_rows, prompts = _make_synthetic_pair(12)
    res = evaluate_online_routing_pair(
        base_rows, esc_rows, prompts,
        base_label="loop1", esc_label="loop5",
        group_key="image_id", n_folds=4, n_boot=100,
    )
    assert res["policies"]["offline_router_oracle_2cfg"]["accuracy"] == 1.0
    assert res["policies"]["pregen_ridge_lam0p0"]["accuracy"] > res["policies"]["fixed_loop1"]["accuracy"]

    # Zero-shot transfer uses frozen model without refitting
    xfer = evaluate_zero_shot_transfer(
        res["_fitted_runtime"], base_rows, esc_rows, prompts,
        base_label="loop1", esc_label="loop5", n_boot=100,
    )
    assert "zeroshot_pregen_ridge_lam0p0" in xfer["policies"]

    # End-to-end Phase 7A script runner on mock JSONLs
    p4_dir = tmp_path / "p4"
    p5_dir = tmp_path / "p5"
    out_dir = tmp_path / "out"
    for r in base_rows:
        append_jsonl(p4_dir / "synth240_loop1.jsonl", r)
        append_jsonl(p4_dir / "synth240_loop3.jsonl", r)
        append_jsonl(p5_dir / "effort48_low_loop1.jsonl", r)
    for r in esc_rows:
        append_jsonl(p4_dir / "synth240_loop5.jsonl", r)
        append_jsonl(p5_dir / "effort48_low_loop3.jsonl", r)
    qjson = tmp_path / "q.json"
    qjson.write_text(json.dumps({"questions": [{"question_id": k, "prompt": v} for k, v in prompts.items()]}), encoding="utf-8")

    summary = run_phase7_online_router_suite(
        phase4_dir=p4_dir,
        phase5_dir=p5_dir,
        out_dir=out_dir,
        p4_questions_json=qjson,
        p5_effort_questions_json=qjson,
        n_boot=100,
    )
    assert summary["mock"] is True
    assert (out_dir / "phase7_online_router_summary.json").is_file()
    assert (out_dir / "phase7_online_router_summary.csv").is_file()
    assert (out_dir / "online_router_pareto_mock.png").is_file()
