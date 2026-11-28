"""Phase 8A: Official `Inkling-2-Mini` (fixed 3-pass) Cross-Model Protocol Evaluation.

Runs the exact locked LoopBench datasets (`MathVista-Hard-83` at `low` / `medium` / `blank`
and `ChartQA-80` at `medium`) on `thinkingmachines/Inkling-2-Mini`
(`c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, `is_varloop=False`, fixed `48L`) using
throughput-oriented left-padded batching (`--batch-size 4`) plus single-stream Phase 7C
systems profiling so accuracy, CoT length, truncation tax, and KV/latency telemetry can be
compared head-to-head against `TML-d24-varloop-research` (`loop=3`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from loopbench.backends import create_backend
from loopbench.backends.transformers_backend import get_model_cache_stats
from loopbench.config import load_config
from loopbench.data.holdout import build_chartqa_holdout, grade_chartqa_relaxed, validate_holdout_dataset
from loopbench.data.mathvista import build_mathvista_hard_holdout, validate_mathvista_holdout
from loopbench.metrics import cluster_bootstrap_accuracy_ci, paired_cluster_bootstrap_diff_ci, summarize
from loopbench.runner import run
from loopbench.storage import atomic_write_text, load_rows
from run_phase3_smoke48 import _make_progress_printer, collect_env_fingerprint
from run_phase7_mathvista import _regrade_mathvista_rows


def render_mini_comparison_figure(comparison: dict[str, Any], out_path: Path) -> None:
    """Render side-by-side comparison of Official Inkling-2-Mini vs TML-d24-varloop-research (loop=3)."""
    mv_mini = comparison.get("mathvista_mini") or {}
    mv_var = comparison.get("mathvista_varloop_ref") or {}
    cqa_mini = comparison.get("chartqa_mini") or {}
    cqa_var = comparison.get("chartqa_varloop_ref") or {}

    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 9.6}):
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9), dpi=150)
        ax_acc, ax_cot = axes[0], axes[1]

        labels = []
        v_acc, m_acc = [], []
        v_cot, m_cot = [], []
        v_tr, m_tr = [], []

        for key, disp in (("low_loop3", "MathVista-83\n(low, 48L)"), ("medium_loop3", "MathVista-83\n(medium, 48L)")):
            if key in mv_mini and key in mv_var:
                labels.append(disp)
                v_acc.append((mv_var[key].get("accuracy") or 0.0) * 100.0)
                m_acc.append((mv_mini[key].get("accuracy") or 0.0) * 100.0)
                v_cot.append(mv_var[key].get("reasoning_tokens_mean") or 0.0)
                m_cot.append(mv_mini[key].get("reasoning_tokens_mean") or 0.0)
                v_tr.append(mv_var[key].get("truncated") or 0)
                m_tr.append(mv_mini[key].get("truncated") or 0)

        if cqa_mini and cqa_var:
            labels.append("ChartQA-80\n(med ±5%, 48L)")
            v_acc.append((cqa_var.get("relaxed_accuracy") or 0.0) * 100.0)
            m_acc.append((cqa_mini.get("relaxed_accuracy") or 0.0) * 100.0)
            v_cot.append(cqa_var.get("reasoning_tokens_mean") or 0.0)
            m_cot.append(cqa_mini.get("reasoning_tokens_mean") or 0.0)
            v_tr.append(cqa_var.get("truncated") or 0)
            m_tr.append(cqa_mini.get("truncated") or 0)

        if labels:
            xs = np.arange(len(labels))
            w = 0.34
            b1 = ax_acc.bar(xs - w / 2, v_acc, width=w, color="#5B9BD5", edgecolor="#222", label="TML-d24-varloop (loop=3)")
            b2 = ax_acc.bar(xs + w / 2, m_acc, width=w, color="#ED7D31", edgecolor="#222", label="Inkling-2-Mini (fixed 3-pass)")
            for bar in list(b1) + list(b2):
                h = bar.get_height()
                ax_acc.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold")
            ax_acc.set_xticks(xs, labels)
            ax_acc.set_ylabel("Accuracy (%)")
            ax_acc.set_ylim(0, 108)
            ax_acc.grid(axis="y", alpha=0.3)
            ax_acc.legend(loc="lower right", fontsize=8.3, frameon=True)
            ax_acc.set_title("A. Accuracy: Research Varloop vs Official Mini", fontweight="bold")

            c1 = ax_cot.bar(xs - w / 2, v_cot, width=w, color="#5B9BD5", edgecolor="#222", label="TML-d24-varloop (loop=3)")
            c2 = ax_cot.bar(xs + w / 2, m_cot, width=w, color="#ED7D31", edgecolor="#222", label="Inkling-2-Mini (fixed 3-pass)")
            for bar, tr in zip(c1, v_tr):
                h = bar.get_height()
                ax_cot.text(bar.get_x() + bar.get_width() / 2, h + 10, f"{h:.0f}\n(T={tr})", ha="center", va="bottom", fontsize=7.8)
            for bar, tr in zip(c2, m_tr):
                h = bar.get_height()
                ax_cot.text(bar.get_x() + bar.get_width() / 2, h + 10, f"{h:.0f}\n(T={tr})", ha="center", va="bottom", fontsize=7.8)
            ax_cot.set_xticks(xs, labels)
            ax_cot.set_ylabel("Mean CoT Tokens (T = 2048-tok Truncations)")
            ax_cot.set_ylim(0, max(v_cot + m_cot + [100]) * 1.28)
            ax_cot.grid(axis="y", alpha=0.3)
            ax_cot.legend(loc="upper right", fontsize=8.3, frameon=True)
            ax_cot.set_title("B. CoT Length & Truncation Tax", fontweight="bold")

        fig.suptitle("LoopBench Phase 8A: Official Inkling-2-Mini vs TML-d24-varloop-research (Locked Protocol, 48L)", fontweight="bold", fontsize=11.3)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out_path, metadata={"Software": "LoopBench"})
        plt.close(fig)


def _grade_chartqa_pct_norm(predicted: float | None, expected: float) -> bool:
    if grade_chartqa_relaxed(predicted, expected):
        return True
    if predicted is None:
        return False
    return grade_chartqa_relaxed(float(predicted) / 100.0, expected) or grade_chartqa_relaxed(float(predicted) * 100.0, expected)


def summarize_chartqa_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s = summarize(rows)
    rel_rows = [
        {**r, "correct": grade_chartqa_relaxed(r.get("predicted"), float(r["expected"])) and r.get("outcome") not in ("truncated", "backend_error", "parse_failure")}
        for r in rows
    ]
    pct_rows = [
        {**r, "correct": _grade_chartqa_pct_norm(r.get("predicted"), float(r["expected"])) and r.get("outcome") not in ("truncated", "backend_error", "parse_failure")}
        for r in rows
    ]
    s["relaxed_accuracy"] = sum(r["correct"] for r in rel_rows) / max(1, len(rel_rows))
    s["relaxed_ci95"] = cluster_bootstrap_accuracy_ci(rel_rows, n_boot=2000)
    s["pct_norm_accuracy"] = sum(r["correct"] for r in pct_rows) / max(1, len(pct_rows))
    s["pct_norm_ci95"] = cluster_bootstrap_accuracy_ci(pct_rows, n_boot=2000)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase 8A Official Inkling-2-Mini Cross-Model Eval")
    ap.add_argument("--mv-config", default=str(ROOT / "configs/phase8_mini_mathvista.yaml"))
    ap.add_argument("--cqa-config", default=str(ROOT / "configs/phase8_mini_chartqa.yaml"))
    ap.add_argument("--mv-dataset-dir", default=None)
    ap.add_argument("--cqa-dataset-dir", default=None)
    ap.add_argument("--out", default=str(ROOT / "artifacts/phase8a_mini"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--skip-chartqa", action="store_true")
    ap.add_argument("--skip-blank", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mv_cfg = load_config(args.mv_config)
    cqa_cfg = load_config(args.cqa_config)

    mv_ds = Path(args.mv_dataset_dir) if args.mv_dataset_dir else (out_dir / "dataset_mathvista_hard83")
    if not (mv_ds / "questions.json").is_file():
        build_mathvista_hard_holdout(mv_ds)
    validate_mathvista_holdout(mv_ds, expected_questions=None)

    cqa_ds = None
    if not args.skip_chartqa:
        cqa_ds = Path(args.cqa_dataset_dir) if args.cqa_dataset_dir else (out_dir / "dataset_chartqa80")
        if not (cqa_ds / "questions.json").is_file():
            build_chartqa_holdout(cqa_ds)
        validate_holdout_dataset(cqa_ds)

    env_info = collect_env_fingerprint(mv_cfg)
    atomic_write_text(out_dir / "env_fingerprint.json", json.dumps(env_info, indent=2) + "\n")

    backend = create_backend(mv_cfg)
    manifest_mv = json.loads((mv_ds / "questions.json").read_text(encoding="utf-8"))
    meta_mv = {q["question_id"]: q.get("metadata", {}) for q in manifest_mv["questions"]}

    mv_summaries: dict[str, Any] = {}
    for eff in ("low", "medium"):
        cell_cfg = {**mv_cfg, "reasoning_effort": eff, "loop_num_passes": 3, "image_mode": "correct", "concurrency": args.batch_size}
        jsonl_p = out_dir / f"mini_mathvista83_{eff}_loop3.jsonl"
        res = run(cell_cfg, mv_ds, jsonl_p, backend=backend, batch_size=args.batch_size, progress_callback=_make_progress_printer(f"mini mv83 {eff} bs={args.batch_size}"))
        rows = _regrade_mathvista_rows(load_rows(jsonl_p), meta_mv)
        s = summarize(rows)
        by_src: dict[str, list[bool]] = {}
        for r in rows:
            by_src.setdefault(str(r.get("source", "unknown")), []).append(bool(r["correct"]))
        s["by_source_counts"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_src.items())}
        mv_summaries[f"{eff}_loop3"] = s
        print(f"MINI_MV_DONE {eff}:", json.dumps({"run": res, "acc": round(s["accuracy"], 4), "trunc": s["truncated"], "cot": round(s["reasoning_tokens_mean"] or 0, 1)}), flush=True)

    blank_summary = None
    if not args.skip_blank:
        b_cfg = {**mv_cfg, "reasoning_effort": "low", "loop_num_passes": 3, "image_mode": "blank", "concurrency": args.batch_size}
        b_jsonl = out_dir / "mini_mathvista83_blank_low_loop3.jsonl"
        run(b_cfg, mv_ds, b_jsonl, backend=backend, batch_size=args.batch_size, progress_callback=_make_progress_printer(f"mini mv83 blank bs={args.batch_size}"))
        b_rows = _regrade_mathvista_rows(load_rows(b_jsonl), meta_mv)
        blank_summary = summarize(b_rows)
        print("MINI_BLANK_DONE:", json.dumps({"acc": round(blank_summary["accuracy"], 4), "correct": blank_summary["correct"], "n": blank_summary["n"]}), flush=True)

    cqa_summary = None
    if not args.skip_chartqa and cqa_ds is not None:
        c_cfg = {**cqa_cfg, "reasoning_effort": "medium", "loop_num_passes": 3, "image_mode": "correct", "concurrency": args.batch_size}
        c_jsonl = out_dir / "mini_chartqa80_medium_loop3.jsonl"
        run(c_cfg, cqa_ds, c_jsonl, backend=backend, batch_size=args.batch_size, progress_callback=_make_progress_printer(f"mini chartqa80 bs={args.batch_size}"))
        c_rows = load_rows(c_jsonl)
        cqa_summary = summarize_chartqa_rows(c_rows)
        print("MINI_CHARTQA_DONE:", json.dumps({"relaxed": round(cqa_summary["relaxed_accuracy"], 4), "pct_norm": round(cqa_summary["pct_norm_accuracy"], 4), "trunc": cqa_summary["truncated"]}), flush=True)

    cache_stats = get_model_cache_stats()
    payload = {
        "model_id": mv_cfg["model_id"],
        "model_revision": mv_cfg["model_revision"],
        "batch_size": args.batch_size,
        "environment": env_info,
        "model_cache_stats": cache_stats,
        "mathvista_mini": mv_summaries,
        "mathvista_blank_mini": blank_summary,
        "chartqa_mini": cqa_summary,
    }
    atomic_write_text(out_dir / "phase8a_mini_summary.json", json.dumps(payload, indent=2) + "\n")
    print("PHASE8A_MINI_DONE", flush=True)


if __name__ == "__main__":
    main()
