"""Phase 7B: Adversarial Real-World Hard Reasoning Holdout (`MathVista-Hard-83`).

Tests whether the synthetic Phase 5A finding (`effort="low" + loop=3` beats both
`low + loop=1` and `medium + loop=1/3`) actually transfers to genuinely hard
real-world free-form numeric math/geometry/logic problems WITHOUT synthetic
paired-question `?` markers or perception-level saturation.

Default 2x2 matrix: `reasoning_effort` in {low, medium} x `loop_num_passes` in {1, 3}
= 4 x 83 = 332 evaluations under the exact locked Phase 2-5 protocol.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from loopbench.backends import create_backend
from loopbench.backends.transformers_backend import get_model_cache_stats
from loopbench.config import load_config
from loopbench.data.mathvista import (
    build_mathvista_hard_holdout,
    grade_mathvista_numeric,
    validate_mathvista_holdout,
)
from loopbench.metrics import (
    cluster_bootstrap_accuracy_ci,
    paired_cluster_bootstrap_diff_ci,
    summarize,
)
from loopbench.oracle import analyze_offline_oracle
from loopbench.report import MOCK_STAMP
from loopbench.runner import run
from loopbench.storage import atomic_write_text, load_rows
from run_phase3_smoke48 import _make_progress_printer, collect_env_fingerprint, write_warmup_subset


def _regrade_mathvista_rows(rows: list[dict[str, Any]], meta_by_qid: dict[str, dict[str, Any]], *, relaxed_rel_tol: float | None = None) -> list[dict[str, Any]]:
    regraded: list[dict[str, Any]] = []
    for r in rows:
        is_trunc_or_err = (
            r.get("finish_reason") in ("length", "error")
            or r.get("outcome") in ("truncated", "backend_error", "parse_failure")
        )
        m = meta_by_qid.get(r["question_id"], {})
        ok = (not is_trunc_or_err) and grade_mathvista_numeric(
            r.get("predicted"),
            float(r["expected"]),
            answer_type=str(m.get("answer_type", "integer")),
            precision=m.get("precision"),
            relaxed_rel_tol=relaxed_rel_tol,
        )
        outcome = r.get("outcome") if is_trunc_or_err else ("correct" if ok else "wrong_answer")
        regraded.append({**r, "correct": ok, "outcome": outcome, "source": m.get("source", "unknown")})
    return regraded


def build_mathvista_report(
    cell_jsonl_map: dict[tuple[str, int], Path | str],
    questions_json: Path,
    out_dir: Path,
    *,
    env_info: dict[str, Any] | None = None,
    cache_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(questions_json.read_text(encoding="utf-8"))
    meta_by_qid = {q["question_id"]: q.get("metadata", {}) for q in manifest["questions"]}

    effort_order = ["none", "low", "medium", "high"]
    efforts = sorted({e for e, _ in cell_jsonl_map}, key=lambda e: effort_order.index(e) if e in effort_order else 99)
    passes = sorted({int(p) for _, p in cell_jsonl_map})

    cell_summaries: dict[str, dict[str, Any]] = {}
    regraded_strict_map: dict[str, list[dict[str, Any]]] = {}
    any_mock = False

    for eff in efforts:
        for p in passes:
            if (eff, p) not in cell_jsonl_map:
                continue
            key = f"{eff}_loop{p}"
            raw_rows = load_rows(cell_jsonl_map[(eff, p)])
            strict_rows = _regrade_mathvista_rows(raw_rows, meta_by_qid, relaxed_rel_tol=None)
            rel1_rows = _regrade_mathvista_rows(raw_rows, meta_by_qid, relaxed_rel_tol=0.01)
            regraded_strict_map[key] = strict_rows
            summ = summarize(strict_rows)
            any_mock = any_mock or bool(summ["mock"])

            by_src: dict[str, list[bool]] = {}
            for r in strict_rows:
                by_src.setdefault(str(r.get("source", "unknown")), []).append(bool(r["correct"]))
            summ["by_source"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_src.items())}
            summ["by_source_counts"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_src.items())}
            summ["relaxed_1pct_accuracy"] = sum(r["correct"] for r in rel1_rows) / max(1, len(rel1_rows))
            summ["relaxed_1pct_ci95"] = cluster_bootstrap_accuracy_ci(rel1_rows, n_boot=2000)
            cell_summaries[key] = summ

    # Paired contrasts
    paired: dict[str, dict[str, Any]] = {}
    for eff in efforts:
        avail = [p for p in passes if f"{eff}_loop{p}" in regraded_strict_map]
        if len(avail) >= 2:
            b = avail[0]
            for p in avail[1:]:
                paired[f"{eff}_loop{p}_minus_{eff}_loop{b}"] = paired_cluster_bootstrap_diff_ci(
                    regraded_strict_map[f"{eff}_loop{b}"], regraded_strict_map[f"{eff}_loop{p}"], n_boot=5000
                )
    for p in passes:
        avail_e = [e for e in efforts if f"{e}_loop{p}" in regraded_strict_map]
        if len(avail_e) >= 2:
            be = avail_e[0]
            for e in avail_e[1:]:
                paired[f"{e}_loop{p}_minus_{be}_loop{p}"] = paired_cluster_bootstrap_diff_ci(
                    regraded_strict_map[f"{be}_loop{p}"], regraded_strict_map[f"{e}_loop{p}"], n_boot=5000
                )
    if "low_loop3" in regraded_strict_map and "medium_loop1" in regraded_strict_map:
        paired["low_loop3_minus_medium_loop1"] = paired_cluster_bootstrap_diff_ci(
            regraded_strict_map["medium_loop1"], regraded_strict_map["low_loop3"], n_boot=5000
        )
    if "low_loop3" in regraded_strict_map and "medium_loop3" in regraded_strict_map:
        paired["medium_loop3_minus_low_loop3"] = paired_cluster_bootstrap_diff_ci(
            regraded_strict_map["low_loop3"], regraded_strict_map["medium_loop3"], n_boot=5000
        )

    oracle = analyze_offline_oracle(regraded_strict_map) if len(regraded_strict_map) >= 2 else None

    payload = {
        "label": MOCK_STAMP if any_mock else "REAL MODEL EVALUATION — PHASE 7B MATHVISTA-HARD-83",
        "mock": any_mock,
        "source_dataset": manifest.get("source_dataset"),
        "source_revision": manifest.get("source_revision"),
        "question_count": manifest.get("question_count"),
        "by_source": manifest.get("by_source"),
        "environment": env_info,
        "model_cache_stats": cache_stats,
        "cell_summaries": cell_summaries,
        "paired_contrasts": paired,
        "offline_oracle": oracle,
    }
    atomic_write_text(out_dir / "mathvista_summary.json", json.dumps(payload, indent=2) + "\n")

    buf = io.StringIO()
    cw = csv.writer(buf, lineterminator="\n")
    cw.writerow([
        "config", "effort", "loop_num_passes", "effective_layers", "n",
        "accuracy", "ci95_lo", "ci95_hi", "relaxed_1pct_acc",
        "correct", "wrong", "parse_failures", "truncated",
        "cot_mean_toks", "ttft_p50_ms", "tpot_p50_ms", "e2e_p50_ms", "e2e_mean_ms", "peak_vram_max_mb",
    ])
    for eff in efforts:
        for p in passes:
            key = f"{eff}_loop{p}"
            if key not in cell_summaries:
                continue
            s = cell_summaries[key]
            ci = s["accuracy_cluster_bootstrap_95ci"]
            cw.writerow([
                key, eff, p, 24 + 12 * (p - 1), s["n"],
                round(s["accuracy"] or 0.0, 6), round(ci[0] or 0.0, 6), round(ci[1] or 0.0, 6),
                round(s["relaxed_1pct_accuracy"], 6),
                s["correct"], s["wrong_answers"], s["parse_failures"], s["truncated"],
                round(s["reasoning_tokens_mean"] or 0.0, 2),
                round(s["ttft_p50_ms"] or 0.0, 2),
                round(s["tpot_p50_ms"] or 0.0, 2),
                round(s["e2e_p50_ms"] or 0.0, 2),
                round(s["e2e_mean_ms"] or 0.0, 2),
                round(s["peak_vram_max_mb"] or 0.0, 2),
            ])
    atomic_write_text(out_dir / "mathvista_grid.csv", buf.getvalue(), encoding="utf-8-sig")

    plot_name = "pareto_mathvista_hard_mock.png" if any_mock else "pareto_mathvista_hard.png"
    _render_mathvista_figure(efforts, passes, cell_summaries, oracle, any_mock, out_dir / plot_name)
    return payload


def _render_mathvista_figure(
    efforts: list[str],
    passes: list[int],
    cell_summaries: dict[str, dict[str, Any]],
    oracle: dict[str, Any] | None,
    any_mock: bool,
    out_path: Path,
) -> None:
    palette = {"none": "#2E75B6", "low": "#70AD47", "medium": "#ED7D31", "high": "#C00000"}
    markers = {"none": "o", "low": "s", "medium": "^", "high": "D"}
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 9.8}):
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=150)
        ax_p, ax_s = axes[0], axes[1]

        all_y: list[float] = []
        for eff in efforts:
            xs, ys, ylo, yhi = [], [], [], []
            for p in passes:
                k = f"{eff}_loop{p}"
                if k not in cell_summaries:
                    continue
                s = cell_summaries[k]
                x = (s["e2e_mean_ms"] or 0.0) / 1000.0
                y = (s["accuracy"] or 0.0) * 100.0
                ci = s["accuracy_cluster_bootstrap_95ci"]
                xs.append(x)
                ys.append(y)
                all_y.append(y)
                ylo.append(max(0.0, y - (ci[0] or 0.0) * 100.0))
                yhi.append(max(0.0, (ci[1] or 0.0) * 100.0 - y))
                ax_p.annotate(f"{eff}-L{p}", (x, y), textcoords="offset points", xytext=(6, 7), fontsize=8.3, fontweight="bold")
            if xs:
                ax_p.errorbar(xs, ys, yerr=[ylo, yhi], fmt=f"-{markers.get(eff, 'o')}", color=palette.get(eff, "#333"), linewidth=2.0, markersize=7, capsize=4, label=f"effort={eff}")
        if oracle and oracle.get("router_oracle"):
            ro = oracle["router_oracle"]
            ax_p.scatter([ro["e2e_mean_ms"] / 1000.0], [ro["accuracy"] * 100.0], marker="*", s=180, color="#7030A0", zorder=6, label="Offline Router Oracle")
            all_y.append(ro["accuracy"] * 100.0)
        ax_p.set_xlabel("Mean E2E Latency (s)")
        ax_p.set_ylabel("Exact Numeric Accuracy (%) [95% CI]")
        ax_p.set_ylim(max(0.0, min(all_y) - 14.0) if all_y else 0, min(105.0, max(all_y) + 12.0) if all_y else 100)
        ax_p.margins(x=0.18)
        ax_p.grid(True, alpha=0.3)
        ax_p.legend(loc="lower right", fontsize=8.3, frameon=True)
        ax_p.set_title("A. MathVista-Hard-83 Accuracy vs. Mean Latency", fontweight="bold")

        # Panel B: Per-source accuracy comparison across configs
        cfg_keys = [f"{e}_loop{p}" for e in efforts for p in passes if f"{e}_loop{p}" in cell_summaries]
        sources = sorted({src for k in cfg_keys for src in cell_summaries[k].get("by_source", {})})
        if sources and cfg_keys:
            import numpy as np
            x_idx = np.arange(len(sources))
            w = 0.8 / len(cfg_keys)
            bar_cols = ["#70AD47", "#385723", "#ED7D31", "#C00000", "#5B9BD5", "#7030A0"]
            for i_c, ck in enumerate(cfg_keys):
                vals = [cell_summaries[ck]["by_source"].get(src, 0.0) * 100.0 for src in sources]
                off = (i_c - (len(cfg_keys) - 1) / 2.0) * w
                ax_s.bar(x_idx + off, vals, width=w * 0.92, color=bar_cols[i_c % len(bar_cols)], edgecolor="#222", linewidth=0.6, label=ck)
            ax_s.set_xticks(x_idx, sources, rotation=15, ha="right")
            ax_s.set_ylabel("Exact Accuracy (%)")
            ax_s.set_ylim(0, 114)
            ax_s.grid(axis="y", alpha=0.3)
            ax_s.legend(loc="upper right", ncol=2, fontsize=7.8, frameon=True)
            ax_s.set_title("B. Breakdown Across 6 Real-World Hard Sources", fontweight="bold")

        fig.suptitle(MOCK_STAMP if any_mock else "LoopBench Phase 7B: Adversarial Real-World Hard Math Holdout (MathVista-Hard-83)", fontweight="bold", fontsize=11.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out_path, metadata={"Software": "LoopBench"})
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase 7B Real-World MathVista-Hard-83 Holdout")
    ap.add_argument("--config", default=str(ROOT / "configs/phase7_mathvista.yaml"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/phase7b_mathvista"))
    ap.add_argument("--dataset-dir", default=None, help="Optional prebuilt dataset directory (skips parquet download)")
    ap.add_argument("--parquet", default=None, help="Optional local MathVista testmini parquet path")
    ap.add_argument("--efforts", default="low,medium")
    ap.add_argument("--passes", default="1,3")
    ap.add_argument("--image-mode", default="correct", choices=["correct", "blank", "mismatched"])
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    ds_dir = Path(args.dataset_dir) if args.dataset_dir else (out_dir / "dataset_mathvista_hard83")
    if not (ds_dir / "questions.json").is_file():
        build_mathvista_hard_holdout(ds_dir, parquet_path=args.parquet)
    valid = validate_mathvista_holdout(ds_dir, expected_questions=None)
    print("MATHVISTA_DATASET_VALID:", json.dumps(valid), flush=True)

    efforts = [x.strip() for x in args.efforts.split(",") if x.strip()]
    passes = [int(x.strip()) for x in args.passes.split(",") if x.strip()]

    env_path = out_dir / "env_fingerprint.json"
    if not args.report_only and not env_path.is_file():
        env_info = collect_env_fingerprint(cfg)
        atomic_write_text(env_path, json.dumps(env_info, indent=2) + "\n")
    elif env_path.is_file():
        env_info = json.loads(env_path.read_text(encoding="utf-8"))
    else:
        env_info = None

    cell_map: dict[tuple[str, int], Path] = {}
    if not args.report_only:
        backend = create_backend(cfg)
        if not args.no_warmup and cfg["backend"] != "mock":
            w_ds = write_warmup_subset(ds_dir, out_dir / "subset_warmup1")
            run({**cfg, "loop_num_passes": 1, "reasoning_effort": "low", "max_new_tokens": 16}, w_ds, out_dir / "_warmup.jsonl", backend=backend)

        for eff in efforts:
            for p in passes:
                cell_cfg = {**cfg, "reasoning_effort": eff, "loop_num_passes": p, "image_mode": args.image_mode}
                mode_prefix = "" if args.image_mode == "correct" else f"{args.image_mode}_"
                jsonl_p = out_dir / f"mathvista83_{mode_prefix}{eff}_loop{p}.jsonl"
                res = run(cell_cfg, ds_dir, jsonl_p, backend=backend, progress_callback=_make_progress_printer(f"mathvista83 {mode_prefix}{eff} loop={p}"))
                rows = load_rows(jsonl_p)
                if cfg["backend"] == "transformers":
                    exp_l = 24 + 12 * (p - 1)
                    bad = [r["question_id"] for r in rows if r.get("effective_layers_verified") != exp_l]
                    if bad:
                        raise RuntimeError(f"Layer verification failed for {eff} loop={p}: {bad[:5]}")
                cell_map[(eff, p)] = jsonl_p
                summ = summarize(rows)
                print(
                    f"MATHVISTA_CELL_DONE {eff} loop={p}:",
                    json.dumps({"run": res, "raw_acc": summ["accuracy"], "trunc": summ["truncated"], "e2e_mean_s": round((summ["e2e_mean_ms"] or 0) / 1000.0, 2)}),
                    flush=True,
                )

        cache_stats = get_model_cache_stats()
        if cfg["backend"] == "transformers" and cache_stats["loads"] > 1:
            raise RuntimeError(f"Zero-reload violated: {cache_stats}")
        atomic_write_text(out_dir / "model_cache_stats.json", json.dumps(cache_stats, indent=2) + "\n")
    else:
        cache_stats = json.loads((out_dir / "model_cache_stats.json").read_text(encoding="utf-8")) if (out_dir / "model_cache_stats.json").is_file() else None
        for eff in efforts:
            for p in passes:
                cand = out_dir / f"mathvista83_{eff}_loop{p}.jsonl"
                if cand.is_file():
                    cell_map[(eff, p)] = cand

    build_mathvista_report(
        cell_jsonl_map=cell_map,
        questions_json=ds_dir / "questions.json",
        out_dir=out_dir / "report",
        env_info=env_info,
        cache_stats=cache_stats,
    )
    print("PHASE7B_MATHVISTA_REPORT_READY", flush=True)


if __name__ == "__main__":
    main()
