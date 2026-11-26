import csv
import io
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .data.holdout import grade_chartqa_relaxed
from .metrics import (
    cluster_bootstrap_accuracy_ci,
    evaluate_synth320_extension_rule,
    infer_difficulty,
    paired_cluster_bootstrap_diff_ci,
    summarize,
)
from .oracle import analyze_offline_oracle
from .storage import atomic_write_text, load_rows

MOCK_STAMP = "MOCK / TEST DATA — NOT REAL MODEL PERFORMANCE"


def build_report(jsonl_path, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(jsonl_path)
    summary = summarize(rows)
    is_mock = bool(summary["mock"])
    summary["label"] = MOCK_STAMP if is_mock else "REAL MODEL EVALUATION"

    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf, lineterminator="\n")
    writer.writerow(["metric", "value"])
    for k, v in summary.items():
        writer.writerow([k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v])
    atomic_write_text(out / "summary.csv", csv_buf.getvalue(), encoding="utf-8-sig")

    banner = f"> {MOCK_STAMP}\n\n" if is_mock else ""
    title = "# LoopBench mock report" if is_mock else "# LoopBench evaluation report"
    md_lines = [
        title,
        "",
        banner.rstrip() if banner else "> Real model evaluation output",
        "",
        "## Summary metrics",
        "",
        f"- Questions: {summary['n']}",
        f"- Accuracy: {summary['accuracy']}",
        f"- Cluster bootstrap 95% CI: {summary['accuracy_cluster_bootstrap_95ci']}",
        f"- Correct: {summary['correct']}",
        f"- Wrong answers: {summary['wrong_answers']}",
        f"- Parse failures: {summary['parse_failures']}",
        f"- Truncated (`finish_reason=length`): {summary['truncated']}",
        f"- Backend errors: {summary['backend_errors']}",
        f"- E2E P50: {summary['e2e_p50_ms']} ms",
        f"- E2E P90: {summary['e2e_p90_ms']} ms",
        "",
        "## Breakdown",
        "",
        f"- By modality: `{json.dumps(summary['by_modality'])}`",
        f"- By task type: `{json.dumps(summary['by_task_type'])}`",
        "",
    ]
    atomic_write_text(out / "report.md", "\n".join(md_lines))

    plot_name = "pareto_mock.png" if is_mock else "pareto.png"
    with matplotlib.rc_context({"font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=128)
        ax.scatter([summary["e2e_p50_ms"] or 0.0], [summary["accuracy"] or 0.0], s=60, color="#1F4E79")
        ax.set_xlabel("E2E P50 (ms)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(MOCK_STAMP if is_mock else "LoopBench Accuracy vs Latency")
        fig.tight_layout()
        fig.savefig(out / plot_name, metadata={"Software": "LoopBench"})
        plt.close(fig)

    return summary


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(den) < 1e-12:
        return None
    return round(num / den, 4)


def build_phase3_report(
    pass_jsonl_map: dict[int, Path | str],
    leakage_jsonl_map: dict[str, Path | str] | None,
    out_dir: Path | str,
    env_info: dict | None = None,
    cache_stats: dict | None = None,
    leakage_passes: int = 3,
) -> dict:
    """Build multi-pass Phase 3 smoke-48 causal scaling & visual leakage report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    passes_sorted = sorted(int(k) for k in pass_jsonl_map)
    pass_rows: dict[int, list[dict]] = {}
    pass_summaries: dict[str, dict] = {}
    any_mock = False

    for p in passes_sorted:
        rows = load_rows(pass_jsonl_map[p])
        pass_rows[p] = rows
        summ = summarize(rows)
        any_mock = any_mock or bool(summ["mock"])
        pass_summaries[str(p)] = summ

    base_p = passes_sorted[0]
    base_summ = pass_summaries[str(base_p)]
    scaling_vs_base: dict[str, dict] = {}
    for p in passes_sorted:
        s = pass_summaries[str(p)]
        vram_delta = (
            round((s["peak_vram_max_mb"] or 0.0) - (base_summ["peak_vram_max_mb"] or 0.0), 2)
            if s["peak_vram_max_mb"] is not None and base_summ["peak_vram_max_mb"] is not None
            else None
        )
        scaling_vs_base[str(p)] = {
            "effective_layers": s["effective_layers_unique"][0]
            if len(s["effective_layers_unique"]) == 1
            else (24 + 12 * (p - 1)),
            "ttft_p50_ratio": _safe_ratio(s["ttft_p50_ms"], base_summ["ttft_p50_ms"]),
            "tpot_p50_ratio": _safe_ratio(s["tpot_p50_ms"], base_summ["tpot_p50_ms"]),
            "e2e_p50_ratio": _safe_ratio(s["e2e_p50_ms"], base_summ["e2e_p50_ms"]),
            "peak_vram_delta_mb": vram_delta,
        }

    paired_diffs: dict[str, dict] = {}
    for p in passes_sorted:
        if p == base_p:
            continue
        paired_diffs[f"loop{p}_minus_loop{base_p}"] = paired_cluster_bootstrap_diff_ci(
            pass_rows[base_p], pass_rows[p]
        )
    if 3 in pass_rows and 5 in pass_rows:
        paired_diffs["loop5_minus_loop3"] = paired_cluster_bootstrap_diff_ci(
            pass_rows[3], pass_rows[5]
        )
    if 2 in pass_rows and 3 in pass_rows:
        paired_diffs["loop3_minus_loop2"] = paired_cluster_bootstrap_diff_ci(
            pass_rows[2], pass_rows[3]
        )

    leakage_summaries: dict[str, dict] = {}
    if leakage_passes in pass_rows:
        leakage_summaries["correct"] = pass_summaries[str(leakage_passes)]
    for mode, lpath in sorted((leakage_jsonl_map or {}).items()):
        if Path(lpath).exists():
            lrows = load_rows(lpath)
            lsumm = summarize(lrows)
            any_mock = any_mock or bool(lsumm["mock"])
            leakage_summaries[mode] = lsumm

    leakage_deltas: dict[str, float | None] = {}
    if "correct" in leakage_summaries:
        c_acc = leakage_summaries["correct"]["accuracy"]
        for mode in ("blank", "mismatched"):
            if mode in leakage_summaries and c_acc is not None and leakage_summaries[mode]["accuracy"] is not None:
                leakage_deltas[f"correct_minus_{mode}"] = round(
                    c_acc - leakage_summaries[mode]["accuracy"], 6
                )

    payload = {
        "label": MOCK_STAMP if any_mock else "REAL MODEL EVALUATION — PHASE 3 SMOKE-48",
        "mock": any_mock,
        "passes_evaluated": passes_sorted,
        "leakage_reference_passes": leakage_passes,
        "environment": env_info,
        "model_cache_stats": cache_stats,
        "pass_summaries": pass_summaries,
        "scaling_vs_loop1": scaling_vs_base,
        "paired_accuracy_diffs": paired_diffs,
        "leakage_controls": leakage_summaries,
        "leakage_accuracy_drops": leakage_deltas,
    }
    atomic_write_text(out / "phase3_summary.json", json.dumps(payload, indent=2) + "\n")

    # Pass-curve CSV
    pass_csv = io.StringIO()
    pw = csv.writer(pass_csv, lineterminator="\n")
    pw.writerow([
        "loop_num_passes",
        "effective_layers",
        "n",
        "accuracy",
        "ci95_low",
        "ci95_high",
        "correct",
        "wrong_answers",
        "parse_failures",
        "truncated",
        "backend_errors",
        "chart_acc",
        "geometry_acc",
        "direct_acc",
        "multi_step_acc",
        "chart_direct_acc",
        "chart_multi_step_acc",
        "geometry_direct_acc",
        "geometry_multi_step_acc",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "tpot_p50_ms",
        "e2e_p50_ms",
        "e2e_p90_ms",
        "peak_vram_max_mb",
        "reasoning_tokens_mean",
        "answer_tokens_mean",
    ])
    for p in passes_sorted:
        s = pass_summaries[str(p)]
        sc = scaling_vs_base[str(p)]
        ci = s["accuracy_cluster_bootstrap_95ci"]
        bmt = s["by_modality_task"]
        pw.writerow([
            p,
            sc["effective_layers"],
            s["n"],
            s["accuracy"],
            ci[0],
            ci[1],
            s["correct"],
            s["wrong_answers"],
            s["parse_failures"],
            s["truncated"],
            s["backend_errors"],
            s["by_modality"].get("chart"),
            s["by_modality"].get("geometry"),
            s["by_task_type"].get("direct"),
            s["by_task_type"].get("multi_step"),
            bmt.get("chart/direct"),
            bmt.get("chart/multi_step"),
            bmt.get("geometry/direct"),
            bmt.get("geometry/multi_step"),
            s["ttft_p50_ms"],
            s["ttft_p90_ms"],
            s["tpot_p50_ms"],
            s["e2e_p50_ms"],
            s["e2e_p90_ms"],
            s["peak_vram_max_mb"],
            s["reasoning_tokens_mean"],
            s["answer_tokens_mean"],
        ])
    atomic_write_text(out / "phase3_pass_curve.csv", pass_csv.getvalue(), encoding="utf-8-sig")

    # Leakage-controls CSV
    leak_csv = io.StringIO()
    lw = csv.writer(leak_csv, lineterminator="\n")
    lw.writerow([
        "image_mode",
        "loop_num_passes",
        "n",
        "accuracy",
        "ci95_low",
        "ci95_high",
        "correct",
        "wrong_answers",
        "parse_failures",
        "truncated",
        "backend_errors",
        "e2e_p50_ms",
    ])
    for mode in ("correct", "blank", "mismatched"):
        if mode not in leakage_summaries:
            continue
        ls = leakage_summaries[mode]
        lci = ls["accuracy_cluster_bootstrap_95ci"]
        lw.writerow([
            mode,
            leakage_passes,
            ls["n"],
            ls["accuracy"],
            lci[0],
            lci[1],
            ls["correct"],
            ls["wrong_answers"],
            ls["parse_failures"],
            ls["truncated"],
            ls["backend_errors"],
            ls["e2e_p50_ms"],
        ])
    atomic_write_text(out / "phase3_leakage_controls.csv", leak_csv.getvalue(), encoding="utf-8-sig")

    # Multi-panel Pareto & Leakage Figure
    plot_name = "pareto_pass_scaling_mock.png" if any_mock else "pareto_pass_scaling.png"
    _render_phase3_figure(
        passes_sorted=passes_sorted,
        pass_summaries=pass_summaries,
        scaling_vs_base=scaling_vs_base,
        leakage_summaries=leakage_summaries,
        leakage_passes=leakage_passes,
        any_mock=any_mock,
        out_path=out / plot_name,
    )

    # Markdown quick summary inside artifact directory
    md = [
        f"# {MOCK_STAMP}" if any_mock else "# Phase 3 Smoke-48 Evaluation Summary",
        "",
        "## Pass-Depth Scaling (`image_mode=correct`)",
        "",
        "| Passes | Eff. Layers | Acc (`n`) | 95% Cluster CI | Chart | Geom | Direct | Multi-step | TTFT P50 (ms) | TPOT P50 (ms) | E2E P50 (ms) | Peak VRAM (MB) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for p in passes_sorted:
        s = pass_summaries[str(p)]
        sc = scaling_vs_base[str(p)]
        ci = s["accuracy_cluster_bootstrap_95ci"]
        acc_str = f"{(s['accuracy'] or 0)*100:.1f}% ({s['correct']}/{s['n']})"
        ci_str = f"[{(ci[0] or 0)*100:.1f}%, {(ci[1] or 0)*100:.1f}%]"
        md.append(
            f"| {p} | {sc['effective_layers']} | {acc_str} | {ci_str} | "
            f"{(s['by_modality'].get('chart') or 0)*100:.1f}% | "
            f"{(s['by_modality'].get('geometry') or 0)*100:.1f}% | "
            f"{(s['by_task_type'].get('direct') or 0)*100:.1f}% | "
            f"{(s['by_task_type'].get('multi_step') or 0)*100:.1f}% | "
            f"{(s['ttft_p50_ms'] or 0):.1f} | {(s['tpot_p50_ms'] or 0):.2f} | "
            f"{(s['e2e_p50_ms'] or 0):.1f} | {(s['peak_vram_max_mb'] or 0):.1f} |"
        )
    if leakage_summaries:
        md.extend([
            "",
            f"## Visual Leakage Controls (`loop_num_passes={leakage_passes}`)",
            "",
            "| Image Mode | Acc (`n`) | Correct | Wrong | Parse Fail | Truncated | Backend Err |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for mode in ("correct", "blank", "mismatched"):
            if mode not in leakage_summaries:
                continue
            ls = leakage_summaries[mode]
            md.append(
                f"| `{mode}` | {(ls['accuracy'] or 0)*100:.1f}% ({ls['correct']}/{ls['n']}) | "
                f"{ls['correct']} | {ls['wrong_answers']} | {ls['parse_failures']} | "
                f"{ls['truncated']} | {ls['backend_errors']} |"
            )
    atomic_write_text(out / "report.md", "\n".join(md) + "\n")
    return payload


def _render_phase3_figure(
    passes_sorted: list[int],
    pass_summaries: dict[str, dict],
    scaling_vs_base: dict[str, dict],
    leakage_summaries: dict[str, dict],
    leakage_passes: int,
    any_mock: bool,
    out_path: Path,
) -> None:
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), dpi=140)
        ax_pareto, ax_sub, ax_lat, ax_leak = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        # Panel A: Accuracy vs E2E P50 Latency Pareto Curve with 95% Cluster CI
        e2e_s = [(pass_summaries[str(p)]["e2e_p50_ms"] or 0.0) / 1000.0 for p in passes_sorted]
        accs = [(pass_summaries[str(p)]["accuracy"] or 0.0) * 100.0 for p in passes_sorted]
        ci_lo = [
            max(0.0, accs[i] - ((pass_summaries[str(p)]["accuracy_cluster_bootstrap_95ci"][0] or 0.0) * 100.0))
            for i, p in enumerate(passes_sorted)
        ]
        ci_hi = [
            max(0.0, ((pass_summaries[str(p)]["accuracy_cluster_bootstrap_95ci"][1] or 0.0) * 100.0) - accs[i])
            for i, p in enumerate(passes_sorted)
        ]
        ax_pareto.plot(e2e_s, accs, color="#1F4E79", linewidth=2.0, linestyle="-", zorder=2)
        ax_pareto.errorbar(
            e2e_s,
            accs,
            yerr=[ci_lo, ci_hi],
            fmt="o",
            markersize=8,
            color="#C00000",
            ecolor="#4472C4",
            elinewidth=1.6,
            capsize=4,
            zorder=3,
        )
        for idx_pt, (x_val, y_val, p) in enumerate(zip(e2e_s, accs, passes_sorted)):
            layers = scaling_vs_base[str(p)]["effective_layers"]
            dx = -14 if idx_pt == len(passes_sorted) - 1 else 6
            dy = -26 if y_val > 85 else 10
            ax_pareto.annotate(
                f"loop={p}\n({layers}L)",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8.5,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.82},
                zorder=4,
            )
        if len(e2e_s) >= 2:
            span = max(1.0, max(e2e_s) - min(e2e_s))
            ax_pareto.set_xlim(min(e2e_s) - 0.10 * span, max(e2e_s) + 0.16 * span)
        ax_pareto.set_xlabel("Single-Stream E2E Latency P50 (s)")
        ax_pareto.set_ylabel("Accuracy (%) [with 95% Cluster CI]")
        ax_pareto.set_ylim(-5, 108)
        ax_pareto.grid(True, alpha=0.3)
        ax_pareto.set_title("A. Accuracy–Latency Pareto Curve (smoke-48)")

        # Panel B: Modality x Task Type Breakdown across Pass Depths
        series_specs = [
            ("chart/direct", "Chart - Direct", "#1F4E79", "o", "-", 2.4, 8.0),
            ("chart/multi_step", "Chart - Multi-step", "#5B9BD5", "s", "--", 1.9, 6.5),
            ("geometry/direct", "Geometry - Direct", "#C55A11", "^", "-", 2.0, 7.0),
            ("geometry/multi_step", "Geometry - Multi-step", "#70AD47", "D", ":", 1.9, 5.5),
        ]
        for key, label, color, marker, ls_style, lw_val, ms_val in series_specs:
            ys = [
                (pass_summaries[str(p)]["by_modality_task"].get(key, 0.0) or 0.0) * 100.0
                for p in passes_sorted
            ]
            ax_sub.plot(
                passes_sorted,
                ys,
                label=label,
                color=color,
                marker=marker,
                linestyle=ls_style,
                linewidth=lw_val,
                markersize=ms_val,
            )
        ax_sub.set_xticks(passes_sorted, [f"loop={p}\n({scaling_vs_base[str(p)]['effective_layers']}L)" for p in passes_sorted])
        ax_sub.set_ylabel("Subgroup Accuracy (%)")
        ax_sub.set_ylim(-5, 108)
        ax_sub.grid(True, alpha=0.3)
        ax_sub.legend(loc="lower right", fontsize=8.5, frameon=True)
        ax_sub.set_title("B. Modality × Task-Type Pass Scaling")

        # Panel C: Observable TTFT, Decode TPOT & VRAM Scaling Ratio vs loop=1
        ttft_r = [scaling_vs_base[str(p)]["ttft_p50_ratio"] or 1.0 for p in passes_sorted]
        tpot_r = [scaling_vs_base[str(p)]["tpot_p50_ratio"] or 1.0 for p in passes_sorted]
        e2e_r = [scaling_vs_base[str(p)]["e2e_p50_ratio"] or 1.0 for p in passes_sorted]
        layer_r = [
            scaling_vs_base[str(p)]["effective_layers"] / max(1, scaling_vs_base[str(passes_sorted[0])]["effective_layers"])
            for p in passes_sorted
        ]
        ax_lat.plot(passes_sorted, layer_r, "--", color="#7F7F7F", linewidth=1.5, label="Effective Layer Ratio")
        ax_lat.plot(passes_sorted, ttft_r, "-o", color="#4472C4", linewidth=2.0, label="Observable TTFT P50 Ratio")
        ax_lat.plot(passes_sorted, tpot_r, "-s", color="#ED7D31", linewidth=2.0, label="Decode TPOT P50 Ratio")
        ax_lat.plot(passes_sorted, e2e_r, "-^", color="#7030A0", linewidth=2.0, label="E2E Latency P50 Ratio")
        ax_lat.set_xticks(passes_sorted, [f"loop={p}" for p in passes_sorted])
        ax_lat.set_ylabel("Scaling Ratio vs loop=1 (×)")
        ax_lat.grid(True, alpha=0.3)
        ax_lat.legend(loc="upper left", fontsize=8.5, frameon=True)
        ax_lat.set_title("C. Latency Scaling vs Effective Pass Depth")

        # Panel D: Visual Leakage Control Gate (Stacked 5-way Outcome Accounting)
        modes = [m for m in ("correct", "blank", "mismatched") if m in leakage_summaries]
        if modes:
            x = np.arange(len(modes))
            outcome_specs = [
                ("correct", "Correct", "#70AD47"),
                ("wrong_answers", "Wrong Answer", "#ED7D31"),
                ("parse_failures", "Parse Failure / Refusal", "#5B9BD5"),
                ("truncated", "Truncated (length)", "#FFC000"),
                ("backend_errors", "Backend Error", "#C00000"),
            ]
            bottom = np.zeros(len(modes))
            for field, lbl, col in outcome_specs:
                vals = np.array([
                    100.0 * leakage_summaries[m][field] / max(1, leakage_summaries[m]["n"])
                    for m in modes
                ])
                ax_leak.bar(x, vals, width=0.52, bottom=bottom, label=lbl, color=col, edgecolor="#222222", linewidth=0.7)
                bottom += vals
            ax_leak.set_xticks(x, [f"{m}\n(loop={leakage_passes})" for m in modes])
            ax_leak.set_ylabel("Share of Questions (%)")
            ax_leak.set_ylim(0, 112)
            ax_leak.grid(axis="y", alpha=0.3)
            ax_leak.legend(loc="upper right", fontsize=8.0, frameon=True)
        ax_leak.set_title(f"D. Visual Leakage Control Gate (loop={leakage_passes})")

        suptitle = (
            MOCK_STAMP
            if any_mock
            else "LoopBench Phase 3 (smoke-48): Causal Pass-Depth Scaling & Visual Leakage Controls"
        )
        fig.suptitle(suptitle, fontsize=12.5, fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(out_path, metadata={"Software": "LoopBench"})
        plt.close(fig)


def _compute_paired_suite(pass_rows: dict[int, list[dict]]) -> dict[str, dict]:
    passes_sorted = sorted(pass_rows)
    if not passes_sorted:
        return {}
    base_p = passes_sorted[0]
    diffs: dict[str, dict] = {}
    for p in passes_sorted:
        if p != base_p:
            diffs[f"loop{p}_minus_loop{base_p}"] = paired_cluster_bootstrap_diff_ci(
                pass_rows[base_p], pass_rows[p]
            )
    if 2 in pass_rows and 3 in pass_rows:
        diffs["loop3_minus_loop2"] = paired_cluster_bootstrap_diff_ci(pass_rows[2], pass_rows[3])
    if 3 in pass_rows and 5 in pass_rows:
        diffs["loop5_minus_loop3"] = paired_cluster_bootstrap_diff_ci(pass_rows[3], pass_rows[5])
    return diffs


def build_phase4_report(
    pass_jsonl_map: dict[int, Path | str],
    out_dir: Path | str,
    scale_tag: str = "synth240",
    env_info: dict | None = None,
    cache_stats: dict | None = None,
    initial_synth240_decision: dict | None = None,
    max_ci_half_width: float = 0.05,
    ambiguous_min_abs_diff: float = 0.025,
) -> dict:
    """Build Phase 4 main synthetic study (synth-240 / synth-320) causal pass-depth report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    passes_sorted = sorted(int(k) for k in pass_jsonl_map)
    pass_rows: dict[int, list[dict]] = {}
    pass_summaries: dict[str, dict] = {}
    any_mock = False

    for p in passes_sorted:
        rows = load_rows(pass_jsonl_map[p])
        pass_rows[p] = rows
        summ = summarize(rows)
        any_mock = any_mock or bool(summ["mock"])
        pass_summaries[str(p)] = summ

    base_p = passes_sorted[0]
    base_summ = pass_summaries[str(base_p)]
    scaling_vs_base: dict[str, dict] = {}
    for p in passes_sorted:
        s = pass_summaries[str(p)]
        vram_delta = (
            round((s["peak_vram_max_mb"] or 0.0) - (base_summ["peak_vram_max_mb"] or 0.0), 2)
            if s["peak_vram_max_mb"] is not None and base_summ["peak_vram_max_mb"] is not None
            else None
        )
        scaling_vs_base[str(p)] = {
            "effective_layers": s["effective_layers_unique"][0]
            if len(s["effective_layers_unique"]) == 1
            else (24 + 12 * (p - 1)),
            "ttft_p50_ratio": _safe_ratio(s["ttft_p50_ms"], base_summ["ttft_p50_ms"]),
            "tpot_p50_ratio": _safe_ratio(s["tpot_p50_ms"], base_summ["tpot_p50_ms"]),
            "e2e_p50_ratio": _safe_ratio(s["e2e_p50_ms"], base_summ["e2e_p50_ms"]),
            "reasoning_tokens_mean_ratio": _safe_ratio(
                s["reasoning_tokens_mean"], base_summ["reasoning_tokens_mean"]
            ),
            "peak_vram_delta_mb": vram_delta,
        }

    paired_diffs = _compute_paired_suite(pass_rows)

    paired_by_difficulty: dict[str, dict] = {}
    for diff_tier in ("standard", "hard"):
        sub_rows = {
            p: [r for r in pass_rows[p] if infer_difficulty(r) == diff_tier]
            for p in passes_sorted
        }
        if any(sub_rows.values()):
            paired_by_difficulty[diff_tier] = _compute_paired_suite(sub_rows)

    paired_by_mod_task: dict[str, dict] = {}
    for cell in ("chart/direct", "chart/multi_step", "geometry/direct", "geometry/multi_step"):
        mod_name, task_name = cell.split("/")
        sub_rows = {
            p: [
                r
                for r in pass_rows[p]
                if r.get("modality") == mod_name and r.get("task_type") == task_name
            ]
            for p in passes_sorted
        }
        if any(sub_rows.values()):
            paired_by_mod_task[cell] = _compute_paired_suite(sub_rows)

    current_decision = evaluate_synth320_extension_rule(
        paired_diffs,
        max_ci_half_width=max_ci_half_width,
        ambiguous_min_abs_diff=ambiguous_min_abs_diff,
    )

    payload = {
        "label": MOCK_STAMP if any_mock else f"REAL MODEL EVALUATION — PHASE 4 {scale_tag.upper()}",
        "mock": any_mock,
        "dataset_scale": scale_tag,
        "passes_evaluated": passes_sorted,
        "environment": env_info,
        "model_cache_stats": cache_stats,
        "pass_summaries": pass_summaries,
        "scaling_vs_loop1": scaling_vs_base,
        "paired_accuracy_diffs": paired_diffs,
        "paired_accuracy_diffs_by_difficulty": paired_by_difficulty,
        "paired_accuracy_diffs_by_modality_task": paired_by_mod_task,
        "uncertainty_extension_decision": current_decision,
        "initial_synth240_extension_decision": initial_synth240_decision or current_decision,
    }
    atomic_write_text(out / "phase4_summary.json", json.dumps(payload, indent=2) + "\n")

    # 1. Overall pass-curve CSV
    pass_csv = io.StringIO()
    pw = csv.writer(pass_csv, lineterminator="\n")
    pw.writerow([
        "dataset_scale",
        "loop_num_passes",
        "effective_layers",
        "n",
        "accuracy",
        "ci95_low",
        "ci95_high",
        "correct",
        "wrong_answers",
        "parse_failures",
        "truncated",
        "backend_errors",
        "standard_acc",
        "hard_acc",
        "chart_acc",
        "geometry_acc",
        "direct_acc",
        "multi_step_acc",
        "chart_direct_acc",
        "chart_multi_step_acc",
        "geometry_direct_acc",
        "geometry_multi_step_acc",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "tpot_p50_ms",
        "e2e_p50_ms",
        "e2e_p90_ms",
        "peak_vram_max_mb",
        "reasoning_tokens_mean",
        "reasoning_tokens_p50",
        "answer_tokens_mean",
    ])
    for p in passes_sorted:
        s = pass_summaries[str(p)]
        sc = scaling_vs_base[str(p)]
        ci = s["accuracy_cluster_bootstrap_95ci"]
        bmt = s["by_modality_task"]
        bdiff = s["by_difficulty"]
        pw.writerow([
            scale_tag,
            p,
            sc["effective_layers"],
            s["n"],
            s["accuracy"],
            ci[0],
            ci[1],
            s["correct"],
            s["wrong_answers"],
            s["parse_failures"],
            s["truncated"],
            s["backend_errors"],
            bdiff.get("standard"),
            bdiff.get("hard"),
            s["by_modality"].get("chart"),
            s["by_modality"].get("geometry"),
            s["by_task_type"].get("direct"),
            s["by_task_type"].get("multi_step"),
            bmt.get("chart/direct"),
            bmt.get("chart/multi_step"),
            bmt.get("geometry/direct"),
            bmt.get("geometry/multi_step"),
            s["ttft_p50_ms"],
            s["ttft_p90_ms"],
            s["tpot_p50_ms"],
            s["e2e_p50_ms"],
            s["e2e_p90_ms"],
            s["peak_vram_max_mb"],
            s["reasoning_tokens_mean"],
            s["reasoning_tokens_p50"],
            s["answer_tokens_mean"],
        ])
    atomic_write_text(out / "phase4_pass_curve.csv", pass_csv.getvalue(), encoding="utf-8-sig")

    # 2. Difficulty x Modality x Task breakdown CSV
    sub_csv = io.StringIO()
    sw = csv.writer(sub_csv, lineterminator="\n")
    sw.writerow([
        "loop_num_passes",
        "standard_direct_acc",
        "standard_multi_step_acc",
        "hard_direct_acc",
        "hard_multi_step_acc",
        "chart_standard_acc",
        "chart_hard_acc",
        "geometry_standard_acc",
        "geometry_hard_acc",
    ])
    for p in passes_sorted:
        s = pass_summaries[str(p)]
        bdt = s["by_difficulty_task"]
        bmd = s["by_modality_difficulty"]
        sw.writerow([
            p,
            bdt.get("standard/direct"),
            bdt.get("standard/multi_step"),
            bdt.get("hard/direct"),
            bdt.get("hard/multi_step"),
            bmd.get("chart/standard"),
            bmd.get("chart/hard"),
            bmd.get("geometry/standard"),
            bmd.get("geometry/hard"),
        ])
    atomic_write_text(out / "phase4_difficulty_breakdown.csv", sub_csv.getvalue(), encoding="utf-8-sig")

    plot_name = f"pareto_{scale_tag}_mock.png" if any_mock else f"pareto_{scale_tag}.png"
    _render_phase4_figure(
        passes_sorted=passes_sorted,
        pass_summaries=pass_summaries,
        scaling_vs_base=scaling_vs_base,
        scale_tag=scale_tag,
        any_mock=any_mock,
        out_path=out / plot_name,
    )

    md = [
        f"# {MOCK_STAMP}" if any_mock else f"# Phase 4 Main Synthetic Study ({scale_tag})",
        "",
        "## Causal Pass-Depth Curve (`image_mode=correct`, `effort=medium`)",
        "",
        "| Passes | Eff. Layers | Acc (`n`) | 95% Cluster CI | Standard (`v1`) | Hard (`v2`) | Chart | Geom | TTFT P50 (ms) | TPOT P50 (ms) | E2E P50 (ms) | Peak VRAM (MB) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for p in passes_sorted:
        s = pass_summaries[str(p)]
        sc = scaling_vs_base[str(p)]
        ci = s["accuracy_cluster_bootstrap_95ci"]
        acc_str = f"{(s['accuracy'] or 0)*100:.1f}% ({s['correct']}/{s['n']})"
        ci_str = f"[{(ci[0] or 0)*100:.1f}%, {(ci[1] or 0)*100:.1f}%]"
        md.append(
            f"| {p} | {sc['effective_layers']} | {acc_str} | {ci_str} | "
            f"{(s['by_difficulty'].get('standard') or 0)*100:.1f}% | "
            f"{(s['by_difficulty'].get('hard') or 0)*100:.1f}% | "
            f"{(s['by_modality'].get('chart') or 0)*100:.1f}% | "
            f"{(s['by_modality'].get('geometry') or 0)*100:.1f}% | "
            f"{(s['ttft_p50_ms'] or 0):.1f} | {(s['tpot_p50_ms'] or 0):.2f} | "
            f"{(s['e2e_p50_ms'] or 0):.1f} | {(s['peak_vram_max_mb'] or 0):.1f} |"
        )

    md.extend([
        "",
        "## Paired `image_id` Cluster Bootstrap Differences",
        "",
        "| Comparison | Point Δ Acc | 95% Cluster Bootstrap CI |",
        "| :--- | :---: | :---: |",
    ])
    for k, d in paired_diffs.items():
        dci = d["diff_cluster_bootstrap_95ci"]
        md.append(
            f"| `{k}` | {(d['diff_accuracy'] or 0)*100:+.2f}% | "
            f"[{(dci[0] or 0)*100:+.2f}%, {(dci[1] or 0)*100:+.2f}%] |"
        )

    md.extend([
        "",
        "## Predeclared `synth-320` Uncertainty Rule Decision",
        "",
        f"- `trigger_synth320`: `{current_decision['trigger_synth320']}`",
        f"- `reasons`: `{json.dumps(current_decision['reasons'])}`",
    ])
    atomic_write_text(out / "report.md", "\n".join(md) + "\n")
    return payload


def _render_phase4_figure(
    passes_sorted: list[int],
    pass_summaries: dict[str, dict],
    scaling_vs_base: dict[str, dict],
    scale_tag: str,
    any_mock: bool,
    out_path: Path,
) -> None:
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6), dpi=140)
        ax_pareto, ax_diff, ax_mod, ax_lat = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        # Panel A: Accuracy vs E2E P50 Latency Pareto Curve with 95% Cluster CI
        e2e_s = [(pass_summaries[str(p)]["e2e_p50_ms"] or 0.0) / 1000.0 for p in passes_sorted]
        accs = [(pass_summaries[str(p)]["accuracy"] or 0.0) * 100.0 for p in passes_sorted]
        ci_lo = [
            max(0.0, accs[i] - ((pass_summaries[str(p)]["accuracy_cluster_bootstrap_95ci"][0] or 0.0) * 100.0))
            for i, p in enumerate(passes_sorted)
        ]
        ci_hi = [
            max(0.0, ((pass_summaries[str(p)]["accuracy_cluster_bootstrap_95ci"][1] or 0.0) * 100.0) - accs[i])
            for i, p in enumerate(passes_sorted)
        ]
        ax_pareto.plot(e2e_s, accs, color="#1F4E79", linewidth=2.1, linestyle="-", zorder=2)
        ax_pareto.errorbar(
            e2e_s,
            accs,
            yerr=[ci_lo, ci_hi],
            fmt="o",
            markersize=8,
            color="#C00000",
            ecolor="#4472C4",
            elinewidth=1.7,
            capsize=4,
            zorder=3,
        )
        for idx_pt, (x_val, y_val, p) in enumerate(zip(e2e_s, accs, passes_sorted)):
            layers = scaling_vs_base[str(p)]["effective_layers"]
            dx = -16 if idx_pt == len(passes_sorted) - 1 else 7
            dy = -26 if y_val > 82 else 10
            ax_pareto.annotate(
                f"loop={p}\n({layers}L)",
                (x_val, y_val),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8.5,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.85},
                zorder=4,
            )
        if len(e2e_s) >= 2:
            span = max(1.0, max(e2e_s) - min(e2e_s))
            ax_pareto.set_xlim(min(e2e_s) - 0.12 * span, max(e2e_s) + 0.18 * span)
        ax_pareto.set_xlabel("Single-Stream E2E Latency P50 (s)")
        ax_pareto.set_ylabel("Overall Accuracy (%) [with 95% Cluster CI]")
        ax_pareto.set_ylim(-5, 108)
        ax_pareto.grid(True, alpha=0.3)
        ax_pareto.set_title(f"A. Accuracy–Latency Pareto Curve ({scale_tag})")

        # Panel B: Difficulty Tier x Task Type Pass Scaling
        diff_specs = [
            ("standard/direct", "Standard (v1) - Direct", "#1F4E79", "o", "-", 2.2, 7.5),
            ("standard/multi_step", "Standard (v1) - Multi-step", "#5B9BD5", "s", "--", 2.0, 6.8),
            ("hard/direct", "Hard (v2) - Direct", "#C55A11", "^", "-", 2.2, 7.5),
            ("hard/multi_step", "Hard (v2) - Multi-step", "#C00000", "D", "-.", 2.1, 6.8),
        ]
        for key, label, color, marker, ls_style, lw_val, ms_val in diff_specs:
            ys = [
                (pass_summaries[str(p)]["by_difficulty_task"].get(key, 0.0) or 0.0) * 100.0
                for p in passes_sorted
            ]
            ax_diff.plot(
                passes_sorted,
                ys,
                label=label,
                color=color,
                marker=marker,
                linestyle=ls_style,
                linewidth=lw_val,
                markersize=ms_val,
            )
        ax_diff.set_xticks(passes_sorted, [f"loop={p}\n({scaling_vs_base[str(p)]['effective_layers']}L)" for p in passes_sorted])
        ax_diff.set_ylabel("Subgroup Accuracy (%)")
        ax_diff.set_ylim(-5, 108)
        ax_diff.grid(True, alpha=0.3)
        ax_diff.legend(loc="lower right", fontsize=8.5, frameon=True)
        ax_diff.set_title("B. Difficulty Tier × Task-Type Pass Scaling")

        # Panel C: Modality x Difficulty Tier Pass Scaling
        mod_specs = [
            ("chart/standard", "Chart - Standard (v1)", "#2F5597", "o", "-", 2.2, 7.5),
            ("chart/hard", "Chart - Hard (v2)", "#3D85C6", "s", "--", 2.0, 6.8),
            ("geometry/standard", "Geometry - Standard (v1)", "#548235", "^", "-", 2.2, 7.5),
            ("geometry/hard", "Geometry - Hard (v2)", "#ED7D31", "D", "-.", 2.1, 6.8),
        ]
        for key, label, color, marker, ls_style, lw_val, ms_val in mod_specs:
            ys = [
                (pass_summaries[str(p)]["by_modality_difficulty"].get(key, 0.0) or 0.0) * 100.0
                for p in passes_sorted
            ]
            ax_mod.plot(
                passes_sorted,
                ys,
                label=label,
                color=color,
                marker=marker,
                linestyle=ls_style,
                linewidth=lw_val,
                markersize=ms_val,
            )
        ax_mod.set_xticks(passes_sorted, [f"loop={p}\n({scaling_vs_base[str(p)]['effective_layers']}L)" for p in passes_sorted])
        ax_mod.set_ylabel("Subgroup Accuracy (%)")
        ax_mod.set_ylim(-5, 108)
        ax_mod.grid(True, alpha=0.3)
        ax_mod.legend(loc="lower right", fontsize=8.5, frameon=True)
        ax_mod.set_title("C. Modality × Difficulty Tier Pass Scaling")

        # Panel D: Latency & Reasoning Token Ratios vs loop=1
        ttft_r = [scaling_vs_base[str(p)]["ttft_p50_ratio"] or 1.0 for p in passes_sorted]
        tpot_r = [scaling_vs_base[str(p)]["tpot_p50_ratio"] or 1.0 for p in passes_sorted]
        e2e_r = [scaling_vs_base[str(p)]["e2e_p50_ratio"] or 1.0 for p in passes_sorted]
        rtok_r = [scaling_vs_base[str(p)]["reasoning_tokens_mean_ratio"] or 1.0 for p in passes_sorted]
        layer_r = [
            scaling_vs_base[str(p)]["effective_layers"] / max(1, scaling_vs_base[str(passes_sorted[0])]["effective_layers"])
            for p in passes_sorted
        ]
        ax_lat.plot(passes_sorted, layer_r, "--", color="#7F7F7F", linewidth=1.5, label="Effective Layer Ratio")
        ax_lat.plot(passes_sorted, ttft_r, "-o", color="#4472C4", linewidth=2.0, label="Observable TTFT P50 Ratio")
        ax_lat.plot(passes_sorted, tpot_r, "-s", color="#ED7D31", linewidth=2.0, label="Decode TPOT P50 Ratio")
        ax_lat.plot(passes_sorted, e2e_r, "-^", color="#7030A0", linewidth=2.0, label="E2E Latency P50 Ratio")
        ax_lat.plot(passes_sorted, rtok_r, ":D", color="#548235", linewidth=1.8, label="Mean Reasoning Tokens Ratio")
        ax_lat.set_xticks(passes_sorted, [f"loop={p}" for p in passes_sorted])
        ax_lat.set_ylabel("Scaling Ratio vs loop=1 (×)")
        ax_lat.grid(True, alpha=0.3)
        ax_lat.legend(loc="upper left", fontsize=8.3, frameon=True)
        ax_lat.set_title("D. Latency & CoT Length Scaling vs Pass Depth")

        suptitle = (
            MOCK_STAMP
            if any_mock
            else f"LoopBench Phase 4 ({scale_tag}): Main Synthetic Pass-Depth Study (effort=medium)"
        )
        fig.suptitle(suptitle, fontsize=12.5, fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(out_path, metadata={"Software": "LoopBench"})
        plt.close(fig)


def _grade_holdout_pct_norm_relaxed(predicted: float | None, expected: float) -> bool:
    if grade_chartqa_relaxed(predicted, expected, rel_tol=0.05, abs_tol=1e-2):
        return True
    if predicted is None:
        return False
    exp = float(expected)
    pred = float(predicted)
    if 0.0 < abs(exp) <= 1.0 < abs(pred) <= 100.0:
        return grade_chartqa_relaxed(pred / 100.0, exp, rel_tol=0.05, abs_tol=1e-3)
    if 0.0 < abs(pred) <= 1.0 < abs(exp) <= 100.0:
        return grade_chartqa_relaxed(pred * 100.0, exp, rel_tol=0.05, abs_tol=1e-2)
    return False


def build_phase5_reports(
    effort_jsonl_map: dict[tuple[str, int], Path | str],
    holdout_jsonl_map: dict[int, Path | str],
    out_dir: Path | str,
    effort_tag: str = "effort48",
    holdout_tag: str = "chartqa80",
    env_info: dict | None = None,
    cache_stats: dict | None = None,
) -> dict:
    """Build strictly separated Phase 5 reports:
    1. `effort_matrix/`: Reasoning Effort × Pass-Depth Matrix on balanced synthetic subset.
    2. `holdout_<tag>/`: One-source real-world ChartQA holdout pass-depth scaling.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    if effort_jsonl_map:
        result["effort_matrix"] = _build_effort_matrix_report(
            effort_jsonl_map=effort_jsonl_map,
            out_dir=out / "effort_matrix",
            effort_tag=effort_tag,
            env_info=env_info,
            cache_stats=cache_stats,
        )
    if holdout_jsonl_map:
        result["holdout"] = _build_holdout_report(
            holdout_jsonl_map=holdout_jsonl_map,
            out_dir=out / f"holdout_{holdout_tag}",
            holdout_tag=holdout_tag,
            env_info=env_info,
            cache_stats=cache_stats,
        )
    return result


def _build_effort_matrix_report(
    effort_jsonl_map: dict[tuple[str, int], Path | str],
    out_dir: Path,
    effort_tag: str,
    env_info: dict | None,
    cache_stats: dict | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    effort_order_pref = ["none", "low", "medium", "high"]
    present_efforts = sorted(
        {eff for eff, _ in effort_jsonl_map},
        key=lambda e: effort_order_pref.index(e) if e in effort_order_pref else 99,
    )
    present_passes = sorted({int(p) for _, p in effort_jsonl_map})

    cell_rows: dict[str, list[dict]] = {}
    cell_summaries: dict[str, dict] = {}
    any_mock = False

    for eff in present_efforts:
        for p in present_passes:
            if (eff, p) not in effort_jsonl_map:
                continue
            key = f"{eff}_loop{p}"
            rows = load_rows(effort_jsonl_map[(eff, p)])
            cell_rows[key] = rows
            summ = summarize(rows)
            any_mock = any_mock or bool(summ["mock"])
            cell_summaries[key] = summ

    # Paired pass-depth gains within each fixed reasoning_effort
    within_effort_pass_diffs: dict[str, dict] = {}
    for eff in present_efforts:
        avail_p = [p for p in present_passes if f"{eff}_loop{p}" in cell_rows]
        if len(avail_p) < 2:
            continue
        base_p = avail_p[0]
        within_effort_pass_diffs[eff] = {}
        for p in avail_p[1:]:
            within_effort_pass_diffs[eff][f"loop{p}_minus_loop{base_p}"] = paired_cluster_bootstrap_diff_ci(
                cell_rows[f"{eff}_loop{base_p}"], cell_rows[f"{eff}_loop{p}"]
            )

    # Paired effort gains within each fixed loop_num_passes
    within_pass_effort_diffs: dict[str, dict] = {}
    for p in present_passes:
        avail_e = [e for e in present_efforts if f"{e}_loop{p}" in cell_rows]
        if len(avail_e) < 2:
            continue
        base_e = avail_e[0]
        within_pass_effort_diffs[str(p)] = {}
        for e in avail_e[1:]:
            within_pass_effort_diffs[str(p)][f"{e}_minus_{base_e}"] = paired_cluster_bootstrap_diff_ci(
                cell_rows[f"{base_e}_loop{p}"], cell_rows[f"{e}_loop{p}"]
            )

    oracle_analysis = analyze_offline_oracle(cell_rows) if len(cell_rows) >= 2 else None

    payload = {
        "label": MOCK_STAMP if any_mock else f"REAL MODEL EVALUATION — PHASE 5 EFFORT MATRIX ({effort_tag.upper()})",
        "mock": any_mock,
        "dataset_scale": effort_tag,
        "efforts_evaluated": present_efforts,
        "passes_evaluated": present_passes,
        "environment": env_info,
        "model_cache_stats": cache_stats,
        "cell_summaries": cell_summaries,
        "within_effort_pass_diffs": within_effort_pass_diffs,
        "within_pass_effort_diffs": within_pass_effort_diffs,
        "offline_oracle": oracle_analysis,
    }
    atomic_write_text(out_dir / "effort_summary.json", json.dumps(payload, indent=2) + "\n")

    csv_buf = io.StringIO()
    cw = csv.writer(csv_buf, lineterminator="\n")
    cw.writerow([
        "reasoning_effort",
        "loop_num_passes",
        "effective_layers",
        "n",
        "accuracy",
        "ci95_low",
        "ci95_high",
        "correct",
        "wrong_answers",
        "parse_failures",
        "truncated",
        "standard_acc",
        "hard_acc",
        "direct_acc",
        "multi_step_acc",
        "geometry_direct_acc",
        "geometry_multi_step_acc",
        "ttft_p50_ms",
        "tpot_p50_ms",
        "e2e_p50_ms",
        "reasoning_tokens_mean",
        "answer_tokens_mean",
        "peak_vram_max_mb",
    ])
    for eff in present_efforts:
        for p in present_passes:
            key = f"{eff}_loop{p}"
            if key not in cell_summaries:
                continue
            s = cell_summaries[key]
            ci = s["accuracy_cluster_bootstrap_95ci"]
            eff_layers = s["effective_layers_unique"][0] if len(s["effective_layers_unique"]) == 1 else (24 + 12 * (p - 1))
            cw.writerow([
                eff,
                p,
                eff_layers,
                s["n"],
                s["accuracy"],
                ci[0],
                ci[1],
                s["correct"],
                s["wrong_answers"],
                s["parse_failures"],
                s["truncated"],
                s["by_difficulty"].get("standard"),
                s["by_difficulty"].get("hard"),
                s["by_task_type"].get("direct"),
                s["by_task_type"].get("multi_step"),
                s["by_modality_task"].get("geometry/direct"),
                s["by_modality_task"].get("geometry/multi_step"),
                s["ttft_p50_ms"],
                s["tpot_p50_ms"],
                s["e2e_p50_ms"],
                s["reasoning_tokens_mean"],
                s["answer_tokens_mean"],
                s["peak_vram_max_mb"],
            ])
    atomic_write_text(out_dir / "effort_grid.csv", csv_buf.getvalue(), encoding="utf-8-sig")

    plot_name = "pareto_effort_matrix_mock.png" if any_mock else "pareto_effort_matrix.png"
    _render_effort_matrix_figure(
        efforts=present_efforts,
        passes=present_passes,
        cell_summaries=cell_summaries,
        oracle_analysis=oracle_analysis,
        effort_tag=effort_tag,
        any_mock=any_mock,
        out_path=out_dir / plot_name,
    )

    md = [
        f"# {MOCK_STAMP}" if any_mock else f"# Phase 5A: Reasoning Effort × Pass-Depth Matrix ({effort_tag})",
        "",
        "| Effort | Passes | Layers | Acc (`n`) | 95% CI | Standard | Hard | Direct | Multi-step | Truncated | Mean CoT Toks | E2E P50 (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for eff in present_efforts:
        for p in present_passes:
            key = f"{eff}_loop{p}"
            if key not in cell_summaries:
                continue
            s = cell_summaries[key]
            ci = s["accuracy_cluster_bootstrap_95ci"]
            md.append(
                f"| `{eff}` | {p} | {24 + 12*(p-1)} | {(s['accuracy'] or 0)*100:.1f}% ({s['correct']}/{s['n']}) | "
                f"[{(ci[0] or 0)*100:.1f}%, {(ci[1] or 0)*100:.1f}%] | "
                f"{(s['by_difficulty'].get('standard') or 0)*100:.1f}% | "
                f"{(s['by_difficulty'].get('hard') or 0)*100:.1f}% | "
                f"{(s['by_task_type'].get('direct') or 0)*100:.1f}% | "
                f"{(s['by_task_type'].get('multi_step') or 0)*100:.1f}% | "
                f"{s['truncated']} | {(s['reasoning_tokens_mean'] or 0):.1f} | {(s['e2e_p50_ms'] or 0)/1000:.2f} |"
            )
    atomic_write_text(out_dir / "report.md", "\n".join(md) + "\n")
    return payload


def _render_effort_matrix_figure(
    efforts: list[str],
    passes: list[int],
    cell_summaries: dict[str, dict],
    oracle_analysis: dict | None,
    effort_tag: str,
    any_mock: bool,
    out_path: Path,
) -> None:
    palette = {"none": "#2E75B6", "low": "#70AD47", "medium": "#ED7D31", "high": "#C00000"}
    markers = {"none": "o", "low": "s", "medium": "^", "high": "D"}

    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6), dpi=140)
        ax_pareto, ax_pass, ax_tok, ax_task = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        # Panel A: Joint Accuracy vs Latency Pareto Space
        eff_label_offsets = {
            ("none", 1): (6, -12),
            ("none", 3): (6, 6),
            ("none", 5): (6, -12),
            ("low", 1): (6, -12),
            ("low", 3): (6, 6),
            ("low", 5): (6, 6),
            ("medium", 1): (6, 6),
            ("medium", 3): (-36, 7),
            ("medium", 5): (6, 6),
            ("high", 1): (6, -12),
            ("high", 3): (6, -12),
            ("high", 5): (-38, 7),
        }
        all_pareto_ys: list[float] = []
        for eff in efforts:
            xs, ys = [], []
            for p in passes:
                key = f"{eff}_loop{p}"
                if key not in cell_summaries:
                    continue
                s = cell_summaries[key]
                x_s = (s["e2e_p50_ms"] or 0.0) / 1000.0
                y_a = (s["accuracy"] or 0.0) * 100.0
                xs.append(x_s)
                ys.append(y_a)
                all_pareto_ys.append(y_a)
                dx, dy = eff_label_offsets.get((eff, p), (5, 5))
                ax_pareto.annotate(
                    f"{eff[:3]}-L{p}",
                    (x_s, y_a),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=7.8,
                )
            if xs:
                ax_pareto.plot(
                    xs, ys,
                    marker=markers.get(eff, "o"),
                    color=palette.get(eff, "#333333"),
                    linewidth=1.9,
                    markersize=7,
                    label=f"effort={eff}",
                )
        if oracle_analysis and oracle_analysis.get("router_oracle"):
            ro = oracle_analysis["router_oracle"]
            ro_y = (ro["accuracy"] or 0.0) * 100.0
            all_pareto_ys.append(ro_y)
            ax_pareto.scatter(
                [(ro["e2e_p50_ms"] or 0.0) / 1000.0],
                [ro_y],
                marker="*",
                s=170,
                color="#7030A0",
                zorder=5,
                label="Offline Router Oracle",
            )
        ax_pareto.set_xlabel("Single-Stream E2E Latency P50 (s)")
        ax_pareto.set_ylabel("Accuracy (%)")
        y_lo_a = max(0.0, min(all_pareto_ys) - 10.0) if (all_pareto_ys and min(all_pareto_ys) > 55.0) else -5.0
        ax_pareto.set_ylim(y_lo_a, 104.0)
        ax_pareto.margins(x=0.12)
        ax_pareto.grid(True, alpha=0.3)
        ax_pareto.legend(loc="lower right", fontsize=8.3, frameon=True)
        ax_pareto.set_title(f"A. Joint Effort × Pass Pareto Trade-off ({effort_tag})")

        # Panel B: Pass-Depth Accuracy Curves Stratified by Reasoning Effort
        all_pass_ys: list[float] = []
        for eff in efforts:
            ps_avail = [p for p in passes if f"{eff}_loop{p}" in cell_summaries]
            ys = [(cell_summaries[f"{eff}_loop{p}"]["accuracy"] or 0.0) * 100.0 for p in ps_avail]
            all_pass_ys.extend(ys)
            ax_pass.plot(
                ps_avail, ys,
                marker=markers.get(eff, "o"),
                color=palette.get(eff, "#333333"),
                linewidth=2.1,
                markersize=7,
                label=f"effort={eff}",
            )
        ax_pass.set_xticks(passes, [f"loop={p}\n({24+12*(p-1)}L)" for p in passes])
        ax_pass.set_ylabel("Overall Accuracy (%)")
        y_lo_b = max(0.0, min(all_pass_ys) - 10.0) if (all_pass_ys and min(all_pass_ys) > 55.0) else -5.0
        ax_pass.set_ylim(y_lo_b, 104.0)
        ax_pass.grid(True, alpha=0.3)
        ax_pass.legend(loc="lower right", fontsize=8.5, frameon=True)
        ax_pass.set_title("B. Latent Pass Scaling Across Explicit Effort Levels")

        # Panel C: Mean Explicit CoT Reasoning Tokens Across Effort x Pass
        x_idx = np.arange(len(efforts))
        width = 0.8 / max(1, len(passes))
        p_colors = ["#5B9BD5", "#ED7D31", "#7030A0", "#70AD47"]
        max_tok_val = 0.0
        for idx_p, p in enumerate(passes):
            vals = [
                float(cell_summaries.get(f"{eff}_loop{p}", {}).get("reasoning_tokens_mean") or 0.0)
                for eff in efforts
            ]
            if vals:
                max_tok_val = max(max_tok_val, max(vals))
            offset = (idx_p - (len(passes) - 1) / 2.0) * width
            ax_tok.bar(
                x_idx + offset, vals, width=width * 0.92,
                color=p_colors[idx_p % len(p_colors)],
                edgecolor="#222222", linewidth=0.7,
                label=f"loop={p}",
            )
        ax_tok.set_xticks(x_idx, [f"effort={e}" for e in efforts])
        ax_tok.set_ylabel("Mean Explicit Reasoning Tokens")
        if max_tok_val > 0:
            ax_tok.set_ylim(0, max_tok_val * 1.22)
        ax_tok.grid(axis="y", alpha=0.3)
        ax_tok.legend(loc="upper left", ncol=len(passes), fontsize=8.3, frameon=True)
        ax_tok.set_title("C. Explicit CoT Token Budget Consumption")

        # Panel D: Multi-step vs Direct Accuracy under Suppressed vs Full CoT
        for eff in efforts:
            ps_avail = [p for p in passes if f"{eff}_loop{p}" in cell_summaries]
            ms_ys = [
                (cell_summaries[f"{eff}_loop{p}"]["by_task_type"].get("multi_step") or 0.0) * 100.0
                for p in ps_avail
            ]
            ax_task.plot(
                ps_avail, ms_ys,
                marker=markers.get(eff, "o"),
                color=palette.get(eff, "#333333"),
                linestyle="-",
                linewidth=2.0,
                label=f"{eff} (multi_step)",
            )
        ax_task.set_xticks(passes, [f"loop={p}" for p in passes])
        ax_task.set_ylabel("Multi-Step Accuracy (%)")
        ax_task.set_ylim(-5, 108)
        ax_task.grid(True, alpha=0.3)
        ax_task.legend(loc="lower right", fontsize=8.3, frameon=True)
        ax_task.set_title("D. Multi-Step Deduction Scaling by Effort")

        suptitle = (
            MOCK_STAMP
            if any_mock
            else f"LoopBench Phase 5A ({effort_tag}): Latent Loop Depth × Explicit Reasoning Effort Matrix"
        )
        fig.suptitle(suptitle, fontsize=12.5, fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(out_path, metadata={"Software": "LoopBench"})
        plt.close(fig)


def _build_holdout_report(
    holdout_jsonl_map: dict[int, Path | str],
    out_dir: Path,
    holdout_tag: str,
    env_info: dict | None,
    cache_stats: dict | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    passes_sorted = sorted(int(k) for k in holdout_jsonl_map)
    pass_rows: dict[int, list[dict]] = {}
    relaxed_rows_map: dict[int, list[dict]] = {}
    pct_norm_rows_map: dict[int, list[dict]] = {}
    pass_summaries: dict[str, dict] = {}
    any_mock = False

    for p in passes_sorted:
        rows = load_rows(holdout_jsonl_map[p])
        pass_rows[p] = rows
        summ = summarize(rows)
        any_mock = any_mock or bool(summ["mock"])

        strict_0p01_rows = []
        rel_rows = []
        pct_rows = []
        vtoks = sorted(int(r["visual_tokens"]) for r in rows if r.get("visual_tokens") is not None)

        for r in rows:
            is_trunc_or_err = r.get("finish_reason") in ("length", "error") or r.get("outcome") in ("truncated", "backend_error", "parse_failure")
            pred = r.get("predicted")
            exp = float(r["expected"])
            ok_01 = (not is_trunc_or_err) and (pred is not None) and (abs(float(pred) - exp) <= 1e-2)
            ok_rel = (not is_trunc_or_err) and grade_chartqa_relaxed(pred, exp, rel_tol=0.05, abs_tol=1e-2)
            ok_pct = (not is_trunc_or_err) and _grade_holdout_pct_norm_relaxed(pred, exp)

            strict_0p01_rows.append({**r, "correct": ok_01, "outcome": "correct" if ok_01 else "wrong_answer"})
            rel_rows.append({**r, "correct": ok_rel, "outcome": "correct" if ok_rel else "wrong_answer"})
            pct_rows.append({**r, "correct": ok_pct, "outcome": "correct" if ok_pct else "wrong_answer"})

        relaxed_rows_map[p] = rel_rows
        pct_norm_rows_map[p] = pct_rows
        n_q = max(1, len(rows))
        summ["strict_tol_0p01_accuracy"] = sum(x["correct"] for x in strict_0p01_rows) / n_q
        summ["strict_tol_0p01_ci95"] = cluster_bootstrap_accuracy_ci(strict_0p01_rows)
        summ["relaxed_5pct_accuracy"] = sum(x["correct"] for x in rel_rows) / n_q
        summ["relaxed_5pct_correct"] = sum(x["correct"] for x in rel_rows)
        summ["relaxed_5pct_ci95"] = cluster_bootstrap_accuracy_ci(rel_rows)
        summ["relaxed_pct_norm_5pct_accuracy"] = sum(x["correct"] for x in pct_rows) / n_q
        summ["relaxed_pct_norm_5pct_ci95"] = cluster_bootstrap_accuracy_ci(pct_rows)
        summ["visual_tokens_min"] = vtoks[0] if vtoks else None
        summ["visual_tokens_p50"] = vtoks[len(vtoks) // 2] if vtoks else None
        summ["visual_tokens_max"] = vtoks[-1] if vtoks else None
        pass_summaries[str(p)] = summ

    paired_relaxed_diffs = _compute_paired_suite(relaxed_rows_map)
    paired_strict_diffs = _compute_paired_suite(pass_rows)
    oracle_relaxed = (
        analyze_offline_oracle({f"loop{p}": relaxed_rows_map[p] for p in passes_sorted})
        if len(passes_sorted) >= 2
        else None
    )

    payload = {
        "label": MOCK_STAMP if any_mock else f"REAL MODEL EVALUATION — PHASE 5 REAL-WORLD HOLDOUT ({holdout_tag.upper()})",
        "mock": any_mock,
        "holdout_dataset": "HuggingFaceM4/ChartQA (test split, human numeric paired clusters)",
        "source_revision": "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5",
        "dataset_scale": holdout_tag,
        "passes_evaluated": passes_sorted,
        "environment": env_info,
        "model_cache_stats": cache_stats,
        "pass_summaries": pass_summaries,
        "paired_relaxed_5pct_diffs": paired_relaxed_diffs,
        "paired_exact_diffs": paired_strict_diffs,
        "offline_oracle_relaxed_5pct": oracle_relaxed,
    }
    atomic_write_text(out_dir / "holdout_summary.json", json.dumps(payload, indent=2) + "\n")

    csv_buf = io.StringIO()
    hw = csv.writer(csv_buf, lineterminator="\n")
    hw.writerow([
        "loop_num_passes",
        "effective_layers",
        "n",
        "exact_accuracy",
        "strict_tol_0p01_accuracy",
        "relaxed_5pct_accuracy",
        "relaxed_5pct_ci95_low",
        "relaxed_5pct_ci95_high",
        "relaxed_pct_norm_5pct_accuracy",
        "parse_failures",
        "truncated",
        "backend_errors",
        "visual_tokens_min",
        "visual_tokens_p50",
        "visual_tokens_max",
        "ttft_p50_ms",
        "tpot_p50_ms",
        "e2e_p50_ms",
        "peak_vram_max_mb",
        "reasoning_tokens_mean",
    ])
    for p in passes_sorted:
        s = pass_summaries[str(p)]
        rci = s["relaxed_5pct_ci95"]
        eff_layers = s["effective_layers_unique"][0] if len(s["effective_layers_unique"]) == 1 else (24 + 12 * (p - 1))
        hw.writerow([
            p,
            eff_layers,
            s["n"],
            s["accuracy"],
            s["strict_tol_0p01_accuracy"],
            s["relaxed_5pct_accuracy"],
            rci[0],
            rci[1],
            s["relaxed_pct_norm_5pct_accuracy"],
            s["parse_failures"],
            s["truncated"],
            s["backend_errors"],
            s["visual_tokens_min"],
            s["visual_tokens_p50"],
            s["visual_tokens_max"],
            s["ttft_p50_ms"],
            s["tpot_p50_ms"],
            s["e2e_p50_ms"],
            s["peak_vram_max_mb"],
            s["reasoning_tokens_mean"],
        ])
    atomic_write_text(out_dir / "holdout_pass_curve.csv", csv_buf.getvalue(), encoding="utf-8-sig")

    plot_name = f"pareto_holdout_{holdout_tag}_mock.png" if any_mock else f"pareto_holdout_{holdout_tag}.png"
    sample_vtoks = [
        int(r["visual_tokens"])
        for r in pass_rows[passes_sorted[0]]
        if r.get("visual_tokens") is not None
    ]
    _render_holdout_figure(
        passes_sorted=passes_sorted,
        pass_summaries=pass_summaries,
        visual_tokens_list=sample_vtoks,
        holdout_tag=holdout_tag,
        any_mock=any_mock,
        out_path=out_dir / plot_name,
    )

    md = [
        f"# {MOCK_STAMP}" if any_mock else f"# Phase 5B: One-Source Real-World Holdout ({holdout_tag})",
        "",
        "- Source: `HuggingFaceM4/ChartQA` (`test` split, `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`), human-authored numeric paired clusters.",
        "",
        "| Passes | Layers | Relaxed 5% Acc | Relaxed 95% CI | Strict (±0.01) Acc | Pct-Norm 5% Acc | Truncated | TTFT P50 (ms) | TPOT P50 (ms) | E2E P50 (s) | Peak VRAM (MB) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for p in passes_sorted:
        s = pass_summaries[str(p)]
        rci = s["relaxed_5pct_ci95"]
        md.append(
            f"| {p} | {24+12*(p-1)} | {(s['relaxed_5pct_accuracy'] or 0)*100:.1f}% ({s['relaxed_5pct_correct']}/{s['n']}) | "
            f"[{(rci[0] or 0)*100:.1f}%, {(rci[1] or 0)*100:.1f}%] | "
            f"{(s['strict_tol_0p01_accuracy'] or 0)*100:.1f}% | "
            f"{(s['relaxed_pct_norm_5pct_accuracy'] or 0)*100:.1f}% | "
            f"{s['truncated']} | {(s['ttft_p50_ms'] or 0):.1f} | {(s['tpot_p50_ms'] or 0):.2f} | "
            f"{(s['e2e_p50_ms'] or 0)/1000:.2f} | {(s['peak_vram_max_mb'] or 0):.1f} |"
        )
    atomic_write_text(out_dir / "report.md", "\n".join(md) + "\n")
    return payload


def _render_holdout_figure(
    passes_sorted: list[int],
    pass_summaries: dict[str, dict],
    visual_tokens_list: list[int],
    holdout_tag: str,
    any_mock: bool,
    out_path: Path,
) -> None:
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4), dpi=140)
        ax_pareto, ax_tier, ax_lat, ax_vtok = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        e2e_s = [(pass_summaries[str(p)]["e2e_p50_ms"] or 0.0) / 1000.0 for p in passes_sorted]
        rel_accs = [(pass_summaries[str(p)]["relaxed_5pct_accuracy"] or 0.0) * 100.0 for p in passes_sorted]
        ci_lo = [
            max(0.0, rel_accs[i] - ((pass_summaries[str(p)]["relaxed_5pct_ci95"][0] or 0.0) * 100.0))
            for i, p in enumerate(passes_sorted)
        ]
        ci_hi = [
            max(0.0, ((pass_summaries[str(p)]["relaxed_5pct_ci95"][1] or 0.0) * 100.0) - rel_accs[i])
            for i, p in enumerate(passes_sorted)
        ]
        ax_pareto.plot(e2e_s, rel_accs, color="#1F4E79", linewidth=2.1, zorder=2)
        ax_pareto.errorbar(
            e2e_s, rel_accs, yerr=[ci_lo, ci_hi],
            fmt="o", markersize=8, color="#C00000", ecolor="#4472C4", elinewidth=1.6, capsize=4, zorder=3,
        )
        for idx_p, (x_v, y_v, p) in enumerate(zip(e2e_s, rel_accs, passes_sorted)):
            dx_p = -32 if idx_p == len(passes_sorted) - 1 else 6
            ax_pareto.annotate(f"loop={p}", (x_v, y_v), textcoords="offset points", xytext=(dx_p, -16), fontsize=8.5, fontweight="bold")
        ax_pareto.set_xlabel("Single-Stream E2E Latency P50 (s)")
        ax_pareto.set_ylabel("ChartQA Relaxed 5% Accuracy (%)")
        ax_pareto.set_ylim(-5, 108)
        ax_pareto.margins(x=0.12)
        ax_pareto.grid(True, alpha=0.3)
        ax_pareto.set_title(f"A. Real-World Holdout Pareto Curve ({holdout_tag})")

        strict_ys = [(pass_summaries[str(p)]["strict_tol_0p01_accuracy"] or 0.0) * 100.0 for p in passes_sorted]
        pct_ys = [(pass_summaries[str(p)]["relaxed_pct_norm_5pct_accuracy"] or 0.0) * 100.0 for p in passes_sorted]
        ax_tier.plot(passes_sorted, strict_ys, "-s", color="#2E75B6", linewidth=2.0, label="Strict (±0.01)")
        ax_tier.plot(passes_sorted, rel_accs, "-o", color="#C00000", linewidth=2.2, label="ChartQA Relaxed (±5%)")
        ax_tier.plot(passes_sorted, pct_ys, "--^", color="#70AD47", linewidth=1.9, label="Pct-Normalized (±5%)")
        ax_tier.set_xticks(passes_sorted, [f"loop={p}\n({24+12*(p-1)}L)" for p in passes_sorted])
        ax_tier.set_ylabel("Accuracy (%)")
        ax_tier.set_ylim(-5, 108)
        ax_tier.grid(True, alpha=0.3)
        ax_tier.legend(loc="lower right", fontsize=8.5, frameon=True)
        ax_tier.set_title("B. Holdout Accuracy Across Grading Tolerances")

        base_s = pass_summaries[str(passes_sorted[0])]
        ttft_r = [_safe_ratio(pass_summaries[str(p)]["ttft_p50_ms"], base_s["ttft_p50_ms"]) or 1.0 for p in passes_sorted]
        tpot_r = [_safe_ratio(pass_summaries[str(p)]["tpot_p50_ms"], base_s["tpot_p50_ms"]) or 1.0 for p in passes_sorted]
        e2e_r = [_safe_ratio(pass_summaries[str(p)]["e2e_p50_ms"], base_s["e2e_p50_ms"]) or 1.0 for p in passes_sorted]
        ax_lat.plot(passes_sorted, ttft_r, "-o", color="#4472C4", linewidth=2.0, label="Observable TTFT P50 Ratio")
        ax_lat.plot(passes_sorted, tpot_r, "-s", color="#ED7D31", linewidth=2.0, label="Decode TPOT P50 Ratio")
        ax_lat.plot(passes_sorted, e2e_r, "-^", color="#7030A0", linewidth=2.0, label="E2E Latency P50 Ratio")
        ax_lat.set_xticks(passes_sorted, [f"loop={p}" for p in passes_sorted])
        ax_lat.set_ylabel("Scaling Ratio vs loop=1 (×)")
        ax_lat.grid(True, alpha=0.3)
        ax_lat.legend(loc="upper left", fontsize=8.5, frameon=True)
        ax_lat.set_title("C. Native-Resolution Latency Scaling")

        if visual_tokens_list:
            ax_vtok.hist(visual_tokens_list, bins=min(12, max(4, len(set(visual_tokens_list)))), color="#5B9BD5", edgecolor="#222222")
        ax_vtok.set_xlabel("Processor-Derived Visual Tokens per Image")
        ax_vtok.set_ylabel("Question Count")
        ax_vtok.grid(axis="y", alpha=0.3)
        ax_vtok.set_title("D. Native Chart Visual Token Distribution")

        suptitle = (
            MOCK_STAMP
            if any_mock
            else f"LoopBench Phase 5B ({holdout_tag}): One-Source Real-World ChartQA Holdout"
        )
        fig.suptitle(suptitle, fontsize=12.5, fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(out_path, metadata={"Software": "LoopBench"})
        plt.close(fig)
