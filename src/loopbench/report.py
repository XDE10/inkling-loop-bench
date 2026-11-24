import csv
import io
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .metrics import (
    evaluate_synth320_extension_rule,
    infer_difficulty,
    paired_cluster_bootstrap_diff_ci,
    summarize,
)
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
