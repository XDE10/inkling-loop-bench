"""Phase 7A: Strict Out-of-Fold Online Adaptive Routing & 2-Stage Escalation Cascade.

Quantifies how much of the idealized Phase 6 Offline Oracle is actually recoverable
by deployable, strictly causal online policies under:
  1. `image_id`-grouped 5-Fold CV (`synth-240` loop1->loop5 and `effort48` low1->low3)
  2. Leave-One-Template-Out (LOTO, k=24) CV (`synth-240` loop1->loop5)
  3. Zero-Shot Cross-Dataset Transfer (`synth-240`-fitted policies frozen & evaluated
     on `effort48` and real-world `ChartQA-80`, plus optional `mathvista` if present)
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

from loopbench.data.holdout import grade_chartqa_relaxed
from loopbench.online_router import evaluate_online_routing_pair, evaluate_zero_shot_transfer
from loopbench.report import MOCK_STAMP
from loopbench.storage import atomic_write_text, load_rows


def _load_prompts(questions_json: Path | None) -> dict[str, str]:
    if questions_json is None or not questions_json.is_file():
        return {}
    data = json.loads(questions_json.read_text(encoding="utf-8"))
    return {q["question_id"]: q["prompt"] for q in data.get("questions", [])}


def _strip_runtime(res: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in res.items() if k != "_fitted_runtime"}


def _chartqa_ok(r: dict[str, Any]) -> bool:
    return (
        r.get("finish_reason") not in ("length", "error")
        and r.get("outcome") not in ("truncated", "backend_error", "parse_failure")
        and grade_chartqa_relaxed(r.get("predicted"), float(r["expected"]))
    )


def run_phase7_online_router_suite(
    phase4_dir: Path,
    phase5_dir: Path,
    out_dir: Path,
    *,
    p4_questions_json: Path | None = None,
    p5_effort_questions_json: Path | None = None,
    p5_holdout_questions_json: Path | None = None,
    n_boot: int = 2000,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    p4_prompts = _load_prompts(p4_questions_json or (ROOT / "artifacts/phase4/questions_synth240.json"))
    p5_eff_prompts = _load_prompts(p5_effort_questions_json or (ROOT / "artifacts/phase5/questions_effort48.json"))
    p5_cqa_prompts = _load_prompts(p5_holdout_questions_json or (ROOT / "artifacts/phase5/questions_chartqa80.json"))

    p4_l1 = load_rows(phase4_dir / "synth240_loop1.jsonl")
    p4_l5 = load_rows(phase4_dir / "synth240_loop5.jsonl")
    p4_l3 = load_rows(phase4_dir / "synth240_loop3.jsonl")
    any_mock = any(bool(r.get("is_mock")) for r in p4_l1 + p4_l5)

    suites: dict[str, dict[str, Any]] = {}
    transfers: dict[str, dict[str, Any]] = {}

    # 1. Phase 4 synth-240: image_id 5-fold CV (loop1 -> loop5)
    s_p4_img = evaluate_online_routing_pair(
        p4_l1, p4_l5, p4_prompts,
        base_label="loop1", esc_label="loop5",
        group_key="image_id", n_folds=5, n_boot=n_boot,
    )
    suites["synth240_loop1_to_loop5_img5fold"] = _strip_runtime(s_p4_img)

    # 2. Phase 4 synth-240: Leave-One-Template-Out CV (k=24)
    uniq_tpls = {r.get("template_id") for r in p4_l1 if r.get("template_id")}
    if len(uniq_tpls) >= 3:
        s_p4_loto = evaluate_online_routing_pair(
            p4_l1, p4_l5, p4_prompts,
            base_label="loop1", esc_label="loop5",
            group_key="template_id", n_boot=n_boot,
        )
        suites["synth240_loop1_to_loop5_loto"] = _strip_runtime(s_p4_loto)

    # 3. Phase 4 synth-240: loop1 -> loop3 (sanity check against intermediate depth)
    if p4_l3:
        s_p4_l3 = evaluate_online_routing_pair(
            p4_l1, p4_l3, p4_prompts,
            base_label="loop1", esc_label="loop3",
            group_key="image_id", n_folds=5, n_boot=n_boot,
        )
        suites["synth240_loop1_to_loop3_img5fold"] = _strip_runtime(s_p4_l3)

    # 4. Phase 5A effort48: low_loop1 -> low_loop3 (5-fold image_id CV + zero-shot from synth-240)
    eff_l1_p = phase5_dir / "effort48_low_loop1.jsonl"
    eff_l3_p = phase5_dir / "effort48_low_loop3.jsonl"
    if eff_l1_p.is_file() and eff_l3_p.is_file():
        eff_l1 = load_rows(eff_l1_p)
        eff_l3 = load_rows(eff_l3_p)
        any_mock = any_mock or any(bool(r.get("is_mock")) for r in eff_l1)
        s_eff = evaluate_online_routing_pair(
            eff_l1, eff_l3, p5_eff_prompts,
            base_label="low_loop1", esc_label="low_loop3",
            group_key="image_id", n_folds=5, n_boot=n_boot,
        )
        suites["effort48_low1_to_low3_img5fold"] = _strip_runtime(s_eff)

        transfers["zeroshot_synth240_to_effort48_low1_low3"] = evaluate_zero_shot_transfer(
            s_p4_img["_fitted_runtime"], eff_l1, eff_l3, p5_eff_prompts,
            base_label="low_loop1", esc_label="low_loop3", n_boot=n_boot,
        )

    # 5. Phase 5B ChartQA-80: zero-shot transfer of synth-240 frozen router/cascade
    cqa_l1_p = phase5_dir / "chartqa80_loop1.jsonl"
    cqa_l5_p = phase5_dir / "chartqa80_loop5.jsonl"
    if cqa_l1_p.is_file() and cqa_l5_p.is_file():
        cqa_l1 = load_rows(cqa_l1_p)
        cqa_l5 = load_rows(cqa_l5_p)
        any_mock = any_mock or any(bool(r.get("is_mock")) for r in cqa_l1)
        transfers["zeroshot_synth240_to_chartqa80_loop1_loop5"] = evaluate_zero_shot_transfer(
            s_p4_img["_fitted_runtime"], cqa_l1, cqa_l5, p5_cqa_prompts,
            base_label="loop1", esc_label="loop5", ok_fn=_chartqa_ok, n_boot=n_boot,
        )

    payload = {
        "label": MOCK_STAMP if any_mock else "REAL MODEL EVALUATION — PHASE 7A STRICT OUT-OF-FOLD ONLINE ROUTING",
        "mock": any_mock,
        "suites": suites,
        "zero_shot_transfers": transfers,
    }
    atomic_write_text(out_dir / "phase7_online_router_summary.json", json.dumps(payload, indent=2) + "\n")

    # Compact CSV across all suites and zero-shot transfers
    buf = io.StringIO()
    cw = csv.writer(buf, lineterminator="\n")
    cw.writerow([
        "suite",
        "policy",
        "mode",
        "n",
        "accuracy",
        "correct",
        "escalation_rate",
        "repaired",
        "regressed",
        "wasted_escalations",
        "e2e_mean_ms",
        "e2e_p50_ms",
        "speedup_vs_fixed_escalated",
        "diff_vs_base",
        "ci95_vs_base_lo",
        "ci95_vs_base_hi",
        "oracle_acc_recovery_ratio",
    ])
    for section in (suites, transfers):
        for s_name, s_obj in section.items():
            for pol_name, p_obj in s_obj["policies"].items():
                ci_b = p_obj["diff_vs_base"]["diff_cluster_bootstrap_95ci"]
                cw.writerow([
                    s_name,
                    pol_name,
                    p_obj["mode"],
                    p_obj["n"],
                    round(p_obj["accuracy"], 6),
                    p_obj["correct"],
                    round(p_obj["escalation_rate"], 4),
                    p_obj["repaired_count"],
                    p_obj["regressed_count"],
                    p_obj["wasted_escalation_count"],
                    round(p_obj["e2e_mean_ms"], 2),
                    round(p_obj["e2e_p50_ms"] or 0.0, 2),
                    p_obj["speedup_vs_fixed_escalated"],
                    round(p_obj["diff_vs_base"]["diff_accuracy"] or 0.0, 6),
                    round(ci_b[0], 6) if ci_b[0] is not None else "",
                    round(ci_b[1], 6) if ci_b[1] is not None else "",
                    p_obj.get("oracle_acc_recovery_ratio", ""),
                ])
    atomic_write_text(out_dir / "phase7_online_router_summary.csv", buf.getvalue(), encoding="utf-8-sig")

    plot_name = "online_router_pareto_mock.png" if any_mock else "online_router_pareto.png"
    _render_online_router_figure(suites, transfers, any_mock, out_dir / plot_name)
    return payload


def _render_online_router_figure(
    suites: dict[str, dict[str, Any]],
    transfers: dict[str, dict[str, Any]],
    any_mock: bool,
    out_path: Path,
) -> None:
    panels = []
    if "synth240_loop1_to_loop5_img5fold" in suites:
        panels.append(("A. synth-240 (loop1→5, 5-Fold Image CV)", suites["synth240_loop1_to_loop5_img5fold"]["policies"], "loop1", "loop5"))
    if "synth240_loop1_to_loop5_loto" in suites:
        panels.append(("B. synth-240 (loop1→5, Leave-1-Template-Out)", suites["synth240_loop1_to_loop5_loto"]["policies"], "loop1", "loop5"))
    if "effort48_low1_to_low3_img5fold" in suites:
        panels.append(("C. effort48 (low1→low3, 5-Fold Image CV)", suites["effort48_low1_to_low3_img5fold"]["policies"], "low_loop1", "low_loop3"))
    if not panels:
        return

    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 9.5}):
        fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 5.0), dpi=150)
        if len(panels) == 1:
            axes = [axes]

        for ax, (title, pols, b_lbl, e_lbl) in zip(axes, panels):
            all_ys = []

            def _pt(key: str):
                p = pols[key]
                x = p["e2e_mean_ms"] / 1000.0
                y = p["accuracy"] * 100.0
                all_ys.append(y)
                return x, y

            # Fixed policies
            fx = [_pt(f"fixed_{b_lbl}"), _pt(f"fixed_{e_lbl}")]
            ax.scatter([fx[0][0], fx[1][0]], [fx[0][1], fx[1][1]], s=75, color="#1F4E79", marker="o", label="Fixed Baselines", zorder=4)
            ax.annotate(b_lbl, fx[0], textcoords="offset points", xytext=(-18, -14), fontsize=8.0, fontweight="bold")
            ax.annotate(e_lbl, fx[1], textcoords="offset points", xytext=(-16, 8), fontsize=8.0, fontweight="bold")

            # Offline Oracles
            ox, oy = _pt("offline_router_oracle_2cfg")
            ax.scatter([ox], [oy], s=180, color="#C00000", marker="*", label="Offline Router Oracle (Upper Bound)", zorder=6)

            # Heuristics (HardFail, Uncapped CoT, Capped Early-Abort CoT)
            heur_keys = [
                ("cascade_hardfail_only", "HardFail"),
                ("cascade_cot_ge600_uncapped", "CoT≥600"),
                ("cascade_cot_400_1600_early_abort", "CoT 400–1600"),
            ]
            hx, hy = [], []
            for k, short in heur_keys:
                if k in pols:
                    x, y = _pt(k)
                    hx.append(x)
                    hy.append(y)
                    if short == "CoT≥600":
                        dx, dy = -48, -14
                    elif short == "CoT 400–1600":
                        dx, dy = -34, 8
                    else:
                        dx, dy = 6, 6
                    ax.annotate(short, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=7.6, color="#555555")
            if hx:
                ax.scatter(hx, hy, s=55, color="#7F7F7F", marker="s", label="Heuristic CoT Cascades", zorder=3)

            # Learned CoT-Agnostic Cascade
            agn_keys = [k for k in ("cascade_ridge_cot_agnostic_lam0p0015", "cascade_ridge_cot_agnostic_lam0p0005", "cascade_ridge_cot_agnostic_lam0p0") if k in pols]
            if agn_keys:
                pts = sorted([_pt(k) for k in agn_keys], key=lambda t: t[0])
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "--v", color="#ED7D31", linewidth=1.6, markersize=6.5, label="Online Cascade (CoT-Agnostic)", zorder=4)

            # Learned Pre-Gen Router
            pre_keys = [k for k in ("pregen_ridge_lam0p0015", "pregen_ridge_lam0p0005", "pregen_ridge_lam0p0") if k in pols]
            if pre_keys:
                pts = sorted([_pt(k) for k in pre_keys], key=lambda t: t[0])
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-^", color="#2E8B57", linewidth=2.1, markersize=7.5, label="Online Pre-Gen Router (OOF)", zorder=5)

            ax.set_xlabel("True Mean E2E Latency (s) [Cumulative for Cascades]")
            ax.set_ylabel("Out-of-Fold Accuracy (%)")
            y_lo = max(0.0, min(all_ys) - 3.5) if (all_ys and min(all_ys) > 50.0) else -5.0
            y_hi = min(101.5, max(all_ys) + 2.5) if all_ys else 105.0
            ax.set_ylim(y_lo, y_hi)
            ax.margins(x=0.14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", fontsize=7.4, frameon=True)
            ax.set_title(title, fontweight="bold", fontsize=9.8)

        fig.suptitle(
            MOCK_STAMP if any_mock else "LoopBench Phase 7A: Closing vs. Exposing the Oracle Gap (Strict Out-of-Fold Online Routing)",
            fontweight="bold",
            fontsize=11.5,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out_path, metadata={"Software": "LoopBench"})
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase 7A Strict Out-of-Fold Online Router & Cascade Suite")
    ap.add_argument("--phase4-dir", default=str(ROOT / "artifacts/phase4"))
    ap.add_argument("--phase5-dir", default=str(ROOT / "artifacts/phase5"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/phase7"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    res = run_phase7_online_router_suite(
        phase4_dir=Path(args.phase4_dir),
        phase5_dir=Path(args.phase5_dir),
        out_dir=Path(args.out),
        n_boot=args.n_boot,
    )
    print(
        "PHASE7A_ONLINE_ROUTER_DONE:",
        json.dumps({"suites": list(res["suites"].keys()), "transfers": list(res["zero_shot_transfers"].keys())}),
        flush=True,
    )


if __name__ == "__main__":
    main()
