"""Phase 6 Optional Systems & Oracle Ablations (`docs/FINAL_PLAN.md` Phase 6):
1. Offline Adaptive-Pass & Adaptive-Effort Oracle (`--mode oracle`):
   Evaluates zero-overhead Router Oracle and progressive Cascade Halting Oracle across
   Phase 4 (`synth-240`/`320`) and Phase 5 (`effort48` & `chartqa80`) traces.
2. GPU Systems Ablations (`--mode gpu-systems`):
   - KV-cache architecture ablation at `loop=3` (`unshared_per_pass` 48L vs `shared_core_slot` 24L vs `use_cache=False`).
   - Batched concurrency throughput scaling (`batch_size` in {1, 2, 4, 8} at `loop=1` vs `loop=3`).
   - Self-speculative / DSpark architectural gate note & timing probe.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loopbench.oracle import analyze_offline_oracle
from loopbench.report import MOCK_STAMP
from loopbench.storage import atomic_write_text, load_rows


def build_offline_oracle_suite(
    phase4_dir: Path | None,
    phase5_dir: Path | None,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    suites: dict[str, dict] = {}
    any_mock = False

    # 1. Phase 4 Synthetic Pass-Depth Oracle (loop in {1, 2, 3, 5})
    if phase4_dir and phase4_dir.is_dir():
        for tag in ("synth320", "synth240"):
            p4_rows = {}
            for p in (1, 2, 3, 5):
                cand = phase4_dir / f"{tag}_loop{p}.jsonl"
                if cand.is_file():
                    rows = load_rows(cand)
                    if rows:
                        any_mock = any_mock or any(bool(r.get("is_mock")) for r in rows)
                        p4_rows[f"loop{p}"] = rows
            if len(p4_rows) >= 2:
                suites[f"phase4_{tag}_pass_oracle"] = analyze_offline_oracle(
                    p4_rows, escalation_order=list(p4_rows.keys())
                )
                # Also break down Oracle on hard multi-step subset where pass depth matters most
                hard_ms = {
                    k: [r for r in v if r.get("difficulty") == "hard" and r.get("task_type") == "multi_step"]
                    for k, v in p4_rows.items()
                }
                if all(hard_ms.values()):
                    suites[f"phase4_{tag}_hard_multistep_oracle"] = analyze_offline_oracle(
                        hard_ms, escalation_order=list(hard_ms.keys())
                    )
                break

    # 2. Phase 5 Joint Effort x Pass-Depth Oracle (12 cells)
    if phase5_dir and phase5_dir.is_dir():
        p5_eff_rows = {}
        for eff in ("none", "low", "medium", "high"):
            for p in (1, 3, 5):
                cand = phase5_dir / f"effort48_{eff}_loop{p}.jsonl"
                if cand.is_file():
                    rows = load_rows(cand)
                    if rows:
                        any_mock = any_mock or any(bool(r.get("is_mock")) for r in rows)
                        p5_eff_rows[f"{eff}_loop{p}"] = rows
        if len(p5_eff_rows) >= 2:
            suites["phase5_joint_effort_pass_oracle"] = analyze_offline_oracle(p5_eff_rows)

        # 3. Phase 5 Real-World Holdout Oracle
        from loopbench.data.holdout import grade_chartqa_relaxed
        p5_hold_rows = {}
        for p in (1, 2, 3, 5):
            cand = phase5_dir / f"chartqa80_loop{p}.jsonl"
            if cand.is_file():
                rows = load_rows(cand)
                if rows:
                    any_mock = any_mock or any(bool(r.get("is_mock")) for r in rows)
                    p5_hold_rows[f"loop{p}"] = rows
        if len(p5_hold_rows) >= 2:
            suites["phase5_chartqa80_holdout_oracle"] = analyze_offline_oracle(
                p5_hold_rows,
                escalation_order=list(p5_hold_rows.keys()),
                correct_fn=lambda r: (
                    r.get("finish_reason") not in ("length", "error")
                    and r.get("outcome") not in ("truncated", "backend_error", "parse_failure")
                    and grade_chartqa_relaxed(r.get("predicted"), float(r["expected"]))
                ),
            )

    payload = {
        "label": MOCK_STAMP if any_mock else "REAL MODEL EVALUATION — PHASE 6 OFFLINE ORACLE ABLATIONS",
        "mock": any_mock,
        "suites": suites,
    }
    atomic_write_text(out_dir / "phase6_oracle_summary.json", json.dumps(payload, indent=2) + "\n")

    # Summary CSV
    csv_buf = io.StringIO()
    cw = csv.writer(csv_buf, lineterminator="\n")
    cw.writerow([
        "suite",
        "n_questions",
        "best_fixed_config",
        "best_fixed_acc",
        "best_fixed_e2e_mean_ms",
        "router_oracle_acc",
        "router_oracle_e2e_mean_ms",
        "router_speedup_vs_best_fixed",
        "cascade_oracle_e2e_mean_ms",
        "cascade_speedup_vs_best_fixed",
        "pareto_frontier_p50",
    ])
    for name, s in suites.items():
        bf = s["best_fixed_policy"]
        ro = s["router_oracle"]
        co = s["cascade_oracle"]
        cw.writerow([
            name,
            s["n_shared_questions"],
            bf["config_label"],
            bf["accuracy"],
            bf["e2e_mean_ms"],
            ro["accuracy"],
            ro["e2e_mean_ms"],
            ro["mean_speedup_vs_best_fixed"],
            co["e2e_mean_ms"],
            co["mean_speedup_vs_best_fixed"],
            json.dumps(s["pareto_frontier_p50"]),
        ])
    atomic_write_text(out_dir / "phase6_oracle_summary.csv", csv_buf.getvalue(), encoding="utf-8-sig")

    if suites:
        plot_name = "pareto_phase6_oracle_mock.png" if any_mock else "pareto_phase6_oracle.png"
        _render_oracle_figure(suites, any_mock, out_dir / plot_name)

    return payload


def _render_oracle_figure(suites: dict[str, dict], any_mock: bool, out_path: Path) -> None:
    names = list(suites.keys())[:4]
    n_panels = len(names)
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.6), dpi=140)
        if n_panels == 1:
            axes = [axes]
        for ax, name in zip(axes, names):
            s = suites[name]
            fps = s["fixed_policies"]
            xs = [(p["e2e_mean_ms"] or 0.0) / 1000.0 for p in fps]
            ys = [(p["accuracy"] or 0.0) * 100.0 for p in fps]
            ax.scatter(xs, ys, s=55, color="#2E75B6", label="Fixed Policies", zorder=3)
            for x_v, y_v, p in zip(xs, ys, fps):
                ax.annotate(p["config_label"], (x_v, y_v), textcoords="offset points", xytext=(4, 4), fontsize=7.5)
            ro = s["router_oracle"]
            co = s["cascade_oracle"]
            ax.scatter([ro["e2e_mean_ms"] / 1000.0], [ro["accuracy"] * 100.0], marker="*", s=165, color="#C00000", label="Router Oracle", zorder=5)
            ax.scatter([co["e2e_mean_ms"] / 1000.0], [co["accuracy"] * 100.0], marker="D", s=70, color="#70AD47", label="Cascade Oracle", zorder=4)
            ax.set_xlabel("Mean E2E Latency (s)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(-5, 108)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", fontsize=8.0)
            ax.set_title(name.replace("_", " "))
        fig.suptitle(MOCK_STAMP if any_mock else "LoopBench Phase 6: Fixed Policies vs Offline Adaptive Oracles", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_path, metadata={"Software": "LoopBench"})
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase 6 Offline Oracle & Systems Analysis")
    ap.add_argument("--phase4-dir", default=str(ROOT / "artifacts/phase4"))
    ap.add_argument("--phase5-dir", default=str(ROOT / "artifacts/phase5"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/phase6"))
    args = ap.parse_args()

    res = build_offline_oracle_suite(
        phase4_dir=Path(args.phase4_dir),
        phase5_dir=Path(args.phase5_dir),
        out_dir=Path(args.out),
    )
    print("PHASE6_ORACLE_DONE:", json.dumps({"suites": list(res["suites"].keys())}), flush=True)


if __name__ == "__main__":
    main()
