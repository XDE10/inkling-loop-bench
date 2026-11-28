"""Phase 8C: Cross-Pass Latent Convergence, Logit-Lens & Dynamic Early-Exit Probe Suite.

Runs `loop_num_passes=5` latent trajectory probing across a stratified sample of
`MathVista-Hard-83` (covering `F->T` repaired, `T->F` regressed, `F->F` persistent-hard,
and `T->T` loop1-solved items) to quantify:
  1. Per-pass hidden-state convergence (`cos_sim`, `rel_l2`) and Logit-Lens stability
     (`agree_final_rate`, `entropy`, `kl_to_final`).
  2. Whether internal pass-1/pass-2 latent uncertainty separates `F->T` repairs from
     `T->F` regressions where Phase 7A surface CoT length failed.
  3. Achievable Recurrent-Core compute savings vs token fidelity under entropy- and
     cosine-gated Dynamic Early-Exit.
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

from loopbench.backends.transformers_backend import TransformersBackend
from loopbench.config import load_config
from loopbench.probes import probe_question_latent_trajectory, summarize_token_trajectories
from loopbench.storage import atomic_write_text


def render_probes_figure(payload: dict[str, Any], out_path: Path) -> None:
    overall = payload.get("overall_summary") or {}
    per_pass = overall.get("per_pass") or []
    sweep = overall.get("early_exit_sweep") or []
    if not per_pass:
        return

    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 9.5}):
        fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), dpi=150)
        ax_conv, ax_ll, ax_ee = axes[0], axes[1], axes[2]

        ps = [r["pass"] for r in per_pass]
        agree = [r["agree_final_rate"] * 100.0 for r in per_pass]
        ent = [r["mean_entropy"] for r in per_pass]
        cos_ps = [r["pass"] for r in per_pass if r["mean_cos_sim_prev"] is not None]
        cos_vals = [r["mean_cos_sim_prev"] for r in per_pass if r["mean_cos_sim_prev"] is not None]
        l2_vals = [r["mean_rel_l2_prev"] for r in per_pass if r["mean_rel_l2_prev"] is not None]

        # Panel A: Hidden-state convergence
        if cos_ps:
            ax_conv.plot(cos_ps, cos_vals, "-o", color="#2E75B6", linewidth=2.2, markersize=7, label="Cosine Sim (h_{p-1}, h_p)")
            ax_conv2 = ax_conv.twinx()
            ax_conv2.plot(cos_ps, l2_vals, "--s", color="#C00000", linewidth=2.0, markersize=6, label="Rel L2 Δ (||Δh||/||h||)")
            ax_conv2.set_ylabel("Relative L2 State Change", color="#C00000")
            ax_conv.set_xlabel("Recurrent Pass p")
            ax_conv.set_ylabel("Cosine Similarity to Prev Pass", color="#2E75B6")
            ax_conv.set_xticks(cos_ps)
            ax_conv.grid(True, alpha=0.3)
            ax_conv.set_title("A. Hidden-State Convergence Across Passes", fontweight="bold")

        # Panel B: Logit-Lens agreement & entropy
        ax_ll.plot(ps, agree, "-o", color="#70AD47", linewidth=2.2, markersize=7, label="Top-1 Agree w/ Final Pass (%)")
        for x, y in zip(ps, agree):
            ax_ll.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.0)
        ax_ll2 = ax_ll.twinx()
        ax_ll2.plot(ps, ent, "--^", color="#ED7D31", linewidth=2.0, markersize=6, label="Mean Logit Entropy (nats)")
        ax_ll2.set_ylabel("Mean Logit-Lens Entropy (nats)", color="#ED7D31")
        ax_ll.set_xlabel("Recurrent Pass p")
        ax_ll.set_ylabel("Top-1 Agreement with Pass 5 (%)", color="#70AD47")
        ax_ll.set_xticks(ps)
        ax_ll.set_ylim(0, 112)
        ax_ll.grid(True, alpha=0.3)
        ax_ll.set_title("B. Logit-Lens Stability & Entropy Decay", fontweight="bold")

        # Panel C: Dynamic Early-Exit Pareto (Compute Saving vs Token Fidelity)
        for sig, col, mk, lbl in (("entropy", "#7030A0", "o", "Entropy-Gated Exit"), ("cos_sim", "#2E75B6", "s", "Cosine-Stability Exit")):
            pts = [r for r in sweep if r["signal"] == sig]
            if pts:
                xs = [r["core_compute_saving_pct"] for r in pts]
                ys = [r["token_fidelity_vs_final"] * 100.0 for r in pts]
                ax_ee.plot(xs, ys, f"-{mk}", color=col, linewidth=2.0, markersize=6, label=lbl)
                for r in pts:
                    ax_ee.annotate(f"{r['threshold']}", (r["core_compute_saving_pct"], r["token_fidelity_vs_final"] * 100.0), textcoords="offset points", xytext=(4, -9), fontsize=7.3)
        ax_ee.set_xlabel("Recurrent Core Compute Saved (%)")
        ax_ee.set_ylabel("Exact Top-1 Token Fidelity vs Pass 5 (%)")
        ax_ee.set_ylim(max(0.0, min((r["token_fidelity_vs_final"] * 100.0 for r in sweep), default=50) - 10), 103)
        ax_ee.grid(True, alpha=0.3)
        ax_ee.legend(loc="lower left", fontsize=8.0, frameon=True)
        ax_ee.set_title("C. Offline Dynamic Early-Exit Frontier", fontweight="bold")

        fig.suptitle("LoopBench Phase 8C: Cross-Pass Latent Convergence, Logit-Lens & Dynamic Early-Exit (loop=5)", fontweight="bold", fontsize=11.3)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out_path, metadata={"Software": "LoopBench"})
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase 8C Cross-Pass Latent Convergence & Early-Exit Probes")
    ap.add_argument("--config", default=str(ROOT / "configs/phase7_mathvista.yaml"))
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "artifacts/phase8c_probes"))
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_p = out_dir / "probes_summary.json"
    if args.report_only:
        payload = json.loads(summary_p.read_text(encoding="utf-8"))
        render_probes_figure(payload, out_dir / "latent_probes_early_exit.png")
        print("PHASE8C_FIGURE_RENDERED", flush=True)
        return

    cfg = load_config(args.config)
    ds_dir = Path(args.dataset_dir)
    manifest = json.loads((ds_dir / "questions.json").read_text(encoding="utf-8"))
    qs = manifest["questions"][: args.n_items]

    backend = TransformersBackend()
    model, processor, _, _ = backend._get_model_and_processor(cfg)

    all_token_metrics: list[dict[str, Any]] = []
    per_question: list[dict[str, Any]] = []
    for idx, q in enumerate(qs, 1):
        res = probe_question_latent_trajectory(
            model,
            processor,
            str(ds_dir / q["image_path"]),
            q["prompt"],
            loop_num_passes=args.passes,
            reasoning_effort="low",
            max_new_tokens=args.max_new_tokens,
        )
        t_metrics = res.pop("token_metrics")
        all_token_metrics.extend(t_metrics)
        entry = {
            "question_id": q["question_id"],
            "source": q.get("metadata", {}).get("source", "unknown"),
            **res,
        }
        per_question.append(entry)
        p1_agree = res["per_pass"][0]["agree_final_rate"] if res["per_pass"] else 0.0
        p2_cos = res["per_pass"][1]["mean_cos_sim_prev"] if len(res["per_pass"]) > 1 else None
        print(
            f"PROBE [{idx}/{len(qs)}] {q['question_id']}: steps={res['probed_steps']} "
            f"p1_agree={p1_agree*100:.1f}% p2_cos={p2_cos} stable_p={res['mean_first_stable_pass']}",
            flush=True,
        )

    overall = summarize_token_trajectories(all_token_metrics, args.passes)
    payload = {
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "loop_num_passes": args.passes,
        "n_questions": len(per_question),
        "overall_summary": overall,
        "per_question": per_question,
    }
    atomic_write_text(summary_p, json.dumps(payload, indent=2) + "\n")
    render_probes_figure(payload, out_dir / "latent_probes_early_exit.png")
    print("PHASE8C_PROBES_DONE", flush=True)


if __name__ == "__main__":
    main()
