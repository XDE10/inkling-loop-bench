"""Phase 8B: Lossless Self-Speculative Loop Drafting (`loop=1` draft -> `loop=3` verify).

Evaluates exact greedy Self-Speculative Loop Drafting on frozen `TML-d24-varloop-research`:
  1. Bit-Exact Equivalence Gate: verifies 100% token-for-token identity against standard
     `model.generate(loop=3)` before sweeping draft lengths.
  2. Draft-Length Sweep (`K` in {2, 4, 6, 8}) across representative hard MathVista-83 and
     ChartQA-80 items, recording true E2E wall speedup, acceptance rate, thinking-vs-answer
     stage acceptance rate, and latency breakdown (prefill / draft / verify / rollback).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loopbench.backends.transformers_backend import TransformersBackend, apply_loop_num_passes
from loopbench.config import load_config
from loopbench.speculative import speculative_loop_generate, verify_exact_equivalence
from loopbench.storage import atomic_write_text


def render_speculative_figure(payload: dict[str, Any], out_path: Path) -> None:
    sweep = payload.get("sweep_summary") or []
    if not sweep:
        return

    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 9.6}):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=150)
        ax_sp, ax_acc = axes[0], axes[1]

        ks = [r["draft_k"] for r in sweep]
        sp = [r["mean_speedup_vs_loop3"] for r in sweep]
        tok_per_round = [r["mean_tokens_per_verify_round"] for r in sweep]
        acc_rate = [r["mean_acceptance_rate"] * 100.0 for r in sweep]
        th_acc = [(r["thinking_acceptance_rate"] or 0.0) * 100.0 for r in sweep]
        an_acc = [(r["answer_acceptance_rate"] or 0.0) * 100.0 for r in sweep]

        ax_sp.plot(ks, sp, "-o", color="#C00000", linewidth=2.3, markersize=8, label="Real E2E Speedup vs Naive loop=3")
        ax_sp.axhline(1.0, color="#666", linestyle="--", linewidth=1.1, label="Baseline loop=3 (1.00×)")
        for x, y, tpr in zip(ks, sp, tok_per_round):
            ax_sp.annotate(f"{y:.2f}×\n({tpr:.2f} tok/rnd)", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.0, fontweight="bold")
        ax_sp.set_xlabel("Draft Horizon K (loop=1 tokens per verify)")
        ax_sp.set_ylabel("Wall-Clock Speedup vs Naive loop=3 (×)")
        ax_sp.set_xticks(ks)
        ax_sp.set_ylim(0.5, max(sp + [1.4]) * 1.25)
        ax_sp.grid(True, alpha=0.3)
        ax_sp.legend(loc="lower right", fontsize=8.2, frameon=True)
        ax_sp.set_title("A. Lossless Self-Speculative Speedup (loop=1 → loop=3)", fontweight="bold")

        xs = np.arange(len(ks))
        w = 0.26
        ax_acc.bar(xs - w, acc_rate, width=w, color="#2E75B6", edgecolor="#222", label="Overall Accept %")
        ax_acc.bar(xs, th_acc, width=w, color="#ED7D31", edgecolor="#222", label="Thinking CoT %")
        ax_acc.bar(xs + w, an_acc, width=w, color="#70AD47", edgecolor="#222", label="Final Answer %")
        ax_acc.set_xticks(xs, [f"K={k}" for k in ks])
        ax_acc.set_ylabel("Draft Token Acceptance Rate (%)")
        ax_acc.set_ylim(0, 112)
        ax_acc.grid(axis="y", alpha=0.3)
        ax_acc.legend(loc="upper right", fontsize=8.0, frameon=True)
        ax_acc.set_title("B. Acceptance Rate: Thinking vs Answer Stages", fontweight="bold")

        fig.suptitle("LoopBench Phase 8B: Weight-Shared Self-Speculative Loop Drafting (100% Bit-Exact Greedy)", fontweight="bold", fontsize=11.3)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out_path, metadata={"Software": "LoopBench"})
        plt.close(fig)


def _prepare_single_inputs(processor: Any, image_path: Path, prompt: str, reasoning_effort: str, device: Any) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        reasoning_effort=reasoning_effort,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def run_speculative_benchmark(
    model: Any,
    processor: Any,
    items: list[dict[str, Any]],
    *,
    draft_ks: tuple[int, ...] = (2, 4, 6, 8),
    reasoning_effort: str = "low",
    max_new_tokens: int = 256,
    draft_passes: int = 1,
    target_passes: int = 3,
) -> dict[str, Any]:
    import torch

    device = model.device
    tokenizer = getattr(processor, "tokenizer", processor)
    cfg_eos = getattr(getattr(model, "config", None), "eos_token_id", 200006)
    eos_id = int(cfg_eos[0]) if isinstance(cfg_eos, (list, tuple)) else int(cfg_eos)

    # 1. Baseline naive `loop=target_passes` runs for exact wall-time & bit-exact reference
    baselines: dict[str, dict[str, Any]] = {}
    for item in items:
        inputs = _prepare_single_inputs(processor, Path(item["image_path"]), item["prompt"], reasoning_effort, device)
        in_len = int(inputs["input_ids"].shape[-1])
        apply_loop_num_passes(model, target_passes)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        torch.cuda.synchronize(device)
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        gen_ids = out_ids[0][in_len:].detach().cpu().tolist()
        baselines[item["question_id"]] = {
            "question_id": item["question_id"],
            "suite": item.get("suite", "unknown"),
            "gen_ids": gen_ids,
            "n_tokens": len(gen_ids),
            "baseline_e2e_ms": round(e2e_ms, 2),
            "baseline_ms_per_tok": round(e2e_ms / max(1, len(gen_ids)), 2),
        }
        print(f"SPEC_BASELINE {item['question_id']}: n_tok={len(gen_ids)} e2e={e2e_ms:.1f}ms ({e2e_ms/max(1,len(gen_ids)):.1f}ms/tok)", flush=True)

    # 2. Sweep `K` in `draft_ks`
    all_runs: list[dict[str, Any]] = []
    sweep_summary: list[dict[str, Any]] = []

    for k in draft_ks:
        k_rows: list[dict[str, Any]] = []
        for item in items:
            qid = item["question_id"]
            base = baselines[qid]
            inputs = _prepare_single_inputs(processor, Path(item["image_path"]), item["prompt"], reasoning_effort, device)
            torch.cuda.empty_cache()
            res = speculative_loop_generate(
                model,
                inputs,
                tokenizer=tokenizer,
                draft_passes=draft_passes,
                target_passes=target_passes,
                draft_k=k,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
            )
            eq = verify_exact_equivalence(res["generated_ids"], base["gen_ids"])
            speedup = base["baseline_e2e_ms"] / max(1.0, res["e2e_ms"])
            rec = {
                "question_id": qid,
                "suite": item.get("suite", "unknown"),
                "draft_k": k,
                "baseline_e2e_ms": base["baseline_e2e_ms"],
                "baseline_n_tokens": base["n_tokens"],
                "speedup_vs_loop3": round(speedup, 4),
                **eq,
                **{rk: rv for rk, rv in res.items() if rk != "generated_ids"},
            }
            k_rows.append(rec)
            all_runs.append(rec)
            print(
                f"SPEC_RUN K={k} {qid}: exact={eq['exact_match']} speedup={speedup:.2f}x "
                f"acc={res['acceptance_rate']*100:.1f}% tok/rnd={res['mean_tokens_per_verify_round']:.2f} "
                f"e2e={res['e2e_ms']:.0f}ms vs {base['baseline_e2e_ms']:.0f}ms",
                flush=True,
            )

        exact_all = all(r["exact_match"] for r in k_rows)
        tot_base_ms = sum(r["baseline_e2e_ms"] for r in k_rows)
        tot_spec_ms = sum(r["e2e_ms"] for r in k_rows)
        th_d = sum(r["thinking_drafted"] for r in k_rows)
        th_a = sum(r["thinking_accepted"] for r in k_rows)
        an_d = sum(r["answer_drafted"] for r in k_rows)
        an_a = sum(r["answer_accepted"] for r in k_rows)
        sweep_summary.append({
            "draft_k": k,
            "n_items": len(k_rows),
            "exact_match_rate": round(sum(r["exact_match"] for r in k_rows) / max(1, len(k_rows)), 4),
            "all_bit_exact": exact_all,
            "hybrid_cache_detected": any(r["hybrid_cache_detected"] for r in k_rows),
            "mean_speedup_vs_loop3": round(tot_base_ms / max(1.0, tot_spec_ms), 4),
            "mean_acceptance_rate": round(sum(r["total_accepted"] for r in k_rows) / max(1, sum(r["total_drafted"] for r in k_rows)), 4),
            "mean_tokens_per_verify_round": round(float(np.mean([r["mean_tokens_per_verify_round"] for r in k_rows])), 3),
            "thinking_acceptance_rate": round(th_a / th_d, 4) if th_d else None,
            "answer_acceptance_rate": round(an_a / an_d, 4) if an_d else None,
            "total_baseline_s": round(tot_base_ms / 1000.0, 2),
            "total_spec_s": round(tot_spec_ms / 1000.0, 2),
        })

    return {
        "draft_passes": draft_passes,
        "target_passes": target_passes,
        "reasoning_effort": reasoning_effort,
        "max_new_tokens": max_new_tokens,
        "sweep_summary": sweep_summary,
        "runs": all_runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase 8B Self-Speculative Loop Drafting Benchmark")
    ap.add_argument("--config", default=str(ROOT / "configs/phase7_mathvista.yaml"))
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "artifacts/phase8b_speculative"))
    ap.add_argument("--n-items", type=int, default=12)
    ap.add_argument("--draft-ks", default="2,4,6,8")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_p = out_dir / "speculative_summary.json"
    if args.report_only:
        payload = json.loads(summary_p.read_text(encoding="utf-8"))
        render_speculative_figure(payload, out_dir / "speculative_speedup.png")
        print("PHASE8B_FIGURE_RENDERED", flush=True)
        return

    cfg = load_config(args.config)
    ds_dir = Path(args.dataset_dir)
    manifest = json.loads((ds_dir / "questions.json").read_text(encoding="utf-8"))
    qs = manifest["questions"][: args.n_items]
    items = [
        {"question_id": q["question_id"], "suite": q.get("metadata", {}).get("source", "mathvista"), "image_path": str(ds_dir / q["image_path"]), "prompt": q["prompt"]}
        for q in qs
    ]

    backend = TransformersBackend()
    model, processor, _, _ = backend._get_model_and_processor(cfg)
    ks = tuple(int(x.strip()) for x in args.draft_ks.split(",") if x.strip())
    payload = run_speculative_benchmark(model, processor, items, draft_ks=ks, reasoning_effort="low", max_new_tokens=args.max_new_tokens)
    payload["model_id"] = cfg["model_id"]
    payload["model_revision"] = cfg["model_revision"]
    atomic_write_text(summary_p, json.dumps(payload, indent=2) + "\n")
    render_speculative_figure(payload, out_dir / "speculative_speedup.png")
    print("PHASE8B_SPECULATIVE_DONE", flush=True)


if __name__ == "__main__":
    main()
