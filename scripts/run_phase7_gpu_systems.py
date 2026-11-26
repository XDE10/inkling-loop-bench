"""Phase 7C: Blackwell G4 GPU Systems & Profiling Microbenchmarks (`docs/FINAL_PLAN.md` Phase 6 stretch).

Runs three targeted systems microbenchmarks on the locked `thinkingmachines/TML-d24-varloop-research`
checkpoint (zero weight reload across all probes):
  1. CUDA-Event Architectural Stage Breakdown (`loop` in {1, 2, 3, 5}):
     Measures exact GPU time spent in Vision Encoder vs. Prelude (layers 0..5) vs.
     Recurrent Core (layers 6..17 x passes) vs. Coda (layers 18..23) during both
     Prefill and Autoregressive Decode.
  2. Batched Concurrency Throughput & VRAM Scaling (`batch_size` in {1, 2, 4, 8} x `loop` in {1, 3}):
     Measures TTFT, decode TPOT, aggregate throughput (tok/s), and peak VRAM across
     batch sizes to expose the memory-bandwidth-bound regime of weight-tied looped MoE.
  3. Exact KV/Recurrent Cache Footprint & `use_cache` Ablation (`loop` in {1, 2, 3, 5}):
     Introspects `past_key_values` tensors directly (MB allocated at prompt + 64 / 256 / 512
     generated tokens) and contrasts `use_cache=True` vs `use_cache=False` decode latency.
"""

from __future__ import annotations

import argparse
import csv
import io
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
from loopbench.storage import atomic_write_text


def _tensor_nbytes(obj: Any) -> tuple[int, int]:
    """Recursively count unique tensor bytes and tensor count inside a HF Cache object."""
    import torch

    seen: set[int] = set()
    total_bytes = 0
    tensor_count = 0

    def _walk(x: Any) -> None:
        nonlocal total_bytes, tensor_count
        if x is None:
            return
        if isinstance(x, torch.Tensor):
            ptr = (x.data_ptr(), x.numel(), x.element_size())
            if ptr not in seen:
                seen.add(ptr)
                total_bytes += x.numel() * x.element_size()
                tensor_count += 1
            return
        if isinstance(x, (list, tuple)):
            for item in x:
                _walk(item)
            return
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
            return
        # Handle HF Cache objects (DynamicCache, HybridCache, InklingCache, etc.)
        for attr in ("key_cache", "value_cache", "conv_states", "ssm_states", "layers", "caches"):
            if hasattr(x, attr):
                _walk(getattr(x, attr))
        if hasattr(x, "__dict__"):
            for k, v in x.__dict__.items():
                if not k.startswith("__"):
                    _walk(v)

    _walk(obj)
    return total_bytes, tensor_count


def _prepare_batch_inputs(processor: Any, image_path: Path, prompt: str, batch_size: int, device: Any) -> dict[str, Any]:
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for _ in range(batch_size)
    ]
    inputs = processor.apply_chat_template(
        conversations,
        reasoning_effort="low",
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def run_stage_breakdown(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    passes_list: tuple[int, ...] = (1, 2, 3, 5),
    decode_steps: int = 32,
) -> list[dict[str, Any]]:
    """Measure exact CUDA-event stage latencies (Vision / Prelude / Core / Coda) across pass depths."""
    import torch

    device = model.device
    base_inputs = _prepare_batch_inputs(processor, image_path, prompt, batch_size=1, device=device)
    input_len = int(base_inputs["input_ids"].shape[-1])
    lang_model = model.model.language_model
    vision_mod = getattr(model.model, "visual", None) or getattr(model.model, "vision_tower", None)
    layers = list(lang_model.layers)
    n_phys = len(layers)
    # Inkling2Loop physical architecture: prelude=0..5 (6), recurrent core=6..17 (12), coda=18..23 (6)
    core_start = int(getattr(lang_model.config, "loop_start_layer", 6))
    core_end = int(getattr(lang_model.config, "loop_end_layer", n_phys - 6))

    results: list[dict[str, Any]] = []
    for passes in passes_list:
        eff_layers = apply_loop_num_passes(model, passes)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        # Warmup 1 step
        with torch.inference_mode():
            _ = model.generate(**base_inputs, max_new_tokens=4, do_sample=False, use_cache=True)
        torch.cuda.synchronize(device)

        # 1. Profile Prefill + step-by-step Decode with CUDA events via official model.generate()
        #    so M-RoPE / hybrid recurrent caches always receive exact cache_position & position_ids.
        phase_flag = {"mode": "prefill", "lm_step": 0, "record": True}
        prefill_events: dict[str, list[tuple[Any, Any]]] = {"vision": [], "prelude": [], "core": [], "coda": []}
        decode_events: dict[str, list[tuple[Any, Any]]] = {"vision": [], "prelude": [], "core": [], "coda": []}
        layer_calls = {"prefill": 0, "decode_total": 0}
        captured_kv: dict[str, Any] = {"prefill": None, "decode": None}
        decode_span_events: dict[str, Any] = {"start": None, "end": None}
        hooks = []

        def _lm_pre(_m, _inp):
            if not phase_flag["record"]:
                return
            phase_flag["lm_step"] += 1
            if phase_flag["mode"] == "decode_run":
                if phase_flag["lm_step"] == 2 and decode_span_events["start"] is None:
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    decode_span_events["start"] = ev

        def _lm_post(_m, _inp, out):
            if not phase_flag["record"]:
                return
            pkv = getattr(out, "past_key_values", None)
            if pkv is None and isinstance(out, dict):
                pkv = out.get("past_key_values")
            if pkv is not None:
                if phase_flag["mode"] == "prefill":
                    captured_kv["prefill"] = pkv
                elif phase_flag["mode"] == "decode_run":
                    captured_kv["decode"] = pkv

        hooks.append(lang_model.register_forward_pre_hook(_lm_pre))
        hooks.append(lang_model.register_forward_hook(_lm_post))

        def _register_stage(mod: Any, bucket_name: str):
            pending: list[Any] = []

            def _pre(_m, _inp):
                if not phase_flag["record"]:
                    return
                if phase_flag["mode"] == "decode_run" and phase_flag["lm_step"] < 2:
                    return
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                pending.append(ev)

            def _post(_m, _inp, _out):
                if not pending:
                    return
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_end.record()
                ev_start = pending.pop()
                if phase_flag["mode"] == "prefill":
                    prefill_events[bucket_name].append((ev_start, ev_end))
                    if bucket_name != "vision":
                        layer_calls["prefill"] += 1
                elif phase_flag["mode"] == "decode_run" and phase_flag["lm_step"] >= 2:
                    decode_events[bucket_name].append((ev_start, ev_end))
                    if bucket_name != "vision":
                        layer_calls["decode_total"] += 1

            hooks.append(mod.register_forward_pre_hook(_pre))
            hooks.append(mod.register_forward_hook(_post))

        if vision_mod is not None:
            _register_stage(vision_mod, "vision")
        for idx, lyr in enumerate(layers):
            if idx < core_start:
                _register_stage(lyr, "prelude")
            elif idx < core_end:
                _register_stage(lyr, "core")
            else:
                _register_stage(lyr, "coda")

        try:
            with torch.inference_mode():
                # Prefill pass (exact 1 generated token -> single prefill forward)
                phase_flag["mode"] = "prefill"
                phase_flag["lm_step"] = 0
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                out_pre = model.generate(
                    **base_inputs,
                    max_new_tokens=1,
                    min_new_tokens=1,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
                torch.cuda.synchronize(device)
                prefill_wall_ms = (time.perf_counter() - t0) * 1000.0
                pkv_pre = getattr(out_pre, "past_key_values", None) or captured_kv["prefill"]
                kv_prefill_bytes, kv_prefill_tensors = _tensor_nbytes(pkv_pre)
                kv_cache_type = type(pkv_pre).__name__ if pkv_pre is not None else "None"

                # Full decode pass (1 prefill + `decode_steps` autoregressive decode steps)
                phase_flag["mode"] = "decode_run"
                phase_flag["lm_step"] = 0
                torch.cuda.synchronize(device)
                td0 = time.perf_counter()
                out_dec = model.generate(
                    **base_inputs,
                    max_new_tokens=decode_steps + 1,
                    min_new_tokens=decode_steps + 1,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
                ev_dec_end = torch.cuda.Event(enable_timing=True)
                ev_dec_end.record()
                torch.cuda.synchronize(device)
                total_dec_run_ms = (time.perf_counter() - td0) * 1000.0
                if decode_span_events["start"] is not None:
                    decode_wall_ms = float(decode_span_events["start"].elapsed_time(ev_dec_end))
                else:
                    decode_wall_ms = max(1.0, total_dec_run_ms - prefill_wall_ms)
                pkv_dec = getattr(out_dec, "past_key_values", None) or captured_kv["decode"]
                kv_after_bytes, kv_after_tensors = _tensor_nbytes(pkv_dec)
        finally:
            for h in hooks:
                h.remove()

        def _sum_ms(ev_list: list[tuple[Any, Any]]) -> float:
            return round(sum(s.elapsed_time(e) for s, e in ev_list), 3)

        p_vis = _sum_ms(prefill_events["vision"])
        p_pre = _sum_ms(prefill_events["prelude"])
        p_cor = _sum_ms(prefill_events["core"])
        p_cod = _sum_ms(prefill_events["coda"])

        d_pre = round(_sum_ms(decode_events["prelude"]) / decode_steps, 3)
        d_cor = round(_sum_ms(decode_events["core"]) / decode_steps, 3)
        d_cod = round(_sum_ms(decode_events["coda"]) / decode_steps, 3)
        tpot_step_ms = round(decode_wall_ms / decode_steps, 3)
        d_other = round(max(0.0, tpot_step_ms - (d_pre + d_cor + d_cod)), 3)

        peak_vram_mb = round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0), 2)
        results.append({
            "loop_num_passes": passes,
            "effective_layers_expected": eff_layers,
            "prefill_layers_observed": layer_calls["prefill"],
            "decode_layers_per_step": round(layer_calls["decode_total"] / decode_steps, 2),
            "input_tokens": input_len,
            "decode_steps": decode_steps,
            "prefill_wall_ms": round(prefill_wall_ms, 3),
            "prefill_vision_ms": p_vis,
            "prefill_prelude_ms": p_pre,
            "prefill_core_ms": p_cor,
            "prefill_coda_ms": p_cod,
            "decode_tpot_wall_ms": tpot_step_ms,
            "decode_prelude_ms": d_pre,
            "decode_core_ms": d_cor,
            "decode_coda_ms": d_cod,
            "decode_overhead_ms": d_other,
            "core_share_of_decode_pct": round(100.0 * d_cor / max(tpot_step_ms, 1e-6), 2),
            "kv_cache_type": kv_cache_type,
            "kv_cache_prefill_mb": round(kv_prefill_bytes / (1024.0 * 1024.0), 3),
            "kv_cache_after_decode_mb": round(kv_after_bytes / (1024.0 * 1024.0), 3),
            "kv_cache_delta_per_tok_kb": round((kv_after_bytes - kv_prefill_bytes) / decode_steps / 1024.0, 2),
            "kv_tensor_count": kv_after_tensors,
            "peak_vram_mb": peak_vram_mb,
        })
        print(f"STAGE_BREAKDOWN loop={passes}:", json.dumps(results[-1]), flush=True)
    return results


def run_batch_throughput_scaling(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    passes_list: tuple[int, ...] = (1, 3),
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
    max_new_tokens: int = 64,
) -> list[dict[str, Any]]:
    """Measure batched throughput (agg tok/s), TTFT, TPOT, and peak VRAM across batch sizes."""
    import torch

    device = model.device
    rows: list[dict[str, Any]] = []
    for passes in passes_list:
        eff_layers = apply_loop_num_passes(model, passes)
        for bs in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            try:
                inputs = _prepare_batch_inputs(processor, image_path, prompt, batch_size=bs, device=device)
                in_len = int(inputs["input_ids"].shape[-1])

                # Warmup 2 tokens
                with torch.inference_mode():
                    _ = model.generate(**inputs, max_new_tokens=2, min_new_tokens=2, do_sample=False, use_cache=True)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

                # Timed TTFT (1 token) + Full Decode (`max_new_tokens` exact forced tokens)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    _ = model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False, use_cache=True)
                torch.cuda.synchronize(device)
                ttft_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                with torch.inference_mode():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                torch.cuda.synchronize(device)
                e2e_ms = (time.perf_counter() - t1) * 1000.0
                gen_per_req = int(out_ids.shape[-1] - in_len)
                total_gen_toks = gen_per_req * bs
                decode_ms = max(1.0, e2e_ms - ttft_ms)
                tpot_ms = decode_ms / max(1, gen_per_req - 1)
                agg_decode_tps = (bs * max(1, gen_per_req - 1)) / (decode_ms / 1000.0)
                agg_e2e_tps = total_gen_toks / (e2e_ms / 1000.0)
                peak_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)

                entry = {
                    "loop_num_passes": passes,
                    "effective_layers": eff_layers,
                    "batch_size": bs,
                    "input_tokens_per_req": in_len,
                    "gen_tokens_per_req": gen_per_req,
                    "total_gen_tokens": total_gen_toks,
                    "ttft_ms": round(ttft_ms, 2),
                    "tpot_ms": round(tpot_ms, 2),
                    "e2e_ms": round(e2e_ms, 2),
                    "agg_decode_tok_per_s": round(agg_decode_tps, 2),
                    "agg_e2e_tok_per_s": round(agg_e2e_tps, 2),
                    "peak_vram_mb": round(peak_mb, 2),
                    "oom": False,
                }
            except Exception as exc:
                torch.cuda.empty_cache()
                entry = {
                    "loop_num_passes": passes,
                    "effective_layers": eff_layers,
                    "batch_size": bs,
                    "oom": "out of memory" in str(exc).lower(),
                    "error": repr(exc)[:240],
                }
            rows.append(entry)
            print(f"BATCH_BENCH loop={passes} bs={bs}:", json.dumps(entry), flush=True)
    return rows


def run_use_cache_ablation(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    passes_list: tuple[int, ...] = (1, 3),
    max_new_tokens: int = 32,
) -> list[dict[str, Any]]:
    """Quantify KV-cache speedup (`use_cache=True` vs `False`) at `loop=1` vs `loop=3`."""
    import torch

    device = model.device
    inputs = _prepare_batch_inputs(processor, image_path, prompt, batch_size=1, device=device)
    rows: list[dict[str, Any]] = []
    for passes in passes_list:
        apply_loop_num_passes(model, passes)
        for use_cache in (True, False):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                )
            torch.cuda.synchronize(device)
            e2e_ms = (time.perf_counter() - t0) * 1000.0
            peak_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            entry = {
                "loop_num_passes": passes,
                "use_cache": use_cache,
                "gen_tokens": max_new_tokens,
                "e2e_ms": round(e2e_ms, 2),
                "ms_per_tok": round(e2e_ms / max_new_tokens, 2),
                "peak_vram_mb": round(peak_mb, 2),
            }
            rows.append(entry)
            print(f"CACHE_ABL loop={passes} use_cache={use_cache}:", json.dumps(entry), flush=True)
    return rows


def render_gpu_systems_figure(payload: dict[str, Any], out_path: Path) -> None:
    stages = payload.get("stage_breakdown") or []
    batches = [b for b in (payload.get("batch_throughput") or []) if not b.get("oom") and "agg_decode_tok_per_s" in b]
    if not stages and not batches:
        return

    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 9.5}):
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), dpi=150)
        ax_d, ax_b, ax_kv = axes[0], axes[1], axes[2]

        # Panel A: Decode TPOT CUDA-Event Stage Breakdown
        if stages:
            xs = np.arange(len(stages))
            lbls = [f"loop={r['loop_num_passes']}\n({r['effective_layers_expected']}L)" for r in stages]
            pre = np.array([r["decode_prelude_ms"] for r in stages])
            cor = np.array([r["decode_core_ms"] for r in stages])
            cod = np.array([r["decode_coda_ms"] for r in stages])
            oth = np.array([r["decode_overhead_ms"] for r in stages])
            ax_d.bar(xs, pre, label="Prelude (6L)", color="#5B9BD5", edgecolor="#222", linewidth=0.6)
            ax_d.bar(xs, cor, bottom=pre, label="Recurrent Core (12L×p)", color="#C00000", edgecolor="#222", linewidth=0.6)
            ax_d.bar(xs, cod, bottom=pre + cor, label="Coda (6L)", color="#70AD47", edgecolor="#222", linewidth=0.6)
            ax_d.bar(xs, oth, bottom=pre + cor + cod, label="Framework/LM-Head", color="#BFBFBF", edgecolor="#222", linewidth=0.6)
            for i, r in enumerate(stages):
                tot = r["decode_tpot_wall_ms"]
                ax_d.text(i, tot + max(tot * 0.02, 1.5), f"{tot:.1f}ms\n({r['core_share_of_decode_pct']:.0f}% core)", ha="center", va="bottom", fontsize=7.8)
            ax_d.set_xticks(xs, lbls)
            ax_d.set_ylabel("Decode Latency per Token (ms)")
            ax_d.set_ylim(0, max(r["decode_tpot_wall_ms"] for r in stages) * 1.24)
            ax_d.grid(axis="y", alpha=0.3)
            ax_d.legend(loc="upper left", fontsize=7.8, frameon=True)
            ax_d.set_title("A. CUDA-Event Decode Breakdown", fontweight="bold")

        # Panel B: Batched Throughput Scaling (loop=1 vs loop=3)
        if batches:
            by_p: dict[int, list[dict]] = {}
            for b in batches:
                by_p.setdefault(int(b["loop_num_passes"]), []).append(b)
            cols = {1: "#2E75B6", 3: "#C00000", 5: "#7030A0"}
            for p, arr in sorted(by_p.items()):
                arr.sort(key=lambda d: d["batch_size"])
                bs_vals = [d["batch_size"] for d in arr]
                tps = [d["agg_decode_tok_per_s"] for d in arr]
                ax_b.plot(bs_vals, tps, "-o", color=cols.get(p, "#333"), linewidth=2.2, markersize=7, label=f"loop={p} ({24+12*(p-1)}L)")
                for x, y in zip(bs_vals, tps):
                    ax_b.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.0)
            ax_b.set_xlabel("Concurrent Batch Size")
            ax_b.set_ylabel("Aggregate Decode Throughput (tok/s)")
            ax_b.set_xticks(sorted({b["batch_size"] for b in batches}))
            ax_b.grid(True, alpha=0.3)
            ax_b.legend(loc="upper left", fontsize=8.3, frameon=True)
            ax_b.set_title("B. Batched Concurrency Throughput", fontweight="bold")

        # Panel C: Exact KV-Cache Bytes Growth vs Pass Depth
        if stages:
            ps = [r["loop_num_passes"] for r in stages]
            kv_kb = [r["kv_cache_delta_per_tok_kb"] for r in stages]
            kv_mb = [r["kv_cache_after_decode_mb"] for r in stages]
            ax_kv.plot(ps, kv_mb, "-s", color="#7030A0", linewidth=2.2, markersize=7, label="Total KV/Conv Cache (MB)")
            for x, y, kb in zip(ps, kv_mb, kv_kb):
                ax_kv.annotate(f"{y:.1f} MB\n({kb:.0f} KB/tok)", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.8)
            ax_kv.set_xticks(ps, [f"loop={p}\n({24+12*(p-1)}L)" for p in ps])
            ax_kv.set_ylabel("Allocated KV / Recurrent State (MB)")
            ax_kv.set_ylim(0, max(kv_mb) * 1.28 if kv_mb else 100)
            ax_kv.grid(True, alpha=0.3)
            ax_kv.legend(loc="upper left", fontsize=8.3, frameon=True)
            ax_kv.set_title("C. Per-Pass KV/Conv Cache Growth", fontweight="bold")

        fig.suptitle("LoopBench Phase 7C: Blackwell G4 GPU Systems Microbenchmarks (TML-d24-varloop-research)", fontweight="bold", fontsize=11.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out_path, metadata={"Software": "LoopBench"})
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase 7C Blackwell GPU Systems Microbenchmarks")
    ap.add_argument("--config", default=str(ROOT / "configs/phase7_mathvista.yaml"))
    ap.add_argument("--image", required=True, help="Representative PNG image path")
    ap.add_argument("--out", default=str(ROOT / "artifacts/phase7c_gpu_systems"))
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--batch-gen-tokens", type=int, default=64)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "gpu_systems_summary.json"

    if args.report_only:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        render_gpu_systems_figure(payload, out_dir / "gpu_systems_microbench.png")
        print("PHASE7C_FIGURE_RENDERED", flush=True)
        return

    cfg = load_config(args.config)
    backend = TransformersBackend()
    model, processor, load_s, _ = backend._get_model_and_processor(cfg)
    img_path = Path(args.image)
    prompt = "Determine the exact numeric value shown in the diagram. Show brief reasoning and give the final number."

    stages = run_stage_breakdown(model, processor, img_path, prompt, passes_list=(1, 2, 3, 5), decode_steps=args.decode_steps)
    batches = run_batch_throughput_scaling(model, processor, img_path, prompt, passes_list=(1, 3), batch_sizes=(1, 2, 4, 8), max_new_tokens=args.batch_gen_tokens)
    cache_abl = run_use_cache_ablation(model, processor, img_path, prompt, passes_list=(1, 3), max_new_tokens=32)

    import torch
    payload = {
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model_load_s": round(load_s, 2),
        "stage_breakdown": stages,
        "batch_throughput": batches,
        "use_cache_ablation": cache_abl,
    }
    atomic_write_text(summary_path, json.dumps(payload, indent=2) + "\n")

    # Compact CSV for batch throughput
    buf = io.StringIO()
    cw = csv.writer(buf, lineterminator="\n")
    cw.writerow(["loop_num_passes", "effective_layers", "batch_size", "ttft_ms", "tpot_ms", "e2e_ms", "agg_decode_tok_per_s", "agg_e2e_tok_per_s", "peak_vram_mb", "oom"])
    for b in batches:
        cw.writerow([
            b.get("loop_num_passes"), b.get("effective_layers"), b.get("batch_size"),
            b.get("ttft_ms"), b.get("tpot_ms"), b.get("e2e_ms"),
            b.get("agg_decode_tok_per_s"), b.get("agg_e2e_tok_per_s"), b.get("peak_vram_mb"), b.get("oom"),
        ])
    atomic_write_text(out_dir / "batch_throughput.csv", buf.getvalue(), encoding="utf-8-sig")

    render_gpu_systems_figure(payload, out_dir / "gpu_systems_microbench.png")
    print("PHASE7C_GPU_SYSTEMS_DONE", flush=True)


if __name__ == "__main__":
    main()
