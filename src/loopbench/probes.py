"""Cross-Pass True-Coda Teacher-Forced Probes & Offline Early-Exit Simulation.

Evaluates the REAL full-model output distributions across loop depths `p = 1 .. P`
(including Prelude 0..5, p-pass Recurrent Core 6..17 with message gates, full Coda
18..23, `final_input` block-attention residual mixing, `norm`, `lm_head`, and
`logit_softcap`) on the exact generated trajectory:
  1. Post-Coda representation deltas: `cos_sim(h_{p-1}, h_p)`, relative L2 delta
     `||h_p - h_{p-1}||_2 / (||h_p||_2 + eps)`, and adjacent-pass delta ratio
     `R_p = ||h_p - h_{p-1}||_2 / ||h_{p-1} - h_{p-2}||_2` (note: because passes 2+
     restart from Prelude states with gated messages and separate caches, `R_p` is an
     empirical adjacent-delta ratio, not a strict Banach fixed-point contraction proof).
  2. True full-model next-token distribution metrics: Top-1 token, Top-1 prob, Shannon
     entropy `H_p`, KL divergence and Top-1 agreement against BOTH deployed `loop=3`
     real logits AND max-depth `loop=P` real logits.
  3. Offline teacher-forced early-exit frontier reporting BOTH recurrent-core compute
     savings and TOTAL layer-visit savings (`24 + 12*(p-1)`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backends.transformers_backend import apply_loop_num_passes


def _extract_hidden_tensor(layer_output: Any):
    import torch

    if isinstance(layer_output, torch.Tensor):
        return layer_output
    if isinstance(layer_output, (tuple, list)) and layer_output:
        if isinstance(layer_output[0], torch.Tensor):
            return layer_output[0]
    for attr in ("last_hidden_state", "hidden_states"):
        val = getattr(layer_output, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    return None


def _project_to_logits(h_vec: Any, lang_model: Any, lm_head: Any, logit_softcap: Any = None):
    """Apply final RMSNorm/LayerNorm + lm_head (+ optional logit_softcap)."""
    norm = getattr(lang_model, "norm", None) or getattr(lang_model, "final_layernorm", None)
    x = norm(h_vec) if norm is not None else h_vec
    logits = lm_head(x).float()
    if logit_softcap is not None:
        logits = logit_softcap(logits)
    return logits


def compute_pass_metrics_from_logits_and_hiddens(
    pass_hiddens: list[Any],
    pass_logits: list[Any],
    *,
    ref_pass_idx: int | None = None,
) -> dict[str, Any]:
    """Compute exact per-pass hidden-state delta and real-logit agreement metrics for one token.

    `pass_hiddens`: length `P`, each tensor `(D,)`.
    `pass_logits`: length `P`, each tensor `(V,)` already passed through full Coda + norm + lm_head + softcap.
    `ref_pass_idx`: 0-based reference pass index (default `P - 1`, or `2` for deployed `loop=3`).
    """
    import torch
    import torch.nn.functional as F

    P = len(pass_hiddens)
    ref_idx = (P - 1) if ref_pass_idx is None else min(max(0, ref_pass_idx), P - 1)
    hs = [h.detach().float() for h in pass_hiddens]

    cos_sims: list[float | None] = [None]
    rel_l2s: list[float | None] = [None]
    step_l2s: list[float | None] = [None]
    delta_ratios: list[float | None] = [None]
    for p in range(1, P):
        prev_h, cur_h = hs[p - 1], hs[p]
        cos_val = float(F.cosine_similarity(prev_h.unsqueeze(0), cur_h.unsqueeze(0)).item())
        diff_norm = float(torch.linalg.vector_norm(cur_h - prev_h).item())
        l2_val = float(diff_norm / (torch.linalg.vector_norm(cur_h).item() + 1e-8))
        cos_sims.append(round(cos_val, 6))
        rel_l2s.append(round(l2_val, 6))
        step_l2s.append(round(diff_norm, 6))
        if p >= 2 and step_l2s[p - 1] is not None and step_l2s[p - 1] > 1e-8:
            delta_ratios.append(round(diff_norm / step_l2s[p - 1], 6))
        else:
            delta_ratios.append(None)

    top1_ids: list[int] = []
    top1_probs: list[float] = []
    entropies: list[float] = []
    kl_to_final: list[float] = []
    kl_to_loop3: list[float] = []
    probs_list: list[tuple[Any, Any]] = []

    with torch.inference_mode():
        for raw_logits in pass_logits:
            logits = raw_logits.detach().float()
            log_p = F.log_softmax(logits, dim=-1)
            p_dist = log_p.exp()
            top_p, top_i = p_dist.max(dim=-1)
            ent = float(-(p_dist * log_p).sum().item())
            top1_ids.append(int(top_i.item()))
            top1_probs.append(round(float(top_p.item()), 6))
            entropies.append(round(max(0.0, ent), 6))
            probs_list.append((p_dist, log_p))

    final_id = top1_ids[ref_idx]
    final_p, _ = probs_list[ref_idx]
    loop3_idx = min(2, P - 1)
    loop3_id = top1_ids[loop3_idx]
    loop3_p, _ = probs_list[loop3_idx]

    for p_dist, log_p in probs_list:
        kl_f = float(F.kl_div(log_p, final_p, reduction="sum").item())
        kl_3 = float(F.kl_div(log_p, loop3_p, reduction="sum").item())
        kl_to_final.append(round(max(0.0, kl_f), 6))
        kl_to_loop3.append(round(max(0.0, kl_3), 6))

    agree_final = [tid == final_id for tid in top1_ids]
    agree_loop3 = [tid == loop3_id for tid in top1_ids]
    first_stable = P
    for p_idx in range(P):
        if all(agree_final[p_idx:]):
            first_stable = p_idx + 1
            break

    return {
        "passes": P,
        "cos_sim_prev": cos_sims,
        "rel_l2_prev": rel_l2s,
        "adjacent_delta_ratio": delta_ratios,
        "top1_ids": top1_ids,
        "top1_probs": top1_probs,
        "entropies": entropies,
        "kl_to_final": kl_to_final,
        "kl_to_loop3": kl_to_loop3,
        "agree_final": agree_final,
        "agree_loop3": agree_loop3,
        "first_stable_pass": first_stable,
    }


def compute_pass_trajectory_metrics(
    pass_hiddens: list[Any],
    lang_model: Any,
    lm_head: Any,
    logit_softcap: Any = None,
) -> dict[str, Any]:
    """Legacy-compatible helper projecting hidden states through norm + lm_head (+ softcap)."""
    import torch

    with torch.inference_mode():
        pass_logits = [
            _project_to_logits(h_raw.unsqueeze(0).unsqueeze(0), lang_model, lm_head, logit_softcap)[0, 0]
            for h_raw in pass_hiddens
        ]
    return compute_pass_metrics_from_logits_and_hiddens(pass_hiddens, pass_logits)


def probe_question_latent_trajectory(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    *,
    loop_num_passes: int = 5,
    trajectory_passes: int = 3,
    reasoning_effort: str = "low",
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Generate a real trajectory under `trajectory_passes` (default 3, matching deployed Mini),
    then run exact full-model teacher-forced forwards (including full Coda 18..23, `final_input`
    block residual mixing, `norm`, `lm_head`, and `logit_softcap`) for every `p in 1..loop_num_passes`."""
    import torch

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
    device = getattr(model, "device", torch.device("cpu"))
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[-1])

    # 1. Generate real deployed trajectory under `trajectory_passes`
    apply_loop_num_passes(model, trajectory_passes)
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    gen_seq = out_ids[0][input_len:].detach().cpu().tolist()
    n_gen = len(gen_seq)
    if n_gen == 0:
        agg = summarize_token_trajectories([], loop_num_passes)
        agg.update({"input_tokens": input_len, "generated_tokens": 0, "probed_steps": 0, "token_metrics": []})
        return agg

    # 2. Teacher-forced full-model forwards for every p = 1..loop_num_passes on `prompt + gen_seq[:-1]`
    #    Prediction at index `input_len - 1 + t` predicts `gen_seq[t]` for t = 0..n_gen-1.
    full_ids = out_ids[:, : input_len + n_gen - 1]
    tf_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    tf_inputs["input_ids"] = full_ids
    tf_inputs["attention_mask"] = torch.ones_like(full_ids)

    lang_model = model.model.language_model
    per_pass_logits: list[Any] = []
    per_pass_hiddens: list[Any] = []
    captured_last_h: list[Any] = []

    def _lm_post(_m, _inp, out):
        h = _extract_hidden_tensor(out)
        if h is not None:
            captured_last_h.append(h[0, input_len - 1 : input_len - 1 + n_gen, :].detach().cpu())

    hook = lang_model.register_forward_hook(_lm_post)
    try:
        with torch.inference_mode():
            for p in range(1, loop_num_passes + 1):
                captured_last_h.clear()
                apply_loop_num_passes(model, p)
                out_p = model(**tf_inputs, use_cache=False)
                # Real full-model logits (post-Coda + final_input + norm + lm_head + logit_softcap)
                logits_slice = out_p.logits[0, input_len - 1 : input_len - 1 + n_gen, :].detach().cpu()
                per_pass_logits.append(logits_slice)
                if captured_last_h:
                    per_pass_hiddens.append(captured_last_h[-1])
                else:
                    # Fallback if custom mock model lacks language_model.last_hidden_state
                    per_pass_hiddens.append(logits_slice)
                del out_p
    finally:
        hook.remove()
        apply_loop_num_passes(model, trajectory_passes)

    token_metrics: list[dict[str, Any]] = []
    for t in range(n_gen):
        hs_t = [per_pass_hiddens[p_i][t] for p_i in range(loop_num_passes)]
        ls_t = [per_pass_logits[p_i][t] for p_i in range(loop_num_passes)]
        m = compute_pass_metrics_from_logits_and_hiddens(hs_t, ls_t)
        m["actual_generated_id"] = int(gen_seq[t])
        m["agree_actual_gen"] = [tid == int(gen_seq[t]) for tid in m["top1_ids"]]
        token_metrics.append(m)

    agg = summarize_token_trajectories(token_metrics, loop_num_passes)
    agg["probe_method"] = "true_coda_softcap_teacher_forced"
    agg["trajectory_passes"] = trajectory_passes
    agg["input_tokens"] = input_len
    agg["generated_tokens"] = n_gen
    agg["probed_steps"] = len(token_metrics)
    agg["token_metrics"] = token_metrics
    return agg


def summarize_token_trajectories(token_metrics: list[dict[str, Any]], passes: int) -> dict[str, Any]:
    """Aggregate per-token cross-pass metrics into per-pass curves and early-exit frontiers."""
    if not token_metrics:
        return {"passes": passes, "n_steps": 0, "per_pass": [], "early_exit_sweep": [], "early_exit_vs_loop3": []}

    per_pass: list[dict[str, Any]] = []
    for p_idx in range(passes):
        cos_vals = [m["cos_sim_prev"][p_idx] for m in token_metrics if m["cos_sim_prev"][p_idx] is not None]
        l2_vals = [m["rel_l2_prev"][p_idx] for m in token_metrics if m["rel_l2_prev"][p_idx] is not None]
        ratio_key = "adjacent_delta_ratio" if "adjacent_delta_ratio" in token_metrics[0] else "banach_contraction_ratio"
        ratio_vals = [
            m.get(ratio_key, [None] * passes)[p_idx]
            for m in token_metrics
            if p_idx < len(m.get(ratio_key, [])) and m[ratio_key][p_idx] is not None
        ]
        ent_vals = [m["entropies"][p_idx] for m in token_metrics]
        top1_vals = [m["top1_probs"][p_idx] for m in token_metrics]
        kl_vals = [m["kl_to_final"][p_idx] for m in token_metrics]
        kl3_vals = [m["kl_to_loop3"][p_idx] for m in token_metrics if "kl_to_loop3" in m]
        agree_vals = [1.0 if m["agree_final"][p_idx] else 0.0 for m in token_metrics]
        agree3_vals = [1.0 if m["agree_loop3"][p_idx] else 0.0 for m in token_metrics if "agree_loop3" in m]
        agree_gen_vals = [1.0 if m["agree_actual_gen"][p_idx] else 0.0 for m in token_metrics if "agree_actual_gen" in m]

        per_pass.append({
            "pass": p_idx + 1,
            "effective_layers": 24 + 12 * p_idx,
            "mean_cos_sim_prev": round(float(np.mean(cos_vals)), 5) if cos_vals else None,
            "mean_rel_l2_prev": round(float(np.mean(l2_vals)), 5) if l2_vals else None,
            "mean_adjacent_delta_ratio": round(float(np.mean(ratio_vals)), 5) if ratio_vals else None,
            "p50_adjacent_delta_ratio": round(float(np.median(ratio_vals)), 5) if ratio_vals else None,
            "delta_ratio_lt1_rate": round(float(np.mean([1.0 if b < 1.0 else 0.0 for b in ratio_vals])), 5) if ratio_vals else None,
            "mean_entropy": round(float(np.mean(ent_vals)), 5),
            "mean_top1_prob": round(float(np.mean(top1_vals)), 5),
            "mean_kl_to_final": round(float(np.mean(kl_vals)), 5),
            "mean_kl_to_loop3": round(float(np.mean(kl3_vals)), 5) if kl3_vals else None,
            "agree_final_rate": round(float(np.mean(agree_vals)), 5),
            "agree_loop3_rate": round(float(np.mean(agree3_vals)), 5) if agree3_vals else None,
            "agree_actual_gen_rate": round(float(np.mean(agree_gen_vals)), 5) if agree_gen_vals else None,
        })

    def _run_early_exit(max_p: int, agree_key: str) -> list[dict[str, Any]]:
        full_layers = 24 + 12 * (max_p - 1)
        out_sweep: list[dict[str, Any]] = []
        for ent_th in (0.05, 0.15, 0.30, 0.50, 1.00):
            exits, matches = [], []
            for m in token_metrics:
                if agree_key not in m:
                    continue
                chosen_p = max_p
                for p_idx in range(max_p):
                    if m["entropies"][p_idx] <= ent_th:
                        chosen_p = p_idx + 1
                        break
                exits.append(chosen_p)
                matches.append(1.0 if m[agree_key][chosen_p - 1] else 0.0)
            if not exits:
                continue
            mean_p = float(np.mean(exits))
            mean_layers = float(np.mean([24 + 12 * (p - 1) for p in exits]))
            out_sweep.append({
                "rule": f"entropy<={ent_th:.2f}",
                "signal": "entropy",
                "threshold": ent_th,
                "max_passes": max_p,
                "mean_passes": round(mean_p, 3),
                "core_compute_saving_pct": round((1.0 - mean_p / max_p) * 100.0, 2),
                "total_layer_visit_saving_pct": round((1.0 - mean_layers / full_layers) * 100.0, 2),
                "token_fidelity": round(float(np.mean(matches)), 5),
            })

        if max_p >= 2:
            for cos_th in (0.995, 0.99, 0.98, 0.96, 0.94):
                exits, matches = [], []
                for m in token_metrics:
                    if agree_key not in m:
                        continue
                    chosen_p = max_p
                    for p_idx in range(1, max_p):
                        c = m["cos_sim_prev"][p_idx]
                        if c is not None and c >= cos_th:
                            chosen_p = p_idx + 1
                            break
                    exits.append(chosen_p)
                    matches.append(1.0 if m[agree_key][chosen_p - 1] else 0.0)
                if not exits:
                    continue
                mean_p = float(np.mean(exits))
                mean_layers = float(np.mean([24 + 12 * (p - 1) for p in exits]))
                out_sweep.append({
                    "rule": f"cos_sim>={cos_th:.3f}",
                    "signal": "cos_sim",
                    "threshold": cos_th,
                    "max_passes": max_p,
                    "mean_passes": round(mean_p, 3),
                    "core_compute_saving_pct": round((1.0 - mean_p / max_p) * 100.0, 2),
                    "total_layer_visit_saving_pct": round((1.0 - mean_layers / full_layers) * 100.0, 2),
                    "token_fidelity": round(float(np.mean(matches)), 5),
                })
        return out_sweep

    sweep_p5 = _run_early_exit(passes, "agree_final")
    sweep_p3 = _run_early_exit(min(3, passes), "agree_loop3") if passes >= 3 and "agree_loop3" in token_metrics[0] else []

    first_stable_arr = [m["first_stable_pass"] for m in token_metrics]
    return {
        "passes": passes,
        "n_steps": len(token_metrics),
        "mean_first_stable_pass": round(float(np.mean(first_stable_arr)), 3),
        "p50_first_stable_pass": round(float(np.median(first_stable_arr)), 3),
        "per_pass": per_pass,
        "early_exit_sweep": sweep_p5,
        "early_exit_vs_loop3": sweep_p3,
    }
