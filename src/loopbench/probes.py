"""Cross-Pass Latent Convergence, Logit-Lens & Dynamic Early-Exit Probes.

Hooks the final layer of the Recurrent Core (`layers[core_end - 1]`) during a single
`loop_num_passes = P` forward/generation trace to measure how hidden representations
and next-token distributions evolve across passes `p = 1 .. P`:
  1. Hidden-state convergence: `cos_sim(h_{p-1}, h_p)` and relative L2 delta
     `||h_p - h_{p-1}||_2 / (||h_p||_2 + eps)`.
  2. Logit-Lens / Final-Norm projection: projects each pass's hidden state through
     `language_model.norm` + `lm_head` to obtain per-pass Top-1 token, Top-1 probability,
     Shannon entropy `H_p`, and agreement with the final pass (`p = P`).
  3. Offline Dynamic Early-Exit Simulation: sweeps halting thresholds on strictly
     observable pass-1/pass-2 signals (cosine stability, relative L2 delta, logit entropy)
     to quantify achievable compute savings vs. exact-token fidelity.
"""

from __future__ import annotations

import math
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


def _project_to_logits(h_vec: Any, lang_model: Any, lm_head: Any):
    """Apply final RMSNorm/LayerNorm + lm_head to a `[..., D]` hidden state."""
    norm = getattr(lang_model, "norm", None) or getattr(lang_model, "final_layernorm", None)
    x = norm(h_vec) if norm is not None else h_vec
    return lm_head(x)


def compute_pass_trajectory_metrics(
    pass_hiddens: list[Any],
    lang_model: Any,
    lm_head: Any,
) -> dict[str, Any]:
    """Compute per-pass convergence and logit-lens metrics for one token step.

    `pass_hiddens`: list of length `P`, each tensor of shape `(D,)` (float/bfloat16 on device).
    """
    import torch
    import torch.nn.functional as F

    P = len(pass_hiddens)
    hs = [h.detach().float() for h in pass_hiddens]

    cos_sims: list[float | None] = [None]
    rel_l2s: list[float | None] = [None]
    step_l2s: list[float | None] = [None]
    banach_ratios: list[float | None] = [None]
    for p in range(1, P):
        prev_h, cur_h = hs[p - 1], hs[p]
        cos_val = float(F.cosine_similarity(prev_h.unsqueeze(0), cur_h.unsqueeze(0)).item())
        diff_norm = float(torch.linalg.vector_norm(cur_h - prev_h).item())
        l2_val = float(diff_norm / (torch.linalg.vector_norm(cur_h).item() + 1e-8))
        cos_sims.append(round(cos_val, 6))
        rel_l2s.append(round(l2_val, 6))
        step_l2s.append(round(diff_norm, 6))
        if p >= 2 and step_l2s[p - 1] is not None and step_l2s[p - 1] > 1e-8:
            banach_ratios.append(round(diff_norm / step_l2s[p - 1], 6))
        else:
            banach_ratios.append(None)

    top1_ids: list[int] = []
    top1_probs: list[float] = []
    entropies: list[float] = []
    kl_to_final: list[float] = []
    probs_list: list[Any] = []

    with torch.inference_mode():
        for h_raw in pass_hiddens:
            logits = _project_to_logits(h_raw.unsqueeze(0).unsqueeze(0), lang_model, lm_head)[0, 0].float()
            log_p = F.log_softmax(logits, dim=-1)
            p_dist = log_p.exp()
            top_p, top_i = p_dist.max(dim=-1)
            ent = float(-(p_dist * log_p).sum().item())
            top1_ids.append(int(top_i.item()))
            top1_probs.append(round(float(top_p.item()), 6))
            entropies.append(round(max(0.0, ent), 6))
            probs_list.append((p_dist, log_p))

    final_id = top1_ids[-1]
    final_p, final_log_p = probs_list[-1]
    for p_dist, log_p in probs_list:
        kl = float(F.kl_div(log_p, final_p, reduction="sum").item())
        kl_to_final.append(round(max(0.0, kl), 6))

    agree_final = [tid == final_id for tid in top1_ids]
    first_stable = P
    for p_idx in range(P):
        if all(agree_final[p_idx:]):
            first_stable = p_idx + 1
            break

    return {
        "passes": P,
        "cos_sim_prev": cos_sims,
        "rel_l2_prev": rel_l2s,
        "banach_contraction_ratio": banach_ratios,
        "top1_ids": top1_ids,
        "top1_probs": top1_probs,
        "entropies": entropies,
        "kl_to_final": kl_to_final,
        "agree_final": agree_final,
        "first_stable_pass": first_stable,
    }


def probe_question_latent_trajectory(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    *,
    loop_num_passes: int = 5,
    reasoning_effort: str = "low",
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Run one generation under `loop_num_passes` while recording per-token cross-pass convergence."""
    import torch

    apply_loop_num_passes(model, loop_num_passes)
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

    lang_model = model.model.language_model
    lm_head = getattr(model, "lm_head", None) or getattr(lang_model, "lm_head", None)
    layers = list(lang_model.layers)
    loop_cfg = getattr(lang_model.config, "loop_config", None) or {}
    core_end = int(loop_cfg.get("core_end", getattr(lang_model.config, "loop_end_layer", len(layers) - 6)))
    exit_layer = layers[core_end - 1]

    step_buffers: list[list[Any]] = []
    cur_step_h: list[Any] = []

    def _lm_pre(_m, _inp):
        cur_step_h.clear()

    def _core_exit_post(_m, _inp, out):
        h = _extract_hidden_tensor(out)
        if h is not None and h.ndim >= 3:
            # Capture last token position hidden state for this pass
            cur_step_h.append(h[0, -1, :].detach())

    def _lm_post(_m, _inp, _out):
        if len(cur_step_h) == loop_num_passes:
            step_buffers.append(list(cur_step_h))
        cur_step_h.clear()

    hooks = [
        lang_model.register_forward_pre_hook(_lm_pre),
        exit_layer.register_forward_hook(_core_exit_post),
        lang_model.register_forward_hook(_lm_post),
    ]
    try:
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    finally:
        for h in hooks:
            h.remove()

    gen_seq = out_ids[0][input_len:].detach().cpu().tolist()
    token_metrics = [
        compute_pass_trajectory_metrics(buf, lang_model, lm_head)
        for buf in step_buffers
    ]
    agg = summarize_token_trajectories(token_metrics, loop_num_passes)
    agg["input_tokens"] = input_len
    agg["generated_tokens"] = len(gen_seq)
    agg["probed_steps"] = len(token_metrics)
    agg["token_metrics"] = token_metrics
    return agg


def summarize_token_trajectories(token_metrics: list[dict[str, Any]], passes: int) -> dict[str, Any]:
    """Aggregate per-token cross-pass metrics into per-pass curves and early-exit summaries."""
    if not token_metrics:
        return {"passes": passes, "n_steps": 0, "per_pass": [], "early_exit_sweep": []}

    per_pass: list[dict[str, Any]] = []
    for p_idx in range(passes):
        cos_vals = [m["cos_sim_prev"][p_idx] for m in token_metrics if m["cos_sim_prev"][p_idx] is not None]
        l2_vals = [m["rel_l2_prev"][p_idx] for m in token_metrics if m["rel_l2_prev"][p_idx] is not None]
        banach_vals = [
            m.get("banach_contraction_ratio", [None] * passes)[p_idx]
            for m in token_metrics
            if p_idx < len(m.get("banach_contraction_ratio", [])) and m["banach_contraction_ratio"][p_idx] is not None
        ]
        ent_vals = [m["entropies"][p_idx] for m in token_metrics]
        top1_vals = [m["top1_probs"][p_idx] for m in token_metrics]
        kl_vals = [m["kl_to_final"][p_idx] for m in token_metrics]
        agree_vals = [1.0 if m["agree_final"][p_idx] else 0.0 for m in token_metrics]
        per_pass.append({
            "pass": p_idx + 1,
            "effective_layers": 24 + 12 * p_idx,
            "mean_cos_sim_prev": round(float(np.mean(cos_vals)), 5) if cos_vals else None,
            "mean_rel_l2_prev": round(float(np.mean(l2_vals)), 5) if l2_vals else None,
            "mean_banach_ratio": round(float(np.mean(banach_vals)), 5) if banach_vals else None,
            "p50_banach_ratio": round(float(np.median(banach_vals)), 5) if banach_vals else None,
            "banach_contracting_rate": round(float(np.mean([1.0 if b < 1.0 else 0.0 for b in banach_vals])), 5) if banach_vals else None,
            "mean_entropy": round(float(np.mean(ent_vals)), 5),
            "mean_top1_prob": round(float(np.mean(top1_vals)), 5),
            "mean_kl_to_final": round(float(np.mean(kl_vals)), 5),
            "agree_final_rate": round(float(np.mean(agree_vals)), 5),
        })

    # Simulate entropy-gated and cosine-stability-gated dynamic early exit across tokens
    sweep: list[dict[str, Any]] = []
    for ent_th in (0.05, 0.15, 0.30, 0.50, 1.00):
        exits = []
        matches = []
        for m in token_metrics:
            chosen_p = passes
            for p_idx in range(passes):
                if m["entropies"][p_idx] <= ent_th:
                    chosen_p = p_idx + 1
                    break
            exits.append(chosen_p)
            matches.append(1.0 if m["agree_final"][chosen_p - 1] else 0.0)
        mean_p = float(np.mean(exits))
        sweep.append({
            "rule": f"entropy<={ent_th:.2f}",
            "signal": "entropy",
            "threshold": ent_th,
            "mean_passes": round(mean_p, 3),
            "core_compute_saving_pct": round((1.0 - mean_p / passes) * 100.0, 2),
            "token_fidelity_vs_final": round(float(np.mean(matches)), 5),
        })

    if passes >= 2:
        for cos_th in (0.995, 0.99, 0.98, 0.96, 0.94):
            exits = []
            matches = []
            for m in token_metrics:
                chosen_p = passes
                for p_idx in range(1, passes):
                    c = m["cos_sim_prev"][p_idx]
                    if c is not None and c >= cos_th:
                        chosen_p = p_idx + 1
                        break
                exits.append(chosen_p)
                matches.append(1.0 if m["agree_final"][chosen_p - 1] else 0.0)
            mean_p = float(np.mean(exits))
            sweep.append({
                "rule": f"cos_sim>={cos_th:.3f}",
                "signal": "cos_sim",
                "threshold": cos_th,
                "mean_passes": round(mean_p, 3),
                "core_compute_saving_pct": round((1.0 - mean_p / passes) * 100.0, 2),
                "token_fidelity_vs_final": round(float(np.mean(matches)), 5),
            })

    first_stable_arr = [m["first_stable_pass"] for m in token_metrics]
    return {
        "passes": passes,
        "n_steps": len(token_metrics),
        "mean_first_stable_pass": round(float(np.mean(first_stable_arr)), 3),
        "p50_first_stable_pass": round(float(np.median(first_stable_arr)), 3),
        "per_pass": per_pass,
        "early_exit_sweep": sweep,
    }
