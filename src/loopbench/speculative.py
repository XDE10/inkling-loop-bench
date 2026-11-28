"""Self-Speculative Loop Drafting (`loop=draft_passes` -> `loop=target_passes`).

Exploits the weight-tied architecture of `TML-d24-varloop-research`:
- Zero extra model weights in VRAM (`loads = 1`, dual lightweight KV caches).
- Draft phase runs `draft_passes` (default 1, 24L) for `K` autoregressive tokens.
- Verify phase switches in-place to `target_passes` (default 3, 48L) and validates all
  `K + 1` positions in a single parallel forward pass, yielding up to `K + 1` tokens
  per target forward call under greedy decoding (`temperature = 0.0`).
- Supports two rollback backends:
  1. `native_crop` (preferred on `transformers >= 5.24`): initializes `DynamicCache`
     with `activate_past_recording()` before prefill and uses native O(1) `crop(-n)`
     (and `crop(0)` on full acceptance) to roll back Attention + short-conv states
     without snapshot/replay overhead.
  2. `snapshot_replay` (fallback for custom/toy caches without `activate_past_recording`):
     deep-clones recurrent/sliding states and replays accepted prefixes on rejection.
- Note on BF16 equivalence: parallel `K+1` verification dispatches batched GEMM /
  `causal_conv1d_fn` while autoregressive decoding dispatches single-token GEMV /
  `causal_conv1d_update`. Use `compute_first_mismatch_logit_margin()` and a
  `draft_passes == target_passes` control to distinguish near-tie BF16 numerical drift
  from cache bugs.
"""

from __future__ import annotations

import time
from typing import Any

from .backends.transformers_backend import (
    END_MSG_TOKEN,
    EOS_SAMPLING_TOKEN,
    TEXT_START_TOKEN,
    THINK_START_TOKEN,
    apply_loop_num_passes,
)


def _iter_cache_layers(cache: Any) -> list[Any]:
    if cache is None:
        return []
    layers = getattr(cache, "layers", None)
    if isinstance(layers, (list, tuple)):
        return list(layers)
    return []


def _has_non_sliceable_state(cache: Any) -> bool:
    """Detect whether a HF Cache contains recurrent/conv or sliding states."""
    try:
        import torch
        tensor_types: tuple[type, ...] = (torch.Tensor,)
    except ImportError:
        tensor_types = ()

    if cache is None:
        return False
    for attr in ("conv_states", "ssm_states", "recurrent_states"):
        val = getattr(cache, attr, None)
        if isinstance(val, (list, tuple)) and any(tensor_types and isinstance(x, tensor_types) for x in val):
            return True
        if isinstance(val, dict) and any(tensor_types and isinstance(x, tensor_types) for x in val.values()):
            return True
    for lyr in _iter_cache_layers(cache):
        if getattr(lyr, "is_sliding", False) or getattr(lyr, "sliding_window", None) is not None:
            return True
        for attr in ("conv_state", "conv_states", "ssm_state", "recurrent_state", "recurrent_states", "state"):
            val = getattr(lyr, attr, None)
            if val is not None and ((tensor_types and isinstance(val, tensor_types)) or isinstance(val, (list, dict))):
                return True
        cls_name = type(lyr).__name__.lower()
        if any(k in cls_name for k in ("mamba", "conv", "ssm", "recurrent", "linear", "gdn", "deltanet", "sliding")):
            return True
    return False


def _snapshot_cache(cache: Any) -> dict[str, Any]:
    """Deep-clone lightweight cache tensors so recurrent/conv & sliding states can be rolled back."""
    try:
        import torch
        tensor_types: tuple[type, ...] = (torch.Tensor,)
    except ImportError:
        tensor_types = ()

    def _clone(obj: Any) -> Any:
        if tensor_types and isinstance(obj, tensor_types):
            return obj.clone()
        if isinstance(obj, list):
            return [_clone(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_clone(x) for x in obj)
        if isinstance(obj, dict):
            return {k: _clone(v) for k, v in obj.items()}
        return obj

    snap: dict[str, Any] = {"top": {}, "layers": []}
    if cache is None:
        return snap
    for attr in ("conv_states", "ssm_states", "recurrent_states", "_seen_tokens"):
        if hasattr(cache, attr):
            snap["top"][attr] = _clone(getattr(cache, attr))
    for lyr in _iter_cache_layers(cache):
        l_snap: dict[str, Any] = {}
        is_sliding_lyr = (
            getattr(lyr, "is_sliding", False)
            or getattr(lyr, "sliding_window", None) is not None
            or "sliding" in type(lyr).__name__.lower()
        )
        if hasattr(lyr, "__dict__"):
            for k, v in lyr.__dict__.items():
                if not is_sliding_lyr and k in ("keys", "values", "key_cache", "value_cache"):
                    continue
                if isinstance(v, (*tensor_types, int, float, bool, list, tuple, dict)) or v is None:
                    l_snap[k] = _clone(v)
        snap["layers"].append(l_snap)
    return snap


def _restore_recurrent_snapshot(cache: Any, snap: dict[str, Any]) -> None:
    if cache is None:
        return
    for attr, val in snap.get("top", {}).items():
        if hasattr(cache, attr):
            setattr(cache, attr, val)
    layers = _iter_cache_layers(cache)
    for lyr, l_snap in zip(layers, snap.get("layers", [])):
        for k, v in l_snap.items():
            setattr(lyr, k, v)


def _crop_attention_kv(cache: Any, max_length: int) -> None:
    """Truncate sequence-indexed Attention KV tensors in-place to `max_length`."""
    if cache is None:
        return
    if hasattr(cache, "crop") and not _has_non_sliceable_state(cache):
        cache.crop(max_length)
        return
    try:
        import torch
    except ImportError:
        return

    if hasattr(cache, "key_cache") and isinstance(cache.key_cache, list):
        for i in range(len(cache.key_cache)):
            if isinstance(cache.key_cache[i], torch.Tensor) and cache.key_cache[i].ndim >= 3:
                cache.key_cache[i] = cache.key_cache[i][..., :max_length, :]
    if hasattr(cache, "value_cache") and isinstance(cache.value_cache, list):
        for i in range(len(cache.value_cache)):
            if isinstance(cache.value_cache[i], torch.Tensor) and cache.value_cache[i].ndim >= 3:
                cache.value_cache[i] = cache.value_cache[i][..., :max_length, :]
    for lyr in _iter_cache_layers(cache):
        for attr in ("keys", "values"):
            t = getattr(lyr, attr, None)
            if isinstance(t, torch.Tensor) and t.ndim >= 3 and t.shape[-2] > max_length:
                setattr(lyr, attr, t[..., :max_length, :])
    if hasattr(cache, "_seen_tokens") and isinstance(cache._seen_tokens, int):
        cache._seen_tokens = max_length


def _try_create_native_recording_cache(model: Any) -> Any | None:
    """Create a `transformers` DynamicCache with `activate_past_recording()` enabled if supported."""
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return None
    text_cfg = getattr(getattr(model, "config", None), "text_config", None)
    if text_cfg is None:
        lang = getattr(getattr(model, "model", None), "language_model", None)
        text_cfg = getattr(lang, "config", None)
    if text_cfg is None or not hasattr(text_cfg, "num_hidden_layers"):
        return None
    try:
        cache = DynamicCache(config=text_cfg)
        if hasattr(cache, "activate_past_recording"):
            cache.activate_past_recording()
            return cache
    except Exception:
        return None
    return None


def _forward_step(
    model: Any,
    input_ids: Any,
    past_key_values: Any,
    cached_len: int,
    device: Any,
    use_cache_kwarg: bool = False,
) -> Any:
    import torch

    step_len = int(input_ids.shape[-1])
    total_len = cached_len + step_len
    attn_mask = torch.ones((1, total_len), dtype=torch.long, device=device)
    cache_position = torch.arange(cached_len, cached_len + step_len, dtype=torch.long, device=device)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "use_cache": True,
        "cache_position": cache_position,
    }
    if use_cache_kwarg:
        kwargs["cache"] = past_key_values
    else:
        kwargs["past_key_values"] = past_key_values
    try:
        return model(**kwargs)
    except TypeError:
        kwargs.pop("cache_position", None)
        if use_cache_kwarg and "cache" in kwargs:
            kwargs["past_key_values"] = kwargs.pop("cache")
        return model(**kwargs)


def _get_out_cache(out: Any) -> Any:
    c = getattr(out, "cache", None)
    if c is not None:
        return c
    return getattr(out, "past_key_values", None)


def verify_exact_equivalence(spec_ids: list[int], baseline_ids: list[int]) -> dict[str, Any]:
    min_len = min(len(spec_ids), len(baseline_ids))
    first_mismatch = None
    for i in range(min_len):
        if int(spec_ids[i]) != int(baseline_ids[i]):
            first_mismatch = i
            break
    if first_mismatch is None and len(spec_ids) != len(baseline_ids):
        first_mismatch = min_len
    return {
        "exact_match": first_mismatch is None,
        "first_mismatch_idx": first_mismatch,
        "spec_len": len(spec_ids),
        "baseline_len": len(baseline_ids),
    }


def compute_first_mismatch_logit_margin(
    model: Any,
    base_inputs: dict[str, Any],
    baseline_ids: list[int],
    spec_ids: list[int],
    mismatch_idx: int,
    *,
    target_passes: int = 3,
) -> dict[str, Any]:
    """Run one teacher-forced target forward on `prompt + baseline_ids[:mismatch_idx]` to inspect
    the exact logit margin at `mismatch_idx` (distinguishes near-tie BF16 drift from cache bugs)."""
    import torch

    device = getattr(model, "device", base_inputs["input_ids"].device)
    apply_loop_num_passes(model, target_passes)
    prefix_gen = baseline_ids[:mismatch_idx]
    if prefix_gen:
        gen_t = torch.tensor([prefix_gen], dtype=torch.long, device=device)
        full_ids = torch.cat([base_inputs["input_ids"], gen_t], dim=-1)
    else:
        full_ids = base_inputs["input_ids"]
    attn_mask = torch.ones_like(full_ids)
    fwd_in = {k: v for k, v in base_inputs.items() if k not in ("input_ids", "attention_mask")}
    fwd_in["input_ids"] = full_ids
    fwd_in["attention_mask"] = attn_mask

    with torch.inference_mode():
        out = model(**fwd_in, use_cache=False)
        logits = out.logits[0, -1, :].float()
        top_vals, top_idx = torch.topk(logits, k=5)
        b_id = int(baseline_ids[mismatch_idx]) if mismatch_idx < len(baseline_ids) else -1
        s_id = int(spec_ids[mismatch_idx]) if mismatch_idx < len(spec_ids) else -1
        b_logit = float(logits[b_id].item()) if b_id >= 0 else None
        s_logit = float(logits[s_id].item()) if s_id >= 0 else None
        tf_top1 = int(top_idx[0].item())
        top1_minus_top2 = float((top_vals[0] - top_vals[1]).item())
        gap_b_s = float(b_logit - s_logit) if (b_logit is not None and s_logit is not None) else None
        s_rank = int((logits > logits[s_id]).sum().item()) + 1 if s_id >= 0 else None
        b_rank = int((logits > logits[b_id]).sum().item()) + 1 if b_id >= 0 else None

    return {
        "mismatch_idx": int(mismatch_idx),
        "teacher_forced_top1_id": tf_top1,
        "baseline_id": b_id,
        "spec_id": s_id,
        "baseline_rank_in_tf": b_rank,
        "spec_rank_in_tf": s_rank,
        "top1_minus_top2_logit": round(top1_minus_top2, 6),
        "baseline_minus_spec_logit": round(gap_b_s, 6) if gap_b_s is not None else None,
        "top5_ids": [int(x) for x in top_idx.tolist()],
        "top5_logits": [round(float(x), 5) for x in top_vals.tolist()],
    }


def speculative_loop_generate(
    model: Any,
    base_inputs: dict[str, Any],
    *,
    tokenizer: Any = None,
    draft_passes: int = 1,
    target_passes: int = 3,
    draft_k: int = 4,
    max_new_tokens: int = 2048,
    eos_token_id: int | None = None,
    use_native_crop: bool = True,
) -> dict[str, Any]:
    """Run greedy Self-Speculative Loop Drafting (`draft_passes` -> `target_passes`)."""
    import torch

    if draft_k < 1:
        raise ValueError(f"draft_k must be >= 1, got {draft_k}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

    device = getattr(model, "device", base_inputs["input_ids"].device)
    use_cuda = torch.cuda.is_available() and getattr(device, "type", "cpu") == "cuda"

    if eos_token_id is None:
        cfg_eos = getattr(getattr(model, "config", None), "eos_token_id", 200006)
        eos_token_id = int(cfg_eos[0]) if isinstance(cfg_eos, (list, tuple)) else int(cfg_eos)

    tok_think = tokenizer.convert_tokens_to_ids(THINK_START_TOKEN) if tokenizer is not None else 200008
    tok_text = tokenizer.convert_tokens_to_ids(TEXT_START_TOKEN) if tokenizer is not None else 200004
    tok_end = tokenizer.convert_tokens_to_ids(END_MSG_TOKEN) if tokenizer is not None else 200010
    tok_eos_s = tokenizer.convert_tokens_to_ids(EOS_SAMPLING_TOKEN) if tokenizer is not None else eos_token_id

    prompt_len = int(base_inputs["input_ids"].shape[-1])

    def _sync():
        if use_cuda:
            torch.cuda.synchronize(device)

    _sync()
    t_total_start = time.perf_counter()

    with torch.inference_mode():
        # Attempt native DynamicCache past-recording for O(1) zero-replay rollback
        native_target_cache = _try_create_native_recording_cache(model) if use_native_crop else None
        native_draft_cache = _try_create_native_recording_cache(model) if use_native_crop else None
        using_native_crop = (native_target_cache is not None) and (native_draft_cache is not None)

        # 1. Dual prefill
        t_pre0 = time.perf_counter()
        apply_loop_num_passes(model, target_passes)
        if using_native_crop:
            out_t = model(**base_inputs, cache=native_target_cache, use_cache=True)
            target_kv = _get_out_cache(out_t)
            target_kv.crop(0)
        else:
            out_t = model(**base_inputs, use_cache=True)
            target_kv = _get_out_cache(out_t)
        first_tok = int(out_t.logits[:, -1, :].argmax(dim=-1).item())

        apply_loop_num_passes(model, draft_passes)
        if using_native_crop:
            out_d = model(**base_inputs, cache=native_draft_cache, use_cache=True)
            draft_kv = _get_out_cache(out_d)
            draft_kv.crop(0)
        else:
            out_d = model(**base_inputs, use_cache=True)
            draft_kv = _get_out_cache(out_d)
        _sync()
        prefill_ms = (time.perf_counter() - t_pre0) * 1000.0

        hybrid_cache = _has_non_sliceable_state(target_kv) or _has_non_sliceable_state(draft_kv)
        rollback_mode = "native_crop" if using_native_crop else ("snapshot_replay" if hybrid_cache else "kv_crop")

        generated: list[int] = [first_tok]
        if first_tok == eos_token_id or max_new_tokens == 1:
            _sync()
            e2e_ms = (time.perf_counter() - t_total_start) * 1000.0
            return {
                "generated_ids": generated,
                "draft_passes": draft_passes,
                "target_passes": target_passes,
                "draft_k": draft_k,
                "hybrid_cache_detected": hybrid_cache,
                "rollback_mode": rollback_mode,
                "verify_rounds": 0,
                "total_drafted": 0,
                "total_accepted": 0,
                "bonus_tokens": 0,
                "acceptance_rate": 0.0,
                "mean_accepted_drafts_per_round": 0.0,
                "mean_tokens_per_verify_round": 1.0,
                "thinking_drafted": 0,
                "thinking_accepted": 0,
                "thinking_acceptance_rate": None,
                "answer_drafted": 0,
                "answer_accepted": 0,
                "answer_acceptance_rate": None,
                "prefill_ms": round(prefill_ms, 2),
                "draft_ms": 0.0,
                "verify_ms": 0.0,
                "rollback_ms": 0.0,
                "e2e_ms": round(e2e_ms, 2),
                "ms_per_tok": round(e2e_ms, 2),
            }

        cached_len = prompt_len
        draft_cached_len = prompt_len
        pending_draft_catchup: list[int] = []

        cur_mode = "other"
        if first_tok == tok_think:
            cur_mode = "thinking"
        elif first_tok == tok_text:
            cur_mode = "answer"

        verify_rounds = 0
        total_drafted = 0
        total_accepted = 0
        bonus_tokens = 0
        thinking_drafted = 0
        thinking_accepted = 0
        answer_drafted = 0
        answer_accepted = 0
        draft_ms = 0.0
        verify_ms = 0.0
        rollback_ms = 0.0

        while len(generated) < max_new_tokens:
            rem = max_new_tokens - len(generated)
            k_cur = min(draft_k, rem)
            if k_cur <= 0:
                break

            last_tok = generated[-1]
            target_snap = _snapshot_cache(target_kv) if (not using_native_crop and hybrid_cache) else None
            draft_snap = _snapshot_cache(draft_kv) if (not using_native_crop and hybrid_cache) else None
            draft_snap_len = draft_cached_len

            # --- 2. Draft K tokens with `draft_passes` ---
            _sync()
            td0 = time.perf_counter()
            apply_loop_num_passes(model, draft_passes)
            draft_tokens: list[int] = []
            draft_modes: list[str] = []
            sim_mode = cur_mode

            first_draft_chunk = pending_draft_catchup + [last_tok]
            d_in = torch.tensor([first_draft_chunk], dtype=torch.long, device=device)
            d_out = _forward_step(model, d_in, draft_kv, draft_cached_len, device, use_cache_kwarg=using_native_crop)
            draft_kv = _get_out_cache(d_out)
            draft_cached_len += len(first_draft_chunk)
            pending_draft_catchup = []

            d_tok = int(d_out.logits[:, -1, :].argmax(dim=-1).item())
            draft_tokens.append(d_tok)
            draft_modes.append(sim_mode)
            if d_tok == tok_think:
                sim_mode = "thinking"
            elif d_tok == tok_text:
                sim_mode = "answer"
            elif d_tok in (tok_end, tok_eos_s, eos_token_id):
                sim_mode = "other"

            if d_tok != eos_token_id:
                for _ in range(k_cur - 1):
                    d_in = torch.tensor([[d_tok]], dtype=torch.long, device=device)
                    d_out = _forward_step(model, d_in, draft_kv, draft_cached_len, device, use_cache_kwarg=using_native_crop)
                    draft_kv = _get_out_cache(d_out)
                    draft_cached_len += 1
                    d_tok = int(d_out.logits[:, -1, :].argmax(dim=-1).item())
                    draft_tokens.append(d_tok)
                    draft_modes.append(sim_mode)
                    if d_tok == tok_think:
                        sim_mode = "thinking"
                    elif d_tok == tok_text:
                        sim_mode = "answer"
                    elif d_tok in (tok_end, tok_eos_s, eos_token_id):
                        sim_mode = "other"
                    if d_tok == eos_token_id:
                        break
            _sync()
            draft_ms += (time.perf_counter() - td0) * 1000.0
            k_actual = len(draft_tokens)

            # --- 3. Parallel Verify all `1 + k_actual` positions with `target_passes` ---
            _sync()
            tv0 = time.perf_counter()
            apply_loop_num_passes(model, target_passes)
            verify_seq = [last_tok] + draft_tokens
            v_in = torch.tensor([verify_seq], dtype=torch.long, device=device)
            v_out = _forward_step(model, v_in, target_kv, cached_len, device, use_cache_kwarg=using_native_crop)
            target_kv = _get_out_cache(v_out)
            target_preds = v_out.logits[0].argmax(dim=-1).detach().cpu().tolist()
            _sync()
            verify_ms += (time.perf_counter() - tv0) * 1000.0

            verify_rounds += 1
            total_drafted += k_actual

            n_match = 0
            for j in range(k_actual):
                if draft_modes[j] == "thinking":
                    thinking_drafted += 1
                elif draft_modes[j] == "answer":
                    answer_drafted += 1
                if int(target_preds[j]) == int(draft_tokens[j]):
                    n_match += 1
                    if draft_modes[j] == "thinking":
                        thinking_accepted += 1
                    elif draft_modes[j] == "answer":
                        answer_accepted += 1
                else:
                    for j2 in range(j + 1, k_actual):
                        if draft_modes[j2] == "thinking":
                            thinking_drafted += 1
                        elif draft_modes[j2] == "answer":
                            answer_drafted += 1
                    break

            total_accepted += n_match

            committed: list[int] = draft_tokens[:n_match]
            hit_eos = any(t == eos_token_id for t in committed)
            used_bonus = False
            if not hit_eos and len(generated) + len(committed) < max_new_tokens:
                if n_match == k_actual:
                    next_exact = int(target_preds[k_actual])
                    bonus_tokens += 1
                    used_bonus = True
                else:
                    next_exact = int(target_preds[n_match])
                committed.append(next_exact)
                if next_exact == eos_token_id:
                    hit_eos = True

            for t in committed:
                if t == tok_think:
                    cur_mode = "thinking"
                elif t == tok_text:
                    cur_mode = "answer"
                elif t in (tok_end, tok_eos_s, eos_token_id):
                    cur_mode = "other"

            generated.extend(committed)
            if hit_eos or len(generated) >= max_new_tokens:
                break

            # --- 4. Cache Rollback & Alignment ---
            new_cached_len = cached_len + 1 + n_match
            accepted_prefix_to_cache = [last_tok] + draft_tokens[:n_match]

            _sync()
            tr0 = time.perf_counter()
            if using_native_crop:
                # Native O(1) rollback using transformers >= 5.24 `crop(-n)` + `crop(0)`
                n_reject_target = k_actual - n_match
                target_kv.crop(-n_reject_target if n_reject_target > 0 else 0)

                if n_match == k_actual and used_bonus:
                    draft_kv.crop(0)
                    pending_draft_catchup = [draft_tokens[-1]]
                else:
                    n_reject_draft = draft_cached_len - new_cached_len
                    draft_kv.crop(-n_reject_draft if n_reject_draft > 0 else 0)
                    draft_cached_len = new_cached_len
                    pending_draft_catchup = []
            elif not hybrid_cache:
                _crop_attention_kv(target_kv, new_cached_len)
                if n_match == k_actual and used_bonus:
                    pending_draft_catchup = [draft_tokens[-1]]
                else:
                    _crop_attention_kv(draft_kv, new_cached_len)
                    draft_cached_len = new_cached_len
                    pending_draft_catchup = []
            else:
                if n_match < k_actual:
                    _crop_attention_kv(target_kv, cached_len)
                    _restore_recurrent_snapshot(target_kv, target_snap)
                    apply_loop_num_passes(model, target_passes)
                    t_rep = torch.tensor([accepted_prefix_to_cache], dtype=torch.long, device=device)
                    target_kv = _get_out_cache(_forward_step(model, t_rep, target_kv, cached_len, device))

                if n_match == k_actual and used_bonus:
                    pending_draft_catchup = [draft_tokens[-1]]
                elif draft_cached_len != new_cached_len:
                    _crop_attention_kv(draft_kv, draft_snap_len)
                    _restore_recurrent_snapshot(draft_kv, draft_snap)
                    replay_draft = first_draft_chunk[:-1] + accepted_prefix_to_cache
                    apply_loop_num_passes(model, draft_passes)
                    d_rep = torch.tensor([replay_draft], dtype=torch.long, device=device)
                    draft_kv = _get_out_cache(_forward_step(model, d_rep, draft_kv, draft_snap_len, device))
                    draft_cached_len = draft_snap_len + len(replay_draft)
                    pending_draft_catchup = []
            _sync()
            rollback_ms += (time.perf_counter() - tr0) * 1000.0
            cached_len = new_cached_len

    _sync()
    e2e_ms = (time.perf_counter() - t_total_start) * 1000.0
    n_gen = len(generated)
    return {
        "generated_ids": generated,
        "draft_passes": draft_passes,
        "target_passes": target_passes,
        "draft_k": draft_k,
        "hybrid_cache_detected": hybrid_cache,
        "rollback_mode": rollback_mode,
        "verify_rounds": verify_rounds,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "bonus_tokens": bonus_tokens,
        "acceptance_rate": round(total_accepted / max(1, total_drafted), 4),
        "mean_accepted_drafts_per_round": round(total_accepted / max(1, verify_rounds), 3),
        "mean_tokens_per_verify_round": round((n_gen - 1) / max(1, verify_rounds), 3) if verify_rounds else float(n_gen),
        "thinking_drafted": thinking_drafted,
        "thinking_accepted": thinking_accepted,
        "thinking_acceptance_rate": round(thinking_accepted / thinking_drafted, 4) if thinking_drafted else None,
        "answer_drafted": answer_drafted,
        "answer_accepted": answer_accepted,
        "answer_acceptance_rate": round(answer_accepted / answer_drafted, 4) if answer_drafted else None,
        "prefill_ms": round(prefill_ms, 2),
        "draft_ms": round(draft_ms, 2),
        "verify_ms": round(verify_ms, 2),
        "rollback_ms": round(rollback_ms, 2),
        "e2e_ms": round(e2e_ms, 2),
        "ms_per_tok": round(e2e_ms / max(1, n_gen), 2),
    }
