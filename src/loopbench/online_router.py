"""Strictly causal Online Adaptive Routing & 2-Stage Escalation Cascade evaluation.

Tests how much of the idealized Phase 6 Offline Router/Cascade Oracle is actually
recoverable by deployable online policies without label or cluster leakage:

1. Strictly causal feature extraction:
   - Pre-generation features: prompt surface cues + processor token counts only
     (never accesses hidden template_id, difficulty, or ground-truth modality labels).
   - Post-pass-1 features: observable base-pass outputs (truncation, parse failure,
     CoT token count, self-doubt/hedging rate, answer aside/multi-number cues).
2. True latency accounting:
   - Pre-gen routers incur `e2e(chosen)`.
   - Post-pass-1 cascades incur cumulative `e2e(base) + e2e(escalated)` on escalation.
3. Strict anti-leakage evaluation:
   - `image_id`-grouped K-fold CV (paired questions never split across train/test).
   - Leave-One-Template-Out (LOTO) CV where `template_id` has >= 3 distinct templates.
   - Zero-shot cross-dataset transfer (policy fitted on source suite evaluated frozen
     on target suites without refitting).
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .grading import extract_number
from .metrics import _infer_outcome, _percentile, paired_cluster_bootstrap_diff_ci

HEDGE_RE = re.compile(
    r"\b(wait|actually|hold on|double[- ]check|re-?read|hmm|not sure|"
    r"mistake|let me check|careful|confused|alternatively)\b",
    re.I,
)
ASIDE_RE = re.compile(
    r"\b(note:|aside:|remark:|if you meant|in case you meant)\b",
    re.I,
)
NUM_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
GEOM_PROMPT_RE = re.compile(
    r"\b(angle|triangle|vertex|parallel|bisector|perpendicular|degree|"
    r"polygon|quadrilateral|circle|chord|tangent|exterior|interior|radius|diameter|area|perimeter)\b|∠|°",
    re.I,
)
MULTI_PROMPT_RE = re.compile(
    r"\b(difference|sum|total|average|mean|ratio|percent|more than|less than|"
    r"combined|remaining|unknown|solve for|marked ['\"]?\?['\"]?)\b",
    re.I,
)

PREGEN_FEATURES: tuple[str, ...] = (
    "is_geom_prompt",
    "is_multi_prompt",
    "prompt_char_len",
    "prompt_num_count",
    "visual_tokens",
    "input_tokens",
)

COT_AGNOSTIC_FEATURES: tuple[str, ...] = (
    "is_truncated",
    "is_parse_failure",
    "log_reasoning_tokens",
    "hedge_count",
    "hedge_per_100tok",
    "cot_question_marks",
)

ALL_OBSERVABLE_FEATURES: tuple[str, ...] = (
    PREGEN_FEATURES
    + COT_AGNOSTIC_FEATURES
    + (
        "log_answer_tokens",
        "has_aside_in_answer",
        "multi_answer_numbers",
        "last_cot_mismatch",
    )
)


def extract_observable_features(row: dict[str, Any], prompt: str = "") -> dict[str, float]:
    """Extract strictly causal observable features from a prompt and base-pass row."""
    prompt_str = str(prompt or "")
    rtxt = str(row.get("reasoning_text") or "")
    atxt = str(row.get("text") or "")
    rtok = float(row.get("reasoning_tokens") or 0.0)
    atok = float(row.get("answer_tokens") or 0.0)

    is_trunc = 1.0 if (row.get("finish_reason") == "length" or row.get("outcome") == "truncated") else 0.0
    pred = extract_number(atxt)
    is_pf = 1.0 if (pred is None and is_trunc < 0.5) else 0.0

    hedges = float(len(HEDGE_RE.findall(rtxt)))
    qmarks = float(rtxt.count("?"))
    has_aside = 1.0 if (ASIDE_RE.search(atxt) or "(" in atxt) else 0.0
    ans_nums = set(NUM_RE.findall(atxt))
    multi_ans_nums = 1.0 if len(ans_nums) >= 2 else 0.0

    cot_nums = NUM_RE.findall(rtxt)
    last_cot_mismatch = 0.0
    if pred is not None and cot_nums:
        try:
            last_val = float(cot_nums[-1].replace(",", ""))
            if abs(last_val - pred) > 1e-3:
                last_cot_mismatch = 1.0
        except ValueError:
            pass

    return {
        "is_geom_prompt": 1.0 if GEOM_PROMPT_RE.search(prompt_str) else 0.0,
        "is_multi_prompt": 1.0 if MULTI_PROMPT_RE.search(prompt_str) else 0.0,
        "prompt_char_len": float(len(prompt_str)),
        "prompt_num_count": float(len(NUM_RE.findall(prompt_str))),
        "visual_tokens": float(row.get("visual_tokens") or 0.0),
        "input_tokens": float(row.get("input_tokens") or 0.0),
        "is_truncated": is_trunc,
        "is_parse_failure": is_pf,
        "log_reasoning_tokens": math.log1p(max(0.0, rtok)),
        "hedge_count": hedges,
        "hedge_per_100tok": hedges / max(1.0, rtok / 100.0),
        "cot_question_marks": qmarks,
        "log_answer_tokens": math.log1p(max(0.0, atok)),
        "has_aside_in_answer": has_aside,
        "multi_answer_numbers": multi_ans_nums,
        "last_cot_mismatch": last_cot_mismatch,
    }


def _matrix_from_features(feats: list[dict[str, float]], keys: tuple[str, ...]) -> np.ndarray:
    return np.array([[f[k] for k in keys] for f in feats], dtype=np.float64)


def fit_ridge_utility(X: np.ndarray, y_delta: np.ndarray, l2: float = 15.0) -> dict[str, Any]:
    """Fit standardized L2-regularized linear predictor for net accuracy gain Δacc in {-1, 0, +1}."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd_safe = np.where(sd < 1e-8, 1.0, sd)
    Xn = (X - mu) / sd_safe
    Xb = np.hstack([np.ones((Xn.shape[0], 1), dtype=np.float64), Xn])
    reg = np.eye(Xb.shape[1], dtype=np.float64) * float(l2)
    reg[0, 0] = 0.0
    w = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y_delta)
    return {"mu": mu, "sd": sd_safe, "w": w}


def predict_ridge_utility(X: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    Xn = (X - model["mu"]) / model["sd"]
    Xb = np.hstack([np.ones((Xn.shape[0], 1), dtype=np.float64), Xn])
    return Xb @ model["w"]


def _select_utility_threshold(
    scores: np.ndarray,
    y_delta: np.ndarray,
    extra_cost_s: np.ndarray,
    latency_penalty: float,
) -> float:
    """Select decision threshold strictly on training fold to maximize sum(Δacc - λ * Δcost_s)."""
    cand_thresholds = np.unique(
        np.concatenate([
            np.linspace(-0.02, 0.25, 55),
            np.quantile(scores, np.linspace(0.05, 0.98, 25)) if len(scores) >= 4 else np.array([0.0]),
            np.array([float(scores.max()) + 1e-6]),
        ])
    )
    best_t = float(scores.max()) + 1e-6
    best_u = 0.0  # escalating nothing yields utility 0.0
    for t in cand_thresholds:
        mask = scores > t
        if not np.any(mask):
            u = 0.0
        else:
            u = float(np.sum(y_delta[mask] - float(latency_penalty) * extra_cost_s[mask]))
        if u > best_u + 1e-12:
            best_u = u
            best_t = float(t)
    return best_t


def make_group_folds(groups: list[str], n_folds: int = 5, seed: int = 20261126) -> tuple[list[int], int]:
    uniq = sorted(set(groups))
    k = min(int(n_folds), len(uniq))
    if k < 2:
        raise ValueError(f"Need at least 2 distinct groups for CV, got {len(uniq)}")
    rng = np.random.RandomState(seed)
    perm = list(uniq)
    rng.shuffle(perm)
    g2f = {g: i % k for i, g in enumerate(perm)}
    return [g2f[g] for g in groups], k


def evaluate_binary_policy(
    base_rows: list[dict[str, Any]],
    esc_rows: list[dict[str, Any]],
    decisions: list[bool],
    *,
    mode: str = "cascade",
    ok_fn=None,
    n_boot: int = 2000,
    seed: int = 20261123,
) -> dict[str, Any]:
    """Evaluate a binary routing/escalation policy with true latency and cluster-bootstrap CIs."""
    is_ok = ok_fn or (lambda r: _infer_outcome(r) == "correct")
    out_rows: list[dict[str, Any]] = []
    lat_ms: list[float] = []
    repaired = 0
    regressed = 0
    wasted_escalations = 0

    for rb, re_, esc in zip(base_rows, esc_rows, decisions):
        b_ok = bool(is_ok(rb))
        e_ok = bool(is_ok(re_))
        b_ms = float(rb.get("e2e_ms") or 0.0)
        e_ms = float(re_.get("e2e_ms") or 0.0)
        if esc:
            chosen_ok = e_ok
            e2e = (b_ms + e_ms) if mode == "cascade" else e_ms
            if (not b_ok) and e_ok:
                repaired += 1
            elif b_ok and (not e_ok):
                regressed += 1
            else:
                wasted_escalations += 1
        else:
            chosen_ok = b_ok
            e2e = b_ms
        out_rows.append({
            "question_id": rb["question_id"],
            "image_id": rb["image_id"],
            "correct": chosen_ok,
            "outcome": "correct" if chosen_ok else "wrong_answer",
            "e2e_ms": e2e,
        })
        lat_ms.append(e2e)

    n = len(out_rows)
    correct = sum(1 for r in out_rows if r["correct"])
    lat_sorted = sorted(lat_ms)

    base_norm = [
        {
            "question_id": r["question_id"],
            "image_id": r["image_id"],
            "correct": bool(is_ok(r)),
            "outcome": "correct" if is_ok(r) else "wrong_answer",
        }
        for r in base_rows
    ]
    esc_norm = [
        {
            "question_id": r["question_id"],
            "image_id": r["image_id"],
            "correct": bool(is_ok(r)),
            "outcome": "correct" if is_ok(r) else "wrong_answer",
        }
        for r in esc_rows
    ]
    diff_base = paired_cluster_bootstrap_diff_ci(base_norm, out_rows, n_boot=n_boot, seed=seed)
    diff_esc = paired_cluster_bootstrap_diff_ci(esc_norm, out_rows, n_boot=n_boot, seed=seed)

    base_mean_ms = sum(float(r.get("e2e_ms") or 0.0) for r in base_rows) / n
    esc_mean_ms = sum(float(r.get("e2e_ms") or 0.0) for r in esc_rows) / n
    pol_mean_ms = sum(lat_ms) / n

    return {
        "mode": mode,
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "escalation_count": int(sum(decisions)),
        "escalation_rate": sum(decisions) / n,
        "repaired_count": repaired,
        "regressed_count": regressed,
        "wasted_escalation_count": wasted_escalations,
        "e2e_p50_ms": _percentile(lat_sorted, 0.5),
        "e2e_p90_ms": _percentile(lat_sorted, 0.9),
        "e2e_mean_ms": pol_mean_ms,
        "total_gpu_s": round(sum(lat_ms) / 1000.0, 2),
        "speedup_vs_fixed_escalated": round(esc_mean_ms / pol_mean_ms, 4) if pol_mean_ms > 0 else None,
        "slowdown_vs_fixed_base": round(pol_mean_ms / base_mean_ms, 4) if base_mean_ms > 0 else None,
        "diff_vs_base": diff_base,
        "diff_vs_escalated": diff_esc,
    }


def diagnose_oracle_gap(
    base_rows: list[dict[str, Any]],
    esc_rows: list[dict[str, Any]],
    feats: list[dict[str, float]],
    *,
    ok_fn=None,
) -> dict[str, Any]:
    """Quantify why online policies cannot recover 100% of the idealized Offline Oracle."""
    is_ok = ok_fn or (lambda r: _infer_outcome(r) == "correct")
    groups: dict[str, list[tuple[dict, dict, dict]]] = {"F->T": [], "T->F": [], "F->F": [], "T->T": []}
    for rb, re_, f in zip(base_rows, esc_rows, feats):
        k = f"{'T' if is_ok(rb) else 'F'}->{'T' if is_ok(re_) else 'F'}"
        groups[k].append((rb, re_, f))

    def _stats(items: list[tuple[dict, dict, dict]]) -> dict[str, Any]:
        if not items:
            return {"count": 0}
        rtoks = sorted(float(rb.get("reasoning_tokens") or 0.0) for rb, _, _ in items)
        hedges = [f["hedge_count"] for _, _, f in items]
        truncs = sum(1 for _, _, f in items if f["is_truncated"] > 0.5)
        asides = sum(1 for _, _, f in items if f["has_aside_in_answer"] > 0.5)
        multi_nums = sum(1 for _, _, f in items if f["multi_answer_numbers"] > 0.5)
        e2e_esc_s = sum(float(re_.get("e2e_ms") or 0.0) for _, re_, _ in items) / 1000.0
        return {
            "count": len(items),
            "truncated_count": truncs,
            "rtok_median": round(_percentile(rtoks, 0.5) or 0.0, 1),
            "rtok_mean": round(sum(rtoks) / len(rtoks), 1),
            "hedge_mean": round(sum(hedges) / len(hedges), 2),
            "has_aside_rate": round(asides / len(items), 3),
            "multi_num_rate": round(multi_nums / len(items), 3),
            "escalated_gpu_s_total": round(e2e_esc_s, 1),
        }

    silent_repairs = [
        rb["question_id"]
        for rb, _, f in groups["F->T"]
        if (rb.get("reasoning_tokens") or 0) < 300 and f["hedge_count"] <= 1.0
    ]

    return {
        "transition_stats": {k: _stats(v) for k, v in groups.items()},
        "silent_confident_repairs_count": len(silent_repairs),
        "silent_confident_repairs_qids": silent_repairs,
    }


def evaluate_online_routing_pair(
    base_rows_raw: list[dict[str, Any]],
    esc_rows_raw: list[dict[str, Any]],
    prompts_by_qid: dict[str, str],
    *,
    base_label: str = "base",
    esc_label: str = "escalated",
    ok_fn=None,
    group_key: str = "image_id",
    n_folds: int = 5,
    l2: float = 15.0,
    lambdas: tuple[float, ...] = (0.0, 0.0005, 0.0015),
    n_boot: int = 2000,
    seed: int = 20261126,
) -> dict[str, Any]:
    """Run strict out-of-fold evaluation of Pre-Gen Routers & Post-Pass-1 Cascades."""
    is_ok = ok_fn or (lambda r: _infer_outcome(r) == "correct")
    bm = {r["question_id"]: r for r in base_rows_raw}
    em = {r["question_id"]: r for r in esc_rows_raw}
    shared_qids = sorted(set(bm) & set(em))
    if len(shared_qids) < 4:
        raise ValueError(f"Need >= 4 shared questions, got {len(shared_qids)}")

    b_rows = [bm[q] for q in shared_qids]
    e_rows = [em[q] for q in shared_qids]
    feats = [extract_observable_features(r, prompts_by_qid.get(r["question_id"], "")) for r in b_rows]

    y_delta = np.array(
        [(1.0 if is_ok(re_) else 0.0) - (1.0 if is_ok(rb) else 0.0) for rb, re_ in zip(b_rows, e_rows)],
        dtype=np.float64,
    )
    extra_cascade_s = np.array([float(r.get("e2e_ms") or 0.0) / 1000.0 for r in e_rows], dtype=np.float64)
    extra_pregen_s = np.array(
        [(float(re_.get("e2e_ms") or 0.0) - float(rb.get("e2e_ms") or 0.0)) / 1000.0 for rb, re_ in zip(b_rows, e_rows)],
        dtype=np.float64,
    )

    groups = [str(r.get(group_key) or r["image_id"]) for r in b_rows]
    if group_key == "template_id":
        uniq_t = sorted(set(groups))
        folds = [uniq_t.index(g) for g in groups]
        k = len(uniq_t)
    else:
        folds, k = make_group_folds(groups, n_folds=n_folds, seed=seed)

    policies: dict[str, dict[str, Any]] = {}
    policies[f"fixed_{base_label}"] = evaluate_binary_policy(
        b_rows, e_rows, [False] * len(shared_qids), mode="pregen", ok_fn=is_ok, n_boot=n_boot
    )
    policies[f"fixed_{esc_label}"] = evaluate_binary_policy(
        b_rows, e_rows, [True] * len(shared_qids), mode="pregen", ok_fn=is_ok, n_boot=n_boot
    )

    oracle_dec = [(not is_ok(rb)) and bool(is_ok(re_)) for rb, re_ in zip(b_rows, e_rows)]
    policies["offline_router_oracle_2cfg"] = evaluate_binary_policy(
        b_rows, e_rows, oracle_dec, mode="pregen", ok_fn=is_ok, n_boot=n_boot
    )
    policies["offline_cascade_oracle_2cfg"] = evaluate_binary_policy(
        b_rows, e_rows, oracle_dec, mode="cascade", ok_fn=is_ok, n_boot=n_boot
    )

    # Zero-parameter causal heuristic cascades
    dec_hardfail = [f["is_truncated"] > 0.5 or f["is_parse_failure"] > 0.5 for f in feats]
    policies["cascade_hardfail_only"] = evaluate_binary_policy(
        b_rows, e_rows, dec_hardfail, mode="cascade", ok_fn=is_ok, n_boot=n_boot
    )

    dec_cot_uncapped = [(r.get("reasoning_tokens") or 0) >= 600 for r in b_rows]
    policies["cascade_cot_ge600_uncapped"] = evaluate_binary_policy(
        b_rows, e_rows, dec_cot_uncapped, mode="cascade", ok_fn=is_ok, n_boot=n_boot
    )

    dec_cot_capped = [400 <= (r.get("reasoning_tokens") or 0) <= 1600 for r in b_rows]
    policies["cascade_cot_400_1600_early_abort"] = evaluate_binary_policy(
        b_rows, e_rows, dec_cot_capped, mode="cascade", ok_fn=is_ok, n_boot=n_boot
    )

    # Out-of-fold learned utility routers/cascades
    model_specs = [
        ("pregen_ridge", PREGEN_FEATURES, "pregen", extra_pregen_s),
        ("cascade_ridge_cot_agnostic", COT_AGNOSTIC_FEATURES, "cascade", extra_cascade_s),
        ("cascade_ridge_all_observable", ALL_OBSERVABLE_FEATURES, "cascade", extra_cascade_s),
    ]
    full_fitted_models: dict[str, dict[str, Any]] = {}

    for prefix, feat_keys, mode, cost_vec in model_specs:
        X = _matrix_from_features(feats, feat_keys)
        full_mdl = fit_ridge_utility(X, y_delta, l2=l2)
        full_scores = predict_ridge_utility(X, full_mdl)
        weights_named = {
            "intercept": round(float(full_mdl["w"][0]), 5),
            **{k: round(float(full_mdl["w"][i + 1]), 5) for i, k in enumerate(feat_keys)},
        }
        full_fitted_models[prefix] = {
            "feature_keys": list(feat_keys),
            "mode": mode,
            "weights_standardized": weights_named,
            "thresholds_by_lambda": {},
            "_model": full_mdl,
        }

        for lam in lambdas:
            oof_dec = [False] * len(shared_qids)
            fold_thresholds: list[float] = []
            for f_idx in range(k):
                tr = [i for i, f in enumerate(folds) if f != f_idx]
                te = [i for i, f in enumerate(folds) if f == f_idx]
                if not tr or not te:
                    continue
                mdl = fit_ridge_utility(X[tr], y_delta[tr], l2=l2)
                tr_scores = predict_ridge_utility(X[tr], mdl)
                te_scores = predict_ridge_utility(X[te], mdl)
                thresh = _select_utility_threshold(tr_scores, y_delta[tr], cost_vec[tr], lam)
                fold_thresholds.append(thresh)
                for idx_local, i_global in enumerate(te):
                    oof_dec[i_global] = bool(te_scores[idx_local] > thresh)

            lam_tag = str(lam).replace(".", "p")
            pol_key = f"{prefix}_lam{lam_tag}"
            pol_res = evaluate_binary_policy(
                b_rows, e_rows, oof_dec, mode=mode, ok_fn=is_ok, n_boot=n_boot
            )
            pol_res["latency_penalty_lambda"] = lam
            pol_res["mean_fold_threshold"] = round(float(np.mean(fold_thresholds)), 5) if fold_thresholds else None

            # Oracle accuracy recovery ratio relative to fixed base
            base_acc = policies[f"fixed_{base_label}"]["accuracy"]
            ora_acc = policies["offline_router_oracle_2cfg"]["accuracy"]
            pol_res["oracle_acc_recovery_ratio"] = (
                round((pol_res["accuracy"] - base_acc) / (ora_acc - base_acc), 4)
                if (ora_acc - base_acc) > 1e-12
                else None
            )
            policies[pol_key] = pol_res

            # Also store full-fit threshold for frozen zero-shot transfer
            full_t = _select_utility_threshold(full_scores, y_delta, cost_vec, lam)
            full_fitted_models[prefix]["thresholds_by_lambda"][str(lam)] = round(float(full_t), 6)

    gap_diag = diagnose_oracle_gap(b_rows, e_rows, feats, ok_fn=is_ok)

    # Return JSON-safe summary + fitted artifacts for optional zero-shot transfer
    serializable_models = {
        k: {mk: mv for mk, mv in v.items() if mk != "_model"}
        for k, v in full_fitted_models.items()
    }
    return {
        "pair": f"{base_label} -> {esc_label}",
        "n_shared_questions": len(shared_qids),
        "cv_group_key": group_key,
        "cv_folds": k,
        "l2_regularization": l2,
        "policies": policies,
        "oracle_gap_diagnosis": gap_diag,
        "full_fit_feature_weights": serializable_models,
        "_fitted_runtime": full_fitted_models,
    }


def evaluate_zero_shot_transfer(
    fitted_runtime: dict[str, dict[str, Any]],
    base_rows_raw: list[dict[str, Any]],
    esc_rows_raw: list[dict[str, Any]],
    prompts_by_qid: dict[str, str],
    *,
    base_label: str = "base",
    esc_label: str = "escalated",
    ok_fn=None,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Apply frozen source-fitted models zero-shot (zero refitting) to an unseen target suite."""
    is_ok = ok_fn or (lambda r: _infer_outcome(r) == "correct")
    bm = {r["question_id"]: r for r in base_rows_raw}
    em = {r["question_id"]: r for r in esc_rows_raw}
    shared_qids = sorted(set(bm) & set(em))
    b_rows = [bm[q] for q in shared_qids]
    e_rows = [em[q] for q in shared_qids]
    feats = [extract_observable_features(r, prompts_by_qid.get(r["question_id"], "")) for r in b_rows]

    out: dict[str, Any] = {
        "target_pair": f"{base_label} -> {esc_label}",
        "n_shared_questions": len(shared_qids),
        "policies": {
            f"fixed_{base_label}": evaluate_binary_policy(
                b_rows, e_rows, [False] * len(shared_qids), mode="pregen", ok_fn=is_ok, n_boot=n_boot
            ),
            f"fixed_{esc_label}": evaluate_binary_policy(
                b_rows, e_rows, [True] * len(shared_qids), mode="pregen", ok_fn=is_ok, n_boot=n_boot
            ),
        },
    }

    for prefix, spec in fitted_runtime.items():
        keys = tuple(spec["feature_keys"])
        mode = spec["mode"]
        mdl = spec["_model"]
        X = _matrix_from_features(feats, keys)
        scores = predict_ridge_utility(X, mdl)
        for lam_str, thresh in spec["thresholds_by_lambda"].items():
            dec = [bool(s > float(thresh)) for s in scores]
            lam_tag = lam_str.replace(".", "p")
            out["policies"][f"zeroshot_{prefix}_lam{lam_tag}"] = evaluate_binary_policy(
                b_rows, e_rows, dec, mode=mode, ok_fn=is_ok, n_boot=n_boot
            )
    return out
