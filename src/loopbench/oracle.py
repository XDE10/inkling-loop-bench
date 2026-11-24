"""Offline Oracle & Pareto frontier analysis across pass depths and reasoning efforts.

Computes two complementary upper bounds from completed evaluation traces:
1. `router_oracle`: Ideal zero-overhead router that assigns each question `q` to the
   minimum-latency configuration `c` that solves `q` (or fastest `c` if none solve `q`).
2. `cascade_oracle`: Ideal halting oracle over an ordered escalation schedule
   `c_1 -> c_2 -> ... -> c_k` that stops at the earliest correct configuration `c_j`
   while paying cumulative wall/E2E latency `sum_{i <= j} e2e_ms(c_i, q)`.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

from .metrics import _infer_outcome, _percentile, paired_cluster_bootstrap_diff_ci


def compute_pareto_frontier(points: list[dict[str, Any]], cost_key: str = "e2e_p50_ms", acc_key: str = "accuracy") -> list[str]:
    """Return config labels on the non-dominated accuracy-vs-cost Pareto frontier."""
    valid = [p for p in points if p.get(cost_key) is not None and p.get(acc_key) is not None]
    valid.sort(key=lambda p: (float(p[cost_key]), -float(p[acc_key])))
    frontier: list[str] = []
    best_acc = -1.0
    for pt in valid:
        acc = float(pt[acc_key])
        if acc > best_acc + 1e-12:
            frontier.append(str(pt["config_label"]))
            best_acc = acc
    return frontier


def analyze_offline_oracle(
    config_rows_map: dict[str, list[dict]],
    escalation_order: list[str] | None = None,
    correct_fn=None,
    seed: int = 20261123,
) -> dict[str, Any]:
    """Analyze fixed policies vs offline router oracle and progressive cascade oracle."""
    if not config_rows_map:
        raise ValueError("config_rows_map must not be empty")

    is_ok_fn = correct_fn or (lambda r: _infer_outcome(r) == "correct")
    labels = list(escalation_order) if escalation_order is not None else list(config_rows_map.keys())

    by_cfg_q: dict[str, dict[str, dict]] = {
        lbl: {r["question_id"]: r for r in config_rows_map[lbl]}
        for lbl in labels
    }
    shared_qids = sorted(set.intersection(*(set(m.keys()) for m in by_cfg_q.values())))
    if not shared_qids:
        raise ValueError("No shared question_ids across configurations for Oracle analysis")

    fixed_summaries: list[dict[str, Any]] = []
    for lbl in labels:
        rows = [by_cfg_q[lbl][qid] for qid in shared_qids]
        ok_cnt = sum(1 for r in rows if is_ok_fn(r))
        e2es = sorted(float(r.get("e2e_ms") or 0.0) for r in rows)
        rtoks = [float(r.get("reasoning_tokens") or 0.0) for r in rows]
        fixed_summaries.append({
            "config_label": lbl,
            "n": len(rows),
            "accuracy": ok_cnt / len(rows),
            "correct": ok_cnt,
            "e2e_p50_ms": _percentile(e2es, 0.5),
            "e2e_mean_ms": sum(e2es) / len(e2es),
            "total_gpu_s": round(sum(e2es) / 1000.0, 2),
            "reasoning_tokens_mean": round(sum(rtoks) / len(rtoks), 2),
        })

    # Order escalation schedule by empirical mean latency if not explicitly provided
    if escalation_order is None:
        labels = [
            x["config_label"]
            for x in sorted(fixed_summaries, key=lambda d: float(d["e2e_mean_ms"] or 0.0))
        ]

    fastest_lbl = min(fixed_summaries, key=lambda d: float(d["e2e_mean_ms"] or math.inf))["config_label"]
    best_fixed = max(fixed_summaries, key=lambda d: (float(d["accuracy"]), -float(d["e2e_mean_ms"] or 0.0)))
    best_fixed_lbl = best_fixed["config_label"]

    router_rows: list[dict] = []
    cascade_rows: list[dict] = []
    selected_counts: dict[str, int] = defaultdict(int)
    unsolved_count = 0

    for qid in shared_qids:
        candidates = [(lbl, by_cfg_q[lbl][qid]) for lbl in labels]
        solving = [(lbl, r) for lbl, r in candidates if is_ok_fn(r)]

        # 1. Zero-overhead minimum-latency router oracle
        if solving:
            chosen_lbl, chosen_row = min(solving, key=lambda pair: float(pair[1].get("e2e_ms") or 0.0))
            oracle_ok = True
        else:
            unsolved_count += 1
            chosen_lbl = fastest_lbl
            chosen_row = by_cfg_q[fastest_lbl][qid]
            oracle_ok = False
        selected_counts[chosen_lbl] += 1
        router_rows.append({
            "question_id": qid,
            "image_id": chosen_row["image_id"],
            "outcome": "correct" if oracle_ok else "wrong_answer",
            "correct": oracle_ok,
            "selected_config": chosen_lbl,
            "e2e_ms": float(chosen_row.get("e2e_ms") or 0.0),
        })

        # 2. Progressive cascade halting oracle along `labels`
        cum_ms = 0.0
        casc_ok = False
        halt_lbl = labels[-1]
        for lbl, r in candidates:
            cum_ms += float(r.get("e2e_ms") or 0.0)
            if is_ok_fn(r):
                casc_ok = True
                halt_lbl = lbl
                break
        cascade_rows.append({
            "question_id": qid,
            "image_id": candidates[0][1]["image_id"],
            "outcome": "correct" if casc_ok else "wrong_answer",
            "correct": casc_ok,
            "halted_at": halt_lbl,
            "e2e_ms": cum_ms,
        })

    router_e2es = sorted(r["e2e_ms"] for r in router_rows)
    casc_e2es = sorted(r["e2e_ms"] for r in cascade_rows)
    oracle_acc = sum(r["correct"] for r in router_rows) / len(router_rows)

    # Normalize best_fixed rows to match custom `is_ok_fn` if provided
    best_fixed_norm_rows = [
        {
            "question_id": qid,
            "image_id": by_cfg_q[best_fixed_lbl][qid]["image_id"],
            "outcome": "correct" if is_ok_fn(by_cfg_q[best_fixed_lbl][qid]) else "wrong_answer",
            "correct": is_ok_fn(by_cfg_q[best_fixed_lbl][qid]),
        }
        for qid in shared_qids
    ]
    diff_vs_best = paired_cluster_bootstrap_diff_ci(best_fixed_norm_rows, router_rows, seed=seed)

    best_mean_ms = float(best_fixed["e2e_mean_ms"] or 1.0)
    router_mean_ms = sum(router_e2es) / len(router_e2es)
    casc_mean_ms = sum(casc_e2es) / len(casc_e2es)

    return {
        "n_shared_questions": len(shared_qids),
        "escalation_order": labels,
        "fixed_policies": fixed_summaries,
        "pareto_frontier_p50": compute_pareto_frontier(fixed_summaries, cost_key="e2e_p50_ms"),
        "pareto_frontier_mean": compute_pareto_frontier(fixed_summaries, cost_key="e2e_mean_ms"),
        "best_fixed_policy": best_fixed,
        "router_oracle": {
            "accuracy": oracle_acc,
            "correct": sum(r["correct"] for r in router_rows),
            "unsolved_by_all": unsolved_count,
            "e2e_p50_ms": _percentile(router_e2es, 0.5),
            "e2e_mean_ms": router_mean_ms,
            "total_gpu_s": round(sum(router_e2es) / 1000.0, 2),
            "mean_speedup_vs_best_fixed": round(best_mean_ms / router_mean_ms, 4) if router_mean_ms > 0 else None,
            "selection_distribution": {k: selected_counts[k] for k in labels if selected_counts[k] > 0},
            "paired_gain_vs_best_fixed": diff_vs_best,
        },
        "cascade_oracle": {
            "accuracy": oracle_acc,
            "e2e_p50_ms": _percentile(casc_e2es, 0.5),
            "e2e_mean_ms": casc_mean_ms,
            "total_gpu_s": round(sum(casc_e2es) / 1000.0, 2),
            "mean_speedup_vs_best_fixed": round(best_mean_ms / casc_mean_ms, 4) if casc_mean_ms > 0 else None,
        },
    }
