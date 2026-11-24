import math
import random
from collections import defaultdict


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    x = (len(sorted_vals) - 1) * p
    lo = math.floor(x)
    hi = math.ceil(x)
    return sorted_vals[lo] * (hi - x) + sorted_vals[hi] * (x - lo) if hi != lo else sorted_vals[lo]


def _infer_outcome(row: dict) -> str:
    if row.get("outcome"):
        return row["outcome"]
    if row.get("parse_status") == "backend_error" or row.get("finish_reason") == "error":
        return "backend_error"
    if row.get("finish_reason") == "length" or row.get("parse_status") == "truncated":
        return "truncated"
    if row.get("parse_status") == "parse_failure":
        return "parse_failure"
    return "correct" if row.get("correct") else "wrong_answer"


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    outcomes = {
        "correct": 0,
        "wrong_answer": 0,
        "parse_failure": 0,
        "truncated": 0,
        "backend_error": 0,
    }
    for row in rows:
        label = _infer_outcome(row)
        outcomes[label] = outcomes.get(label, 0) + 1

    correct = outcomes["correct"]
    e2e_times = sorted(float(r["e2e_ms"]) for r in rows if r.get("e2e_ms") is not None)
    ttft_times = sorted(float(r["ttft_ms"]) for r in rows if r.get("ttft_ms") is not None)
    vram_vals = [float(r["peak_vram_mb"]) for r in rows if r.get("peak_vram_mb") is not None]

    by = {}
    for key in ("modality", "task_type", "template_id"):
        vals = defaultdict(list)
        for row in rows:
            if row.get(key) is not None:
                vals[row[key]].append(_infer_outcome(row) == "correct")
        by[key] = {k: sum(v) / len(v) for k, v in sorted(vals.items())}

    ci = cluster_bootstrap_accuracy_ci(rows)
    is_mock = bool(rows) and any(bool(r.get("is_mock", False)) for r in rows)

    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "correct": correct,
        "wrong_answers": outcomes["wrong_answer"],
        "parse_failures": outcomes["parse_failure"],
        "truncated": outcomes["truncated"],
        "backend_errors": outcomes["backend_error"],
        "outcomes": outcomes,
        "e2e_p50_ms": _percentile(e2e_times, 0.5),
        "e2e_p90_ms": _percentile(e2e_times, 0.9),
        "ttft_p50_ms": _percentile(ttft_times, 0.5),
        "ttft_p90_ms": _percentile(ttft_times, 0.9),
        "peak_vram_max_mb": max(vram_vals) if vram_vals else None,
        "by_modality": by["modality"],
        "by_task_type": by["task_type"],
        "by_template": by["template_id"],
        "accuracy_cluster_bootstrap_95ci": ci,
        "mock": is_mock,
    }


def cluster_bootstrap_accuracy_ci(
    rows: list[dict], n_boot: int = 1000, seed: int = 20261123
) -> list[float | None]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(_infer_outcome(row) == "correct")
    if not groups:
        return [None, None]
    clusters = list(groups.values())
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        chosen = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        flat = [v for cluster in chosen for v in cluster]
        samples.append(sum(flat) / len(flat))
    samples.sort()
    return [samples[int(0.025 * (n_boot - 1))], samples[int(0.975 * (n_boot - 1))]]
