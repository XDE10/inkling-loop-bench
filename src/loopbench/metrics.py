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


def _mean(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


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
    tpot_times = sorted(float(r["tpot_ms"]) for r in rows if r.get("tpot_ms") is not None)
    vram_vals = sorted(float(r["peak_vram_mb"]) for r in rows if r.get("peak_vram_mb") is not None)

    reasoning_toks = sorted(
        float(r["reasoning_tokens"]) for r in rows if r.get("reasoning_tokens") is not None
    )
    answer_toks = [float(r["answer_tokens"]) for r in rows if r.get("answer_tokens") is not None]
    total_gen_toks = [
        float(r["total_generated_tokens"])
        for r in rows
        if r.get("total_generated_tokens") is not None
    ]
    visual_toks = sorted({int(r["visual_tokens"]) for r in rows if r.get("visual_tokens") is not None})
    eff_layers = sorted(
        {
            int(r["effective_layers_verified"])
            for r in rows
            if r.get("effective_layers_verified") is not None
        }
    )

    by = {}
    for key in ("modality", "task_type", "template_id"):
        vals = defaultdict(list)
        for row in rows:
            if row.get(key) is not None:
                vals[row[key]].append(_infer_outcome(row) == "correct")
        by[key] = {k: sum(v) / len(v) for k, v in sorted(vals.items())}

    mod_task_vals = defaultdict(list)
    for row in rows:
        if row.get("modality") is not None and row.get("task_type") is not None:
            cell = f"{row['modality']}/{row['task_type']}"
            mod_task_vals[cell].append(_infer_outcome(row) == "correct")
    by_modality_task = {k: sum(v) / len(v) for k, v in sorted(mod_task_vals.items())}

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
        "e2e_mean_ms": _mean(e2e_times),
        "ttft_p50_ms": _percentile(ttft_times, 0.5),
        "ttft_p90_ms": _percentile(ttft_times, 0.9),
        "ttft_mean_ms": _mean(ttft_times),
        "tpot_p50_ms": _percentile(tpot_times, 0.5),
        "tpot_p90_ms": _percentile(tpot_times, 0.9),
        "tpot_mean_ms": _mean(tpot_times),
        "peak_vram_p50_mb": _percentile(vram_vals, 0.5),
        "peak_vram_max_mb": max(vram_vals) if vram_vals else None,
        "reasoning_tokens_mean": _mean(reasoning_toks),
        "reasoning_tokens_p50": _percentile(reasoning_toks, 0.5),
        "reasoning_tokens_p90": _percentile(reasoning_toks, 0.9),
        "reasoning_tokens_max": int(max(reasoning_toks)) if reasoning_toks else None,
        "answer_tokens_mean": _mean(answer_toks),
        "total_generated_tokens_mean": _mean(total_gen_toks),
        "visual_tokens_unique": visual_toks,
        "effective_layers_unique": eff_layers,
        "by_modality": by["modality"],
        "by_task_type": by["task_type"],
        "by_modality_task": by_modality_task,
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


def paired_cluster_bootstrap_diff_ci(
    rows_base: list[dict],
    rows_cand: list[dict],
    n_boot: int = 2000,
    seed: int = 20261123,
) -> dict:
    """Paired cluster bootstrap CI for (accuracy_cand - accuracy_base) clustered by image_id."""
    base_by_q = {r["question_id"]: r for r in rows_base}
    cand_by_q = {r["question_id"]: r for r in rows_cand}
    shared_qids = sorted(set(base_by_q) & set(cand_by_q))
    if not shared_qids:
        return {
            "n_questions": 0,
            "n_clusters": 0,
            "base_accuracy": None,
            "cand_accuracy": None,
            "diff_accuracy": None,
            "diff_cluster_bootstrap_95ci": [None, None],
        }

    clusters_map: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for qid in shared_qids:
        rb = base_by_q[qid]
        rc = cand_by_q[qid]
        img_id = rb["image_id"]
        clusters_map[img_id].append(
            (_infer_outcome(rb) == "correct", _infer_outcome(rc) == "correct")
        )

    clusters = [clusters_map[k] for k in sorted(clusters_map)]
    flat_all = [pair for cl in clusters for pair in cl]
    base_acc = sum(b for b, _ in flat_all) / len(flat_all)
    cand_acc = sum(c for _, c in flat_all) / len(flat_all)

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        chosen = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        flat = [pair for cl in chosen for pair in cl]
        b_mean = sum(b for b, _ in flat) / len(flat)
        c_mean = sum(c for _, c in flat) / len(flat)
        diffs.append(c_mean - b_mean)
    diffs.sort()
    return {
        "n_questions": len(flat_all),
        "n_clusters": len(clusters),
        "base_accuracy": base_acc,
        "cand_accuracy": cand_acc,
        "diff_accuracy": cand_acc - base_acc,
        "diff_cluster_bootstrap_95ci": [
            diffs[int(0.025 * (n_boot - 1))],
            diffs[int(0.975 * (n_boot - 1))],
        ],
    }
