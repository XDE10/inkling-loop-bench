"""Independent blind multimodal audit & frontier reference evaluation on `synth-240`
using OpenCode Go `gemini-4-argon` across 5 sticky-cached API shards.

- Never leaks ground-truth answers into the prompt (strict blind solving + visual QA).
- Groups 4 images (8 paired questions) per request = 30 requests across 5 parallel shards
  (6 sequential requests per key/session) to maximize upstream `cache_read` hits while
  persisting every response in SQLite (`~/.cache/loopbench/argon_cache.sqlite3`).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loopbench.argon_client import ArgonClient
from loopbench.data.generate import generate_dataset
from loopbench.data.validate import validate_dataset
from loopbench.metrics import cluster_bootstrap_accuracy_ci
from loopbench.runner import resolve_image_path
from loopbench.storage import atomic_write_text


def _build_batch_parts(ds_dir: Path, image_groups: list[tuple[str, list[dict]]]) -> list:
    parts: list = [
        "Solve each of the following 4 visual reasoning images (2 questions per image: `direct` and `multi_step`) "
        "independently from the provided image pixels and question text, and audit visual legibility.\n"
        "Return ONLY a JSON object with schema:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "image_id": "<exact image_id>",\n'
        '      "visual_quality": "clear" | "crowded" | "ambiguous",\n'
        '      "direct_answer": <number>,\n'
        '      "multi_step_answer": <number>,\n'
        '      "note": "<short <=15 word verification note>"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    ]
    for img_id, q_pair in image_groups:
        q_by_task = {q["task_type"]: q for q in q_pair}
        img_path = resolve_image_path(ds_dir, q_pair[0]["image_path"])
        parts.append(f"\n--- IMAGE `{img_id}` (modality={q_pair[0]['modality']}) ---")
        parts.append(img_path)
        parts.append(
            f"image_id: {img_id}\n"
            f"direct question ({q_by_task['direct']['question_id']}): {q_by_task['direct']['prompt']}\n"
            f"multi_step question ({q_by_task['multi_step']['question_id']}): {q_by_task['multi_step']['prompt']}\n"
        )
    return parts


def _parse_json_lenient(raw_text: str) -> dict:
    txt = raw_text.strip()
    if txt.startswith("```"):
        lines = txt.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        txt = "\n".join(lines).strip()
    return json.loads(txt)


def run_synth_audit(
    out_dir: Path,
    keys_file: str | None = None,
    seed: int = 20261124,
    n_per_type: int = 60,
    batch_images: int = 4,
    thinking_level: str = "MEDIUM",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = ArgonClient(keys_file=keys_file, session_namespace=f"synth{n_per_type*4}")
    num_shards = len(client.keys)

    with tempfile.TemporaryDirectory(prefix="loopbench_argon_synth_") as tmp_ds_str:
        ds_dir = Path(tmp_ds_str)
        t0 = time.perf_counter()
        generate_dataset(
            ds_dir,
            seed=seed,
            n_per_type=n_per_type,
            profile="synth",
            seed_mode="per_item",
        )
        valid_info = validate_dataset(
            ds_dir,
            expected_images=n_per_type * 2,
            expected_questions=n_per_type * 4,
        )
        gen_s = round(time.perf_counter() - t0, 2)

        manifest = json.loads((ds_dir / "questions.json").read_text(encoding="utf-8"))
        grouped_map: dict[str, list[dict]] = defaultdict(list)
        ordered_img_ids: list[str] = []
        for q in manifest["questions"]:
            if q["image_id"] not in grouped_map:
                ordered_img_ids.append(q["image_id"])
            grouped_map[q["image_id"]].append(q)

        batches: list[tuple[int, list[tuple[str, list[dict]]]]] = []
        for b_idx, i in enumerate(range(0, len(ordered_img_ids), batch_images)):
            chunk_ids = ordered_img_ids[i : i + batch_images]
            batches.append((b_idx, [(cid, grouped_map[cid]) for cid in chunk_ids]))

        shards: dict[int, list[tuple[int, list[tuple[str, list[dict]]]]]] = defaultdict(list)
        for b_idx, chunk in batches:
            shards[b_idx % num_shards].append((b_idx, chunk))

        batch_outputs: dict[int, dict] = {}

        def _run_shard(shard_id: int, shard_batches: list[tuple[int, list[tuple[str, list[dict]]]]]) -> None:
            for b_idx, chunk in shard_batches:
                parts = _build_batch_parts(ds_dir, chunk)
                resp = client.generate(
                    parts,
                    thinking_level=thinking_level,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    shard_id=shard_id,
                    timeout=150,
                )
                parsed = _parse_json_lenient(resp["text"])
                batch_outputs[b_idx] = {
                    "parsed": parsed,
                    "meta": {
                        "from_local_cache": resp["from_local_cache"],
                        "elapsed_ms": resp["elapsed_ms"],
                        "key_index": resp["key_index"],
                        "usage": resp["usage"],
                    },
                }
                u = resp["usage"]
                print(
                    f"[Argon shard={shard_id} batch={b_idx+1:02d}/{len(batches):02d}] "
                    f"local_hit={resp['from_local_cache']} cache_read={u['cache_read_tokens']}/{u['prompt_tokens']} "
                    f"dt={resp['elapsed_ms']/1000:.1f}s",
                    flush=True,
                )

        t_api = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_shards) as pool:
            futs = [pool.submit(_run_shard, sid, s_batches) for sid, s_batches in sorted(shards.items())]
            for f in futs:
                f.result()
        api_wall_s = round(time.perf_counter() - t_api, 2)

    pred_by_img: dict[str, dict] = {}
    for b_idx in sorted(batch_outputs):
        for item in batch_outputs[b_idx]["parsed"].get("items", []):
            pred_by_img[item["image_id"]] = item

    question_rows: list[dict] = []
    visual_ratings = Counter()
    disagreements: list[dict] = []

    for img_id in ordered_img_ids:
        p_item = pred_by_img.get(img_id, {})
        vq = str(p_item.get("visual_quality", "missing")).lower()
        visual_ratings[vq] += 1
        for q in grouped_map[img_id]:
            field = "direct_answer" if q["task_type"] == "direct" else "multi_step_answer"
            raw_pred = p_item.get(field)
            pred_val = float(raw_pred) if raw_pred is not None else None
            exp_val = float(q["answer"])
            is_ok = pred_val is not None and abs(pred_val - exp_val) <= 1e-5
            diff_tier = q.get("metadata", {}).get("difficulty", "standard")
            row = {
                "config_id": f"argon-{thinking_level.lower()}",
                "question_id": q["question_id"],
                "image_id": img_id,
                "template_id": q["template_id"],
                "modality": q["modality"],
                "difficulty": diff_tier,
                "task_type": q["task_type"],
                "expected": exp_val,
                "predicted": pred_val,
                "correct": is_ok,
                "outcome": "correct" if is_ok else ("parse_failure" if pred_val is None else "wrong_answer"),
                "visual_quality": vq,
                "note": p_item.get("note", ""),
            }
            question_rows.append(row)
            if not is_ok or vq != "clear":
                disagreements.append(row)

    total_q = len(question_rows)
    correct_q = sum(r["correct"] for r in question_rows)
    ci95 = cluster_bootstrap_accuracy_ci(question_rows, n_boot=2000, seed=seed)

    by_cell = defaultdict(list)
    by_template = defaultdict(list)
    for r in question_rows:
        by_cell[f"{r['modality']}/{r['difficulty']}/{r['task_type']}"].append(r["correct"])
        by_template[r["template_id"]].append(r["correct"])

    stats = client.get_stats()
    summary = {
        "audit_model": "google/gemini-4-argon (via OpenCode Go)",
        "role": "External blind visual QA & frontier reference ONLY (strictly separate from TML-d24-varloop-research pass curve)",
        "thinking_level": thinking_level.upper(),
        "dataset_validation": valid_info,
        "generation_wall_s": gen_s,
        "api_wall_s": api_wall_s,
        "n_images": len(ordered_img_ids),
        "n_questions": total_q,
        "blind_agreement_accuracy": round(correct_q / total_q, 6),
        "blind_agreement_correct": correct_q,
        "cluster_bootstrap_95ci": ci95,
        "visual_quality_counts": dict(visual_ratings),
        "by_modality_difficulty_task": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(by_cell.items())
        },
        "by_template": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(by_template.items())
        },
        "cache_and_billing_telemetry": stats,
        "flagged_or_disagreed_items": disagreements,
    }
    atomic_write_text(out_dir / "synth240_argon_audit.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/argon_audit"))
    ap.add_argument("--keys-file", default=None)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--n-per-type", type=int, default=60)
    ap.add_argument("--batch-images", type=int, default=4)
    ap.add_argument("--thinking-level", default="MEDIUM")
    args = ap.parse_args()

    res = run_synth_audit(
        out_dir=Path(args.out),
        keys_file=args.keys_file,
        seed=args.seed,
        n_per_type=args.n_per_type,
        batch_images=args.batch_images,
        thinking_level=args.thinking_level,
    )
    print("ARGON_SYNTH_AUDIT_SUMMARY:", json.dumps({
        "n_questions": res["n_questions"],
        "blind_agreement_accuracy": res["blind_agreement_accuracy"],
        "ci95": res["cluster_bootstrap_95ci"],
        "visual_quality_counts": res["visual_quality_counts"],
        "cache_telemetry": res["cache_and_billing_telemetry"],
        "flagged_count": len(res["flagged_or_disagreed_items"]),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
