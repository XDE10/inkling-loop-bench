"""Independent blind multimodal evaluation & label audit on `chartqa-80` real-world holdout
using OpenCode Go `gemini-4-argon` across 5 sticky-cached API shards.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loopbench.argon_client import ArgonClient
from loopbench.data.holdout import (
    build_chartqa_holdout,
    grade_chartqa_relaxed,
    validate_holdout_dataset,
)
from loopbench.metrics import cluster_bootstrap_accuracy_ci
from loopbench.runner import resolve_image_path
from loopbench.storage import atomic_write_text


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


def _build_holdout_batch_parts(ds_dir: Path, image_groups: list[tuple[str, list[dict]]]) -> list:
    parts: list = [
        "Solve each of the following real-world chart images (2 human-authored numeric questions per chart: `q1` and `q2`) "
        "independently from the chart pixels and question text.\n"
        "Return ONLY a JSON object with schema:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "image_id": "<exact image_id>",\n'
        '      "q1_answer": <number>,\n'
        '      "q2_answer": <number>,\n'
        '      "visual_quality": "clear" | "crowded" | "ambiguous",\n'
        '      "note": "<short <=15 word note>"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    ]
    for img_id, q_pair in image_groups:
        q_map = {q["question_id"].split("-")[-1]: q for q in q_pair}
        img_path = resolve_image_path(ds_dir, q_pair[0]["image_path"])
        parts.append(f"\n--- REAL-WORLD CHART `{img_id}` ---")
        parts.append(img_path)
        parts.append(
            f"image_id: {img_id}\n"
            f"q1 ({q_map['q1']['question_id']}): {q_map['q1']['prompt']}\n"
            f"q2 ({q_map['q2']['question_id']}): {q_map['q2']['prompt']}\n"
        )
    return parts


def run_holdout_audit(
    out_dir: Path,
    archive_tar_out: Path | None = None,
    keys_file: str | None = None,
    seed: int = 20261125,
    n_images: int = 40,
    batch_images: int = 4,
    thinking_level: str = "MEDIUM",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = ArgonClient(keys_file=keys_file, session_namespace=f"chartqa{n_images*2}")
    num_shards = len(client.keys)

    with tempfile.TemporaryDirectory(prefix="loopbench_chartqa80_") as tmp_str:
        ds_dir = Path(tmp_str) / f"dataset_chartqa{n_images*2}"
        t0 = time.perf_counter()
        manifest = build_chartqa_holdout(ds_dir, seed=seed, n_images=n_images)
        valid_info = validate_holdout_dataset(ds_dir, expected_images=n_images, expected_questions=n_images * 2)
        gen_s = round(time.perf_counter() - t0, 2)

        # Save portable questions_chartqa80.json manifest (metadata only, ~45KB) + optional tarball for cloud sync
        atomic_write_text(
            out_dir / f"questions_chartqa{n_images*2}.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
        if archive_tar_out is not None:
            archive_tar_out.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_tar_out, "w:gz") as tf:
                tf.add(str(ds_dir), arcname=ds_dir.name)

        grouped_map: dict[str, list[dict]] = defaultdict(list)
        ordered_ids: list[str] = []
        for q in manifest["questions"]:
            if q["image_id"] not in grouped_map:
                ordered_ids.append(q["image_id"])
            grouped_map[q["image_id"]].append(q)

        batches = []
        for b_idx, i in enumerate(range(0, len(ordered_ids), batch_images)):
            chunk = [(cid, grouped_map[cid]) for cid in ordered_ids[i : i + batch_images]]
            batches.append((b_idx, chunk))

        shards: dict[int, list] = defaultdict(list)
        for b_idx, chunk in batches:
            shards[b_idx % num_shards].append((b_idx, chunk))

        batch_outputs: dict[int, dict] = {}

        def _worker(sid: int, s_batches: list) -> None:
            for b_idx, chunk in s_batches:
                parts = _build_holdout_batch_parts(ds_dir, chunk)
                resp = client.generate(
                    parts,
                    thinking_level=thinking_level,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    shard_id=sid,
                    timeout=150,
                )
                batch_outputs[b_idx] = _parse_json_lenient(resp["text"])
                u = resp["usage"]
                print(
                    f"[Argon Holdout shard={sid} batch={b_idx+1:02d}/{len(batches):02d}] "
                    f"local_hit={resp['from_local_cache']} cache_read={u['cache_read_tokens']}/{u['prompt_tokens']} "
                    f"dt={resp['elapsed_ms']/1000:.1f}s",
                    flush=True,
                )

        t_api = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_shards) as pool:
            futs = [pool.submit(_worker, sid, sb) for sid, sb in sorted(shards.items())]
            for f in futs:
                f.result()
        api_wall_s = round(time.perf_counter() - t_api, 2)

    pred_by_img: dict[str, dict] = {}
    for b_idx in sorted(batch_outputs):
        for item in batch_outputs[b_idx].get("items", []):
            pred_by_img[item["image_id"]] = item

    strict_rows: list[dict] = []
    relaxed_rows: list[dict] = []
    discrepancies: list[dict] = []

    for img_id in ordered_ids:
        p_item = pred_by_img.get(img_id, {})
        for q in grouped_map[img_id]:
            slot = q["question_id"].split("-")[-1]
            raw_pred = p_item.get(f"{slot}_answer")
            pred_val = float(raw_pred) if raw_pred is not None else None
            exp_val = float(q["answer"])
            strict_ok = pred_val is not None and abs(pred_val - exp_val) <= 1e-2
            relaxed_ok = grade_chartqa_relaxed(pred_val, exp_val, rel_tol=0.05, abs_tol=1e-2)
            base_row = {
                "config_id": f"argon-holdout-{thinking_level.lower()}",
                "question_id": q["question_id"],
                "image_id": img_id,
                "expected": exp_val,
                "predicted": pred_val,
                "raw_query": q["metadata"]["raw_query"],
                "visual_quality": p_item.get("visual_quality", "unknown"),
                "note": p_item.get("note", ""),
            }
            strict_rows.append({**base_row, "correct": strict_ok, "outcome": "correct" if strict_ok else "wrong_answer"})
            relaxed_rows.append({**base_row, "correct": relaxed_ok, "outcome": "correct" if relaxed_ok else "wrong_answer"})
            if not relaxed_ok:
                discrepancies.append({**base_row, "strict_ok": strict_ok, "relaxed_ok": relaxed_ok})

    n_q = len(strict_rows)
    strict_acc = sum(r["correct"] for r in strict_rows) / n_q
    relaxed_acc = sum(r["correct"] for r in relaxed_rows) / n_q
    summary = {
        "audit_model": "google/gemini-4-argon (via OpenCode Go)",
        "role": "External blind frontier reference & holdout QA ONLY (strictly separate from TML-d24-varloop-research pass curve)",
        "thinking_level": thinking_level.upper(),
        "holdout_validation": valid_info,
        "eligible_paired_clusters_total": manifest["eligible_paired_clusters_total"],
        "generation_wall_s": gen_s,
        "api_wall_s": api_wall_s,
        "n_images": n_images,
        "n_questions": n_q,
        "strict_accuracy_tol_0p01": round(strict_acc, 6),
        "strict_cluster_95ci": cluster_bootstrap_accuracy_ci(strict_rows, n_boot=2000, seed=seed),
        "relaxed_accuracy_5pct": round(relaxed_acc, 6),
        "relaxed_cluster_95ci": cluster_bootstrap_accuracy_ci(relaxed_rows, n_boot=2000, seed=seed),
        "cache_and_billing_telemetry": client.get_stats(),
        "relaxed_discrepancies": discrepancies,
    }
    atomic_write_text(
        out_dir / f"chartqa{n_images*2}_argon_audit.json",
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/argon_audit"))
    ap.add_argument("--archive-tar", default=None)
    ap.add_argument("--keys-file", default=None)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--thinking-level", default="MEDIUM")
    args = ap.parse_args()

    res = run_holdout_audit(
        out_dir=Path(args.out),
        archive_tar_out=Path(args.archive_tar) if args.archive_tar else None,
        keys_file=args.keys_file,
        seed=args.seed,
        n_images=args.n_images,
        thinking_level=args.thinking_level,
    )
    print("ARGON_HOLDOUT_AUDIT_SUMMARY:", json.dumps({
        "n_questions": res["n_questions"],
        "eligible_clusters": res["eligible_paired_clusters_total"],
        "strict_acc": res["strict_accuracy_tol_0p01"],
        "relaxed_acc_5pct": res["relaxed_accuracy_5pct"],
        "relaxed_ci95": res["relaxed_cluster_95ci"],
        "cache_telemetry": res["cache_and_billing_telemetry"],
        "discrepancies_count": len(res["relaxed_discrepancies"]),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
