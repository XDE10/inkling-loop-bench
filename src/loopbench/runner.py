import hashlib
import json
import time
from pathlib import Path
from PIL import Image
from .backends import create_backend
from .config import validate_config_dict
from .grading import grade
from .storage import append_jsonl, load_completed

OPTIONAL = {
    "reasoning_tokens": None,
    "answer_tokens": None,
    "visual_tokens": None,
    "image_grid_thw": None,
    "ttft_ms": None,
    "tpot_ms": None,
    "e2e_ms": None,
    "peak_vram_mb": None,
    "speculative_decoding": None,
    "loop_share_cache": None,
}


def resolve_image_path(dataset_dir: Path, raw_image_path: str) -> Path:
    candidate = Path(raw_image_path)
    if not candidate.is_absolute():
        return (dataset_dir / candidate).resolve()
    if candidate.exists():
        return candidate
    return (dataset_dir / "images" / candidate.name).resolve()


def _ensure_blank_image(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    blank_path = cache_dir / "_blank_768x768.png"
    if not blank_path.is_file():
        Image.new("RGB", (768, 768), (255, 255, 255)).save(blank_path)
    return blank_path


def _build_mismatched_map(questions: list[dict], dataset_dir: Path) -> dict[str, Path]:
    by_modality: dict[str, list[tuple[str, Path]]] = {"chart": [], "geometry": []}
    seen = set()
    for q in questions:
        img_id = q["image_id"]
        if img_id in seen:
            continue
        seen.add(img_id)
        by_modality.setdefault(q["modality"], []).append(
            (img_id, resolve_image_path(dataset_dir, q["image_path"]))
        )
    mapping: dict[str, Path] = {}
    charts = by_modality.get("chart", [])
    geoms = by_modality.get("geometry", [])
    if charts and geoms:
        for idx, (img_id, _) in enumerate(charts):
            mapping[img_id] = geoms[idx % len(geoms)][1]
        for idx, (img_id, _) in enumerate(geoms):
            mapping[img_id] = charts[idx % len(charts)][1]
    else:
        all_imgs = charts + geoms
        for idx, (img_id, _) in enumerate(all_imgs):
            mapping[img_id] = all_imgs[(idx + 1) % len(all_imgs)][1]
    return mapping


def compute_config_id(config: dict) -> str:
    canonical = dict(config)
    if canonical.get("image_mode", "correct") == "correct":
        canonical.pop("image_mode", None)
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def run(config: dict, dataset_dir, output_path, backend=None, progress_callback=None, batch_size: int | None = None) -> dict:
    validated = validate_config_dict(config)
    ds_root = Path(dataset_dir)
    out_file = Path(output_path)
    data = json.loads((ds_root / "questions.json").read_text(encoding="utf-8"))
    questions = data["questions"]

    config_id = compute_config_id(validated)
    done = load_completed(out_file)
    active_backend = backend if backend is not None else create_backend(validated)
    image_mode = validated.get("image_mode", "correct")

    blank_path = _ensure_blank_image(out_file.parent) if image_mode == "blank" else None
    mismatched_map = (
        _build_mismatched_map(questions, ds_root) if image_mode == "mismatched" else {}
    )

    eff_bs = int(batch_size) if batch_size is not None else int(validated.get("concurrency", 1))
    use_batched = eff_bs > 1 and hasattr(active_backend, "generate_batch")

    def _prepare_q(q: dict) -> tuple[dict, str]:
        if image_mode == "blank":
            effective_img = blank_path
        elif image_mode == "mismatched":
            effective_img = mismatched_map[q["image_id"]]
        else:
            effective_img = resolve_image_path(ds_root, q["image_path"])
        q_input = {**q, "resolved_image_path": str(effective_img)}
        difficulty = q.get("metadata", {}).get("difficulty") or (
            "hard" if "-v2-" in str(q.get("template_id") or "") else "standard"
        )
        return q_input, difficulty

    def _build_ok_row(q: dict, difficulty: str, response: dict, wall_ms: float) -> dict:
        finish_reason = response.get("finish_reason", "stop")
        scored = grade(
            response.get("text", ""),
            q["answer"],
            finish_reason=finish_reason,
        )
        return {
            **OPTIONAL,
            "config_id": config_id,
            "loop_num_passes": int(validated["loop_num_passes"]),
            "reasoning_effort": validated["reasoning_effort"],
            "question_id": q["question_id"],
            "image_id": q["image_id"],
            "template_id": q.get("template_id"),
            "difficulty": difficulty,
            "modality": q["modality"],
            "task_type": q["task_type"],
            "image_mode": image_mode,
            "expected": q["answer"],
            **response,
            **scored,
            "error": None if finish_reason != "error" else response.get("error"),
            "wall_ms": wall_ms,
        }

    def _build_err_row(q: dict, difficulty: str, exc: Exception, wall_ms: float) -> dict:
        return {
            **OPTIONAL,
            "config_id": config_id,
            "loop_num_passes": int(validated["loop_num_passes"]),
            "reasoning_effort": validated["reasoning_effort"],
            "question_id": q["question_id"],
            "image_id": q["image_id"],
            "template_id": q.get("template_id"),
            "difficulty": difficulty,
            "modality": q["modality"],
            "task_type": q["task_type"],
            "image_mode": image_mode,
            "expected": q["answer"],
            "text": "",
            "finish_reason": "error",
            "correct": False,
            "parse_status": "backend_error",
            "outcome": "backend_error",
            "predicted": None,
            "error": repr(exc),
            "is_mock": validated["backend"] == "mock",
            "wall_ms": wall_ms,
        }

    count = 0
    total_q = len(questions)
    pending: list[tuple[int, dict]] = [
        (idx, q) for idx, q in enumerate(questions, 1) if (config_id, q["question_id"]) not in done
    ]

    if use_batched:
        for start_i in range(0, len(pending), eff_bs):
            chunk = pending[start_i : start_i + eff_bs]
            prepped = [_prepare_q(q) for _, q in chunk]
            q_inputs = [p[0] for p in prepped]
            t0 = time.perf_counter()
            try:
                responses = active_backend.generate_batch(q_inputs, validated)
                per_item_wall = ((time.perf_counter() - t0) * 1000.0) / len(chunk)
                for (idx, q), (_, diff), resp in zip(chunk, prepped, responses):
                    row = _build_ok_row(q, diff, resp, per_item_wall)
                    append_jsonl(out_file, row)
                    done.add((config_id, q["question_id"]))
                    count += 1
                    if progress_callback is not None:
                        progress_callback(idx, total_q, row)
            except Exception:
                # Graceful single-item fallback if a mixed-resolution batch fails
                for (idx, q), (q_input, diff) in zip(chunk, prepped):
                    ts = time.perf_counter()
                    try:
                        resp = active_backend.generate(q_input, validated)
                        row = _build_ok_row(q, diff, resp, (time.perf_counter() - ts) * 1000.0)
                    except Exception as exc:
                        row = _build_err_row(q, diff, exc, (time.perf_counter() - ts) * 1000.0)
                    append_jsonl(out_file, row)
                    done.add((config_id, q["question_id"]))
                    count += 1
                    if progress_callback is not None:
                        progress_callback(idx, total_q, row)
    else:
        for idx, q in pending:
            q_input, difficulty = _prepare_q(q)
            start = time.perf_counter()
            try:
                response = active_backend.generate(q_input, validated)
                row = _build_ok_row(q, difficulty, response, (time.perf_counter() - start) * 1000.0)
            except Exception as e:
                row = _build_err_row(q, difficulty, e, (time.perf_counter() - start) * 1000.0)
            append_jsonl(out_file, row)
            done.add((config_id, q["question_id"]))
            count += 1
            if progress_callback is not None:
                progress_callback(idx, total_q, row)

    return {
        "config_id": config_id,
        "new_rows": count,
        "total_questions": total_q,
        "image_mode": image_mode,
        "batch_size": eff_bs,
    }
