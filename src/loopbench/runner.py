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


def run(config: dict, dataset_dir, output_path, backend=None, progress_callback=None) -> dict:
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

    count = 0
    total_q = len(questions)
    for idx, q in enumerate(questions, 1):
        key = (config_id, q["question_id"])
        if key in done:
            continue

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
        start = time.perf_counter()
        try:
            response = active_backend.generate(q_input, validated)
            finish_reason = response.get("finish_reason", "stop")
            scored = grade(
                response.get("text", ""),
                q["answer"],
                finish_reason=finish_reason,
            )
            row = {
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
            }
        except Exception as e:
            row = {
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
                "error": repr(e),
                "is_mock": validated["backend"] == "mock",
            }
        row["wall_ms"] = (time.perf_counter() - start) * 1000.0
        append_jsonl(out_file, row)
        done.add(key)
        count += 1
        if progress_callback is not None:
            progress_callback(idx, total_q, row)

    return {
        "config_id": config_id,
        "new_rows": count,
        "total_questions": total_q,
        "image_mode": image_mode,
    }
