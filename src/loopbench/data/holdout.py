"""Deterministic one-source real-world holdout builder & validator (`ChartQA-80`).

Source: `HuggingFaceM4/ChartQA` (`split="test"`) pinned to exact revision SHA
`b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`.
Selects `n_images` (default 40) distinct real-world chart clusters that each have
2 human-authored (`human_or_machine == 0`) unambiguous finite numeric questions
(`80` total questions), preserving `image_id` cluster bootstrap pairing.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import urllib.request

from PIL import Image, PngImagePlugin

from .generate import pixel_sha256
from .schema import Question
from ..runner import resolve_image_path
from ..storage import atomic_write_text

CHARTQA_REPO = "HuggingFaceM4/ChartQA"
CHARTQA_REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"
CHARTQA_TEST_PARQUET_REL = "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"
CHARTQA_TEST_URL = (
    f"https://huggingface.co/datasets/{CHARTQA_REPO}/resolve/{CHARTQA_REVISION}/{CHARTQA_TEST_PARQUET_REL}"
)

_STRICT_NUMERIC_LABEL_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def _default_parquet_cache_path() -> Path:
    env_p = os.environ.get("LOOPBENCH_CHARTQA_PARQUET", "").strip()
    if env_p:
        return Path(env_p)
    return Path.home() / ".cache" / "loopbench" / f"chartqa_test_{CHARTQA_REVISION[:10]}.parquet"


def ensure_chartqa_parquet(cache_path: str | Path | None = None) -> Path:
    dest = Path(cache_path) if cache_path is not None else _default_parquet_cache_path()
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.tmp.{os.getpid()}"
    req = urllib.request.Request(CHARTQA_TEST_URL, headers={"User-Agent": "LoopBench/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    tmp.write_bytes(raw)
    os.replace(tmp, dest)
    return dest


def grade_chartqa_relaxed(predicted: float | None, expected: float, rel_tol: float = 0.05, abs_tol: float = 1e-2) -> bool:
    """Standard ChartQA relaxed numeric accuracy (within 5% relative tolerance, exact/abs_tol near zero)."""
    if predicted is None:
        return False
    exp = float(expected)
    diff = abs(float(predicted) - exp)
    if diff <= abs_tol:
        return True
    if abs(exp) > 1e-9 and (diff / abs(exp)) <= rel_tol + 1e-12:
        return True
    return False


def build_chartqa_holdout(
    out_dir: str | Path,
    *,
    seed: int = 20261125,
    n_images: int = 40,
    parquet_path: str | Path | None = None,
) -> dict:
    """Deterministically build `chartqa-{2*n_images}` from pinned ChartQA test split."""
    import pyarrow.parquet as pq

    if n_images <= 0:
        raise ValueError("n_images must be positive")

    pq_file = ensure_chartqa_parquet(parquet_path)
    table = pq.read_table(pq_file)
    records = table.to_pylist()

    by_img_hash: dict[str, dict] = {}
    for row_idx, rec in enumerate(records):
        # Strictly human-authored split only (0 == "human" in HuggingFaceM4/ChartQA)
        if int(rec.get("human_or_machine", -1)) != 0:
            continue
        labels = rec.get("label") or []
        if not isinstance(labels, list) or len(labels) != 1:
            continue
        lbl_str = str(labels[0]).strip()
        if not _STRICT_NUMERIC_LABEL_RE.match(lbl_str):
            continue
        ans_val = float(lbl_str)
        query_str = str(rec.get("query") or "").strip()
        if not query_str:
            continue

        img_field = rec.get("image") or {}
        img_bytes = img_field.get("bytes") if isinstance(img_field, dict) else None
        if not img_bytes:
            continue

        img_digest = hashlib.sha256(img_bytes).hexdigest()
        bucket = by_img_hash.setdefault(img_digest, {"img_bytes": img_bytes, "items": []})
        # Deduplicate identical queries on same chart if any
        if all(it["raw_query"] != query_str for it in bucket["items"]):
            bucket["items"].append({
                "source_row_idx": row_idx,
                "raw_query": query_str,
                "answer": ans_val,
                "raw_label": lbl_str,
            })

    eligible_clusters = [
        (digest, info)
        for digest, info in sorted(by_img_hash.items(), key=lambda kv: kv[0])
        if len(info["items"]) >= 2
    ]
    if len(eligible_clusters) < n_images:
        raise RuntimeError(
            f"Only {len(eligible_clusters)} eligible paired human-numeric ChartQA clusters found; requested {n_images}"
        )

    rng = random.Random(seed)
    chosen = list(eligible_clusters)
    rng.shuffle(chosen)
    chosen = chosen[:n_images]
    chosen.sort(key=lambda kv: kv[0])

    out = Path(out_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    questions: list[Question] = []

    for idx, (src_sha, info) in enumerate(chosen):
        image_id = f"chartqa-{idx:03d}"
        rel_path = f"images/{image_id}.png"
        full_path = out / rel_path

        with Image.open(io.BytesIO(info["img_bytes"])) as src_im:
            rgb = src_im.convert("RGB")
            width, height = rgb.size
            meta_png = PngImagePlugin.PngInfo()
            meta_png.add_text("Software", "LoopBench-ChartQA-Holdout")
            rgb.save(full_path, format="PNG", pnginfo=meta_png)

        pix_sha = pixel_sha256(full_path)
        pair_items = info["items"][:2]
        for pair_pos, q_item in enumerate(pair_items):
            slot = "q1" if pair_pos == 0 else "q2"
            prompt_text = f"{q_item['raw_query']} Answer with a single number."
            questions.append(
                Question(
                    question_id=f"{image_id}-{slot}",
                    image_id=image_id,
                    template_id="chartqa-human-numeric",
                    modality="chart",
                    task_type="holdout_human",
                    prompt=prompt_text,
                    answer=q_item["answer"],
                    unit="chart_value",
                    image_path=rel_path,
                    metadata={
                        "seed": seed,
                        "difficulty": "real_world",
                        "source_dataset": CHARTQA_REPO,
                        "source_revision": CHARTQA_REVISION,
                        "source_row_idx": q_item["source_row_idx"],
                        "source_png_sha256": src_sha,
                        "raw_query": q_item["raw_query"],
                        "raw_label": q_item["raw_label"],
                        "native_size": [width, height],
                        "pixel_sha256": pix_sha,
                    },
                )
            )

    manifest = {
        "schema_version": 2,
        "profile": "holdout_chartqa",
        "source_dataset": CHARTQA_REPO,
        "source_revision": CHARTQA_REVISION,
        "seed": seed,
        "eligible_paired_clusters_total": len(eligible_clusters),
        "image_count": n_images,
        "question_count": len(questions),
        "questions": [q.to_dict() for q in questions],
    }
    atomic_write_text(out / "questions.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def validate_holdout_dataset(
    dataset_dir: str | Path,
    expected_images: int | None = 40,
    expected_questions: int | None = 80,
) -> dict:
    root = Path(dataset_dir)
    data = json.loads((root / "questions.json").read_text(encoding="utf-8"))
    qs = data["questions"]
    errors: list[str] = []

    target_q = expected_questions if expected_questions is not None else data.get("question_count", 80)
    target_img = expected_images if expected_images is not None else data.get("image_count", 40)

    if len(qs) != target_q:
        errors.append(f"expected {target_q} questions, got {len(qs)}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in qs:
        groups[q["image_id"]].append(q)
    if len(groups) != target_img:
        errors.append(f"expected {target_img} images, got {len(groups)}")

    for img_id, items in groups.items():
        if len(items) != 2:
            errors.append(f"{img_id}: expected exactly 2 paired questions, got {len(items)}")

    sizes = []
    for q in qs:
        qid = q["question_id"]
        if Path(q["image_path"]).is_absolute():
            errors.append(f"{qid}: image_path must be portable relative path")
        p = resolve_image_path(root, q["image_path"])
        if not p.is_file():
            errors.append(f"missing holdout image: {p}")
            continue
        with Image.open(p) as im:
            if im.mode != "RGB":
                errors.append(f"{p}: expected RGB mode, got {im.mode}")
            w, h = im.size
            if w < 100 or h < 100:
                errors.append(f"{p}: suspiciously small chart image {im.size}")
            sizes.append((w, h))
            im.verify()
        if pixel_sha256(p) != q["metadata"]["pixel_sha256"]:
            errors.append(f"{p}: pixel_sha256 mismatch")

    if errors:
        raise ValueError("\n".join(errors))

    return {
        "valid": True,
        "profile": data.get("profile", "holdout_chartqa"),
        "source_dataset": data.get("source_dataset"),
        "source_revision": data.get("source_revision"),
        "images": len(groups),
        "questions": len(qs),
        "min_width": min(w for w, _ in sizes) if sizes else None,
        "max_width": max(w for w, _ in sizes) if sizes else None,
        "min_height": min(h for _, h in sizes) if sizes else None,
        "max_height": max(h for _, h in sizes) if sizes else None,
    }
