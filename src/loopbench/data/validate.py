import json
import math
import re
from pathlib import Path
from PIL import Image
from .generate import pixel_sha256
from ..runner import resolve_image_path


def _angle_deg(v: list[float], p1: list[float], p2: list[float]) -> float:
    u1x, u1y = p1[0] - v[0], p1[1] - v[1]
    u2x, u2y = p2[0] - v[0], p2[1] - v[1]
    denom = math.hypot(u1x, u1y) * math.hypot(u2x, u2y)
    dot = max(-1.0, min(1.0, (u1x * u2x + u1y * u2y) / denom))
    return math.degrees(math.acos(dot))


def _recompute_chart(task_type: str, meta: dict, errors: list[str], qid: str) -> float | None:
    sub = meta.get("subtype")
    if sub == "bar4_grid5":
        vals = meta["values"]
        if len(set(vals.values())) != 4 or any(v % 5 != 0 for v in vals.values()):
            errors.append(f"{qid}: bar4_grid5 invariant violated")
        return float(vals[meta["target_region"]] if task_type == "direct" else max(vals.values()) - min(vals.values()))
    if sub == "barh4_grid5":
        vals = meta["values"]
        p0, p1 = meta["sum_pair"]
        if len(set(vals.values())) != 4 or any(v % 5 != 0 for v in vals.values()):
            errors.append(f"{qid}: barh4_grid5 invariant violated")
        return float(max(vals.values()) if task_type == "direct" else vals[p0] + vals[p1])
    if sub == "line5_grid5":
        vals = meta["values"]
        p0, p1 = meta["sum_pair"]
        if len(set(vals.values())) != 5 or any(v % 5 != 0 for v in vals.values()):
            errors.append(f"{qid}: line5_grid5 invariant violated")
        return float(vals[meta["target_month"]] if task_type == "direct" else vals[p0] + vals[p1])
    if sub == "grouped_bar_labeled":
        series = meta["series"]
        return float(
            series[meta["target_series"]][meta["target_plant"]]
            if task_type == "direct"
            else sum(series[meta["sum_series"]].values())
        )
    if sub == "stacked_bar_labeled":
        segs = meta["segments"]
        d0, d1 = meta["sum_divisions"]
        return float(
            segs[meta["target_segment"]][meta["target_division"]]
            if task_type == "direct"
            else (segs["Core"][d0] + segs["Addon"][d0] + segs["Core"][d1] + segs["Addon"][d1])
        )
    if sub == "threshold_bar5_grid5":
        vals = meta["values"]
        target = meta["target"]
        above = [v for v in vals.values() if v > target]
        below = [v for v in vals.values() if v < target]
        if len(above) < 2 or len(below) < 2 or any(v == target or v % 5 != 0 for v in vals.values()):
            errors.append(f"{qid}: threshold_bar5_grid5 invariant violated")
        return float(target if task_type == "direct" else sum(v - target for v in above))
    errors.append(f"{qid}: unknown chart subtype {sub}")
    return None


def _recompute_geometry(task_type: str, meta: dict, errors: list[str], qid: str) -> float | None:
    sub = meta.get("subtype")
    verts = meta.get("vertices", {})
    if sub == "scalene_triangle":
        angs = meta["angles"]
        k1, k2 = meta["known"]
        target = meta["target"]
        if abs(_angle_deg(verts["A"], verts["B"], verts["C"]) - angs["A"]) > 1e-5:
            errors.append(f"{qid}: vertex A Euclidean angle mismatch")
        if abs(_angle_deg(verts["B"], verts["A"], verts["C"]) - angs["B"]) > 1e-5:
            errors.append(f"{qid}: vertex B Euclidean angle mismatch")
        if abs(_angle_deg(verts["C"], verts["A"], verts["B"]) - angs["C"]) > 1e-5:
            errors.append(f"{qid}: vertex C Euclidean angle mismatch")
        return float(angs[k1] if task_type == "direct" else 180 - angs[k1] - angs[k2])
    if sub == "right_triangle_symbol":
        angs = meta["angles"]
        if abs(_angle_deg(verts["B"], verts["A"], verts["C"]) - 90.0) > 1e-5:
            errors.append(f"{qid}: right-angle vertex B mismatch")
        if abs(_angle_deg(verts["A"], verts["B"], verts["C"]) - angs["A"]) > 1e-5:
            errors.append(f"{qid}: acute angle A mismatch")
        known = meta["known_acute"]
        return float(angs[known] if task_type == "direct" else 90 - angs[known])
    if sub == "isosceles_ticks":
        angs = meta["angles"]
        len_ac = math.hypot(verts["C"][0] - verts["A"][0], verts["C"][1] - verts["A"][1])
        len_bc = math.hypot(verts["C"][0] - verts["B"][0], verts["C"][1] - verts["B"][1])
        if abs(len_ac - len_bc) > 1e-6:
            errors.append(f"{qid}: isosceles side lengths AC and BC differ")
        if abs(_angle_deg(verts["C"], verts["A"], verts["B"]) - angs["C"]) > 1e-5:
            errors.append(f"{qid}: apex angle C mismatch")
        gv, tv = meta["given_vertex"], meta["target_vertex"]
        return float(angs[gv] if task_type == "direct" else angs[tv])
    if sub == "exterior_angle":
        if abs(_angle_deg(verts["A"], verts["B"], verts["C"]) - meta["angle_A"]) > 1e-5:
            errors.append(f"{qid}: angle A mismatch")
        if abs(_angle_deg(verts["C"], verts["A"], verts["B"]) - meta["angle_C"]) > 1e-5:
            errors.append(f"{qid}: angle C mismatch")
        if abs(_angle_deg(verts["B"], verts["D"], verts["C"]) - meta["exterior_CBD"]) > 1e-5:
            errors.append(f"{qid}: exterior angle CBD mismatch")
        return float(meta["angle_C"] if task_type == "direct" else meta["angle_A"] + meta["angle_C"])
    if sub == "parallel_line_apex":
        if abs(verts["P"][1] - verts["Q"][1]) > 1e-9 or abs(verts["A"][1] - verts["B"][1]) > 1e-9:
            errors.append(f"{qid}: lines PQ and AB are not parallel")
        if abs(_angle_deg(verts["C"], verts["P"], verts["A"]) - meta["angle_PCA"]) > 1e-5:
            errors.append(f"{qid}: angle PCA mismatch")
        if abs(_angle_deg(verts["C"], verts["A"], verts["B"]) - meta["angle_ACB"]) > 1e-5:
            errors.append(f"{qid}: angle ACB mismatch")
        return float(meta["angle_PCA"] if task_type == "direct" else 180 - meta["angle_PCA"] - meta["angle_CBA"])
    if sub == "cevian_two_triangles":
        if abs(_angle_deg(verts["D"], verts["A"], verts["B"]) - meta["angle_ADB"]) > 1e-5:
            errors.append(f"{qid}: angle ADB mismatch")
        if abs(_angle_deg(verts["C"], verts["D"], verts["B"]) - meta["angle_C"]) > 1e-5:
            errors.append(f"{qid}: angle C mismatch")
        if abs(_angle_deg(verts["B"], verts["D"], verts["C"]) - meta["angle_CBD"]) > 1e-5:
            errors.append(f"{qid}: angle CBD mismatch")
        return float(meta["angle_ADB"] if task_type == "direct" else meta["angle_ADB"] - meta["angle_C"])
    errors.append(f"{qid}: unknown geometry subtype {sub}")
    return None


def validate_dataset(
    dataset_dir,
    expected_images: int | None = None,
    expected_questions: int | None = None,
) -> dict:
    root = Path(dataset_dir)
    data = json.loads((root / "questions.json").read_text(encoding="utf-8"))
    qs = data["questions"]
    errors: list[str] = []

    target_q = expected_questions if expected_questions is not None else data.get("question_count", 48)
    target_img = expected_images if expected_images is not None else data.get("image_count", 24)

    if len(qs) != target_q:
        errors.append(f"expected {target_q} questions, got {len(qs)}")
    groups: dict[str, list[dict]] = {}
    for q in qs:
        groups.setdefault(q["image_id"], []).append(q)
    if len(groups) != target_img:
        errors.append(f"expected {target_img} images, got {len(groups)}")

    for image_id, items in groups.items():
        if len(items) != 2 or {x["task_type"] for x in items} != {"direct", "multi_step"}:
            errors.append(f"{image_id}: expected one direct and one multi_step question")

    for q in qs:
        qid = q["question_id"]
        if Path(q["image_path"]).is_absolute():
            errors.append(f"{qid}: image_path must be portable relative path, got {q['image_path']}")
        # Guard against accidental answer leakage in prompt text (excluding product/year identifiers like P1..P5).
        standalone_nums = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", q["prompt"])
        if standalone_nums:
            errors.append(f"{qid}: prompt contains standalone numeric literal(s) {standalone_nums}")

        p = resolve_image_path(root, q["image_path"])
        if not p.is_file():
            errors.append(f"missing image: {p}")
            continue
        try:
            with Image.open(p) as im:
                if im.size != (768, 768):
                    errors.append(f"{p}: expected 768x768, got {im.size}")
                if im.mode != "RGB":
                    errors.append(f"{p}: expected RGB mode, got {im.mode}")
                im.verify()
            if pixel_sha256(p) != q["metadata"]["pixel_sha256"]:
                errors.append(f"{p}: pixel hash mismatch")
        except Exception as e:
            errors.append(f"invalid image {p}: {e}")

        if q["modality"] == "chart":
            expected = _recompute_chart(q["task_type"], q["metadata"], errors, qid)
        elif q["modality"] == "geometry":
            expected = _recompute_geometry(q["task_type"], q["metadata"], errors, qid)
        else:
            errors.append(f"{qid}: unknown modality {q['modality']}")
            expected = None

        if expected is not None and abs(float(q["answer"]) - float(expected)) > 1e-9:
            errors.append(f"{qid}: answer recomputation mismatch ({q['answer']} vs {expected})")

    chart_count = sum(q["modality"] == "chart" for q in qs)
    geom_count = sum(q["modality"] == "geometry" for q in qs)
    if chart_count != geom_count:
        errors.append(f"modality imbalance: chart={chart_count}, geometry={geom_count}")

    if errors:
        raise ValueError("\n".join(errors))

    return {
        "valid": True,
        "images": len(groups),
        "questions": len(qs),
        "templates": len({q["template_id"] for q in qs}),
        "modalities": {"chart": chart_count, "geometry": geom_count},
    }
