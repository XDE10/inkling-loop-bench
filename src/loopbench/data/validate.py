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

    # v2 (hard) chart sub-templates
    if sub == "grouped_bar3x4_net_diff":
        series = meta["series"]
        regions = list(series["Q1"].keys())
        all_vals = [series[q][r] for q in ("Q1", "Q2", "Q3") for r in regions]
        hi_q3 = max(regions, key=lambda r: series["Q3"][r])
        lo_q1 = min(regions, key=lambda r: series["Q1"][r])
        if len(set(all_vals)) != 12 or hi_q3 == lo_q1:
            errors.append(f"{qid}: grouped_bar3x4_net_diff uniqueness invariant violated")
        if task_type == "direct":
            return float(series[meta["target_quarter"]][meta["target_region"]])
        return float(
            sum(series[q][hi_q3] for q in ("Q1", "Q2", "Q3"))
            - sum(series[q][lo_q1] for q in ("Q1", "Q2", "Q3"))
        )
    if sub == "dual_line6_crossover_grid5":
        series = meta["series"]
        months = list(series["Alpha"].keys())
        a_vals = [series["Alpha"][m] for m in months]
        b_vals = [series["Beta"][m] for m in months]
        if any(v % 5 != 0 for v in a_vals + b_vals) or any(abs(a - b) < 10 for a, b in zip(a_vals, b_vals)):
            errors.append(f"{qid}: dual_line6_crossover_grid5 separation/grid5 invariant violated")
        if task_type == "direct":
            return float(series[meta["target_series"]][meta["target_month"]])
        return float(sum(a - b for a, b in zip(a_vals, b_vals) if a > b))
    if sub == "stacked3_bar4_range":
        segs = meta["segments"]
        plants = list(segs["Base"].keys())
        totals = [segs["Base"][p] + segs["Mid"][p] + segs["Peak"][p] for p in plants]
        if len(set(totals)) != 4:
            errors.append(f"{qid}: stacked3_bar4_range distinct-totals invariant violated")
        if task_type == "direct":
            return float(segs[meta["target_segment"]][meta["target_plant"]])
        return float(max(totals) - min(totals))
    if sub == "diverging_barh6_grid5":
        vals = list(meta["values"].values())
        pos = [v for v in vals if v > 0]
        neg = [v for v in vals if v < 0]
        if len(pos) != 3 or len(neg) != 3 or len(set(vals)) != 6 or any(v % 5 != 0 for v in vals):
            errors.append(f"{qid}: diverging_barh6_grid5 invariant violated")
        if task_type == "direct":
            return float(meta["values"][meta["target_sector"]])
        return float(sum(vals))
    if sub == "band_bar6_grid5":
        vals = list(meta["values"].values())
        fl, cp = meta["floor"], meta["cap"]
        below = [v for v in vals if v < fl]
        inside = [v for v in vals if fl < v < cp]
        above = [v for v in vals if v > cp]
        if len(below) != 2 or len(inside) != 2 or len(above) != 2 or any(v % 5 != 0 for v in vals):
            errors.append(f"{qid}: band_bar6_grid5 invariant violated")
        if task_type == "direct":
            return float(meta["values"][meta["target_product"]])
        return float(sum(fl - v for v in below) + sum(v - cp for v in above))
    if sub == "matrix_heatmap_3x4":
        mat = meta["matrix"]
        r_names = list(mat.keys())
        c_names = list(mat[r_names[0]].keys())
        row_sums = [sum(mat[r][c] for c in c_names) for r in r_names]
        col_sums = [sum(mat[r][c] for r in r_names) for c in c_names]
        if len(set(row_sums)) != 3 or len(set(col_sums)) != 4:
            errors.append(f"{qid}: matrix_heatmap_3x4 row/col sum uniqueness violated")
        if task_type == "direct":
            return float(mat[meta["target_row"]][meta["target_col"]])
        return float(max(col_sums) - min(row_sums))

    errors.append(f"{qid}: unknown chart subtype {sub}")
    return None


def _recompute_geometry(task_type: str, meta: dict, errors: list[str], qid: str) -> float | None:
    sub = meta.get("subtype")
    verts = meta.get("vertices", {})
    if sub == "scalene_triangle":
        angs = meta["angles"]
        k1, k2 = meta["known"]
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

    # v2 (hard) geometry sub-templates
    if sub == "quad_parallel_isosceles":
        abx, aby = verts["B"][0] - verts["A"][0], verts["B"][1] - verts["A"][1]
        dcx, dcy = verts["C"][0] - verts["D"][0], verts["C"][1] - verts["D"][1]
        if abs(abx * dcy - aby * dcx) > 1e-6:
            errors.append(f"{qid}: AB and DC are not parallel")
        len_da = math.hypot(verts["A"][0] - verts["D"][0], verts["A"][1] - verts["D"][1])
        len_dc = math.hypot(verts["C"][0] - verts["D"][0], verts["C"][1] - verts["D"][1])
        if abs(len_da - len_dc) > 1e-6:
            errors.append(f"{qid}: AD and CD are not equal")
        if abs(_angle_deg(verts["B"], verts["A"], verts["C"]) - meta["angle_ABC"]) > 1e-5:
            errors.append(f"{qid}: angle ABC mismatch")
        if abs(_angle_deg(verts["C"], verts["A"], verts["B"]) - meta["angle_ACB"]) > 1e-5:
            errors.append(f"{qid}: angle ACB mismatch")
        expected_adc = 2 * (meta["angle_ABC"] + meta["angle_ACB"]) - 180
        if abs(_angle_deg(verts["D"], verts["A"], verts["C"]) - expected_adc) > 1e-5:
            errors.append(f"{qid}: angle ADC mismatch")
        return float(meta["angle_ABC"] if task_type == "direct" else expected_adc)

    if sub == "collinear_isosceles_exterior":
        len_da = math.hypot(verts["A"][0] - verts["D"][0], verts["A"][1] - verts["D"][1])
        len_db = math.hypot(verts["B"][0] - verts["D"][0], verts["B"][1] - verts["D"][1])
        if abs(len_da - len_db) > 1e-6:
            errors.append(f"{qid}: AD and BD are not equal")
        if abs(_angle_deg(verts["A"], verts["D"], verts["B"]) - meta["angle_BAD"]) > 1e-5:
            errors.append(f"{qid}: angle BAD mismatch")
        if abs(_angle_deg(verts["C"], verts["E"], verts["B"]) - meta["exterior_BCE"]) > 1e-5:
            errors.append(f"{qid}: exterior angle BCE mismatch")
        expected_dbc = meta["exterior_BCE"] - 2 * meta["angle_BAD"]
        if abs(_angle_deg(verts["B"], verts["D"], verts["C"]) - expected_dbc) > 1e-5:
            errors.append(f"{qid}: angle DBC mismatch")
        return float(meta["exterior_BCE"] if task_type == "direct" else expected_dbc)

    if sub == "parallel_zigzag_obtuse":
        if abs(verts["P"][1] - verts["Q"][1]) > 1e-9 or abs(verts["A"][1] - verts["B"][1]) > 1e-9:
            errors.append(f"{qid}: PCQ and ADB are not horizontal/parallel")
        if abs(_angle_deg(verts["C"], verts["P"], verts["M"]) - meta["angle_PCM"]) > 1e-5:
            errors.append(f"{qid}: angle PCM mismatch")
        if abs(_angle_deg(verts["D"], verts["A"], verts["M"]) - meta["angle_ADM"]) > 1e-5:
            errors.append(f"{qid}: angle ADM mismatch")
        expected_cmd = 360 - meta["angle_PCM"] - meta["angle_ADM"]
        if abs(_angle_deg(verts["M"], verts["C"], verts["D"]) - expected_cmd) > 1e-5:
            errors.append(f"{qid}: angle CMD mismatch")
        return float(meta["angle_PCM"] if task_type == "direct" else expected_cmd)

    if sub == "bowtie_right_isosceles":
        if abs(_angle_deg(verts["O"], verts["A"], verts["C"]) - 180.0) > 1e-5:
            errors.append(f"{qid}: A-O-C not collinear")
        if abs(_angle_deg(verts["O"], verts["B"], verts["D"]) - 180.0) > 1e-5:
            errors.append(f"{qid}: B-O-D not collinear")
        if abs(_angle_deg(verts["B"], verts["A"], verts["O"]) - 90.0) > 1e-5:
            errors.append(f"{qid}: angle ABO not 90 deg")
        if abs(_angle_deg(verts["A"], verts["O"], verts["B"]) - meta["angle_BAO"]) > 1e-5:
            errors.append(f"{qid}: angle BAO mismatch")
        len_oc = math.hypot(verts["C"][0] - verts["O"][0], verts["C"][1] - verts["O"][1])
        len_od = math.hypot(verts["D"][0] - verts["O"][0], verts["D"][1] - verts["O"][1])
        if abs(len_oc - len_od) > 1e-6:
            errors.append(f"{qid}: OC and OD not equal")
        expected_ocd = 45.0 + 0.5 * meta["angle_BAO"]
        if abs(_angle_deg(verts["C"], verts["O"], verts["D"]) - expected_ocd) > 1e-5:
            errors.append(f"{qid}: angle OCD mismatch")
        return float(meta["angle_BAO"] if task_type == "direct" else expected_ocd)

    if sub == "three_triangle_isosceles_chain":
        len_bd = math.hypot(verts["D"][0] - verts["B"][0], verts["D"][1] - verts["B"][1])
        len_be = math.hypot(verts["E"][0] - verts["B"][0], verts["E"][1] - verts["B"][1])
        if abs(len_bd - len_be) > 1e-6:
            errors.append(f"{qid}: BD and BE not equal")
        if abs(_angle_deg(verts["E"], verts["B"], verts["C"]) - 180.0) > 1e-5:
            errors.append(f"{qid}: B-E-C not collinear")
        if abs(_angle_deg(verts["A"], verts["C"], verts["B"]) - meta["angle_BAC"]) > 1e-5:
            errors.append(f"{qid}: angle BAC mismatch")
        if abs(_angle_deg(verts["B"], verts["A"], verts["D"]) - meta["angle_ABD"]) > 1e-5:
            errors.append(f"{qid}: angle ABD mismatch")
        if abs(_angle_deg(verts["C"], verts["A"], verts["B"]) - meta["angle_ACB"]) > 1e-5:
            errors.append(f"{qid}: angle ACB mismatch")
        expected_dec = 180.0 - 0.5 * (meta["angle_BAC"] + meta["angle_ACB"] + meta["angle_ABD"])
        if abs(_angle_deg(verts["E"], verts["D"], verts["C"]) - expected_dec) > 1e-5:
            errors.append(f"{qid}: angle DEC mismatch")
        return float(meta["angle_ACB"] if task_type == "direct" else expected_dec)

    if sub == "parallel_right_w_chain":
        if abs(verts["P"][1] - verts["Q"][1]) > 1e-9 or abs(verts["A"][1] - verts["B"][1]) > 1e-9:
            errors.append(f"{qid}: PCQE and ADB are not parallel")
        if abs(_angle_deg(verts["C"], verts["A"], verts["D"]) - 90.0) > 1e-5:
            errors.append(f"{qid}: angle ACD not 90 deg")
        if abs(_angle_deg(verts["A"], verts["D"], verts["C"]) - meta["angle_CAD"]) > 1e-5:
            errors.append(f"{qid}: angle CAD mismatch")
        if abs(_angle_deg(verts["D"], verts["C"], verts["E"]) - meta["angle_CDE"]) > 1e-5:
            errors.append(f"{qid}: angle CDE mismatch")
        expected_qed = 90.0 - meta["angle_CAD"] + meta["angle_CDE"]
        if abs(_angle_deg(verts["E"], verts["Q"], verts["D"]) - expected_qed) > 1e-5:
            errors.append(f"{qid}: angle QED mismatch")
        return float(meta["angle_CDE"] if task_type == "direct" else expected_qed)

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
        # Guard against accidental answer leakage in prompt text (excluding product/year identifiers like P1..P6, Q1..Q3, S1..S6, W1..W4).
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

    difficulties: dict[str, int] = {}
    for q in qs:
        diff = q.get("metadata", {}).get("difficulty") or ("hard" if "-v2-" in q["template_id"] else "standard")
        difficulties[diff] = difficulties.get(diff, 0) + 1

    return {
        "valid": True,
        "images": len(groups),
        "questions": len(qs),
        "templates": len({q["template_id"] for q in qs}),
        "modalities": {"chart": chart_count, "geometry": geom_count},
        "difficulties": difficulties,
    }
