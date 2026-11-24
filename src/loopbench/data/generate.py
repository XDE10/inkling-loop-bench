import hashlib
import json
import math
import random
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.ticker import MultipleLocator
import numpy as np
from PIL import Image, PngImagePlugin
from .schema import Question
from ..storage import atomic_write_text

RC_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.unicode_minus": False,
}


def _save_rgb_figure(fig, path: Path) -> None:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = Image.fromarray(rgba[:, :, :3], mode="RGB")
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Software", "LoopBench")
    rgb.save(path, format="PNG", pnginfo=meta)
    plt.close(fig)


def pixel_sha256(path) -> str:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _unit(vx: float, vy: float) -> tuple[float, float]:
    norm = math.hypot(vx, vy)
    if norm < 1e-12:
        return (0.0, 1.0)
    return (vx / norm, vy / norm)


def _setup_grid5(ax, axis: str = "y", limit: int = 105) -> None:
    if axis == "y":
        ax.set_ylim(0, limit)
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.grid(which="major", axis="y", color="#AEB6C2", linestyle="-", linewidth=0.9, alpha=0.9)
        ax.grid(which="minor", axis="y", color="#D0D7E2", linestyle="--", linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
    else:
        ax.set_xlim(0, limit)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.grid(which="major", axis="x", color="#AEB6C2", linestyle="-", linewidth=0.9, alpha=0.9)
        ax.grid(which="minor", axis="x", color="#D0D7E2", linestyle="--", linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)


def _compute_geometry_span(points: list[tuple[float, float]], pad_ratio: float = 0.24) -> tuple[float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    raw_span = max(max(xs) - min(xs), max(ys) - min(ys), 0.5)
    span = raw_span * (1.0 + 2.0 * pad_ratio)
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    return cx, cy, span


def _draw_angle(
    ax,
    v: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    span: float,
    label: str | None = None,
    right_angle: bool = False,
    color: str = "#1F4E79",
) -> None:
    d1 = math.hypot(p1[0] - v[0], p1[1] - v[1])
    d2 = math.hypot(p2[0] - v[0], p2[1] - v[1])
    u1 = _unit(p1[0] - v[0], p1[1] - v[1])
    u2 = _unit(p2[0] - v[0], p2[1] - v[1])
    r = min(0.088 * span, 0.24 * min(d1, d2))

    if right_angle:
        s = r * 0.78
        c1 = (v[0] + s * u1[0], v[1] + s * u1[1])
        c12 = (v[0] + s * (u1[0] + u2[0]), v[1] + s * (u1[1] + u2[1]))
        c2 = (v[0] + s * u2[0], v[1] + s * u2[1])
        ax.plot([c1[0], c12[0], c2[0]], [c1[1], c12[1], c2[1]], color="#C00000", linewidth=1.7)
    else:
        a1 = math.degrees(math.atan2(u1[1], u1[0]))
        a2 = math.degrees(math.atan2(u2[1], u2[0]))
        diff = (a2 - a1) % 360.0
        if diff > 180.0:
            theta1, theta2 = a2, a1
        else:
            theta1, theta2 = a1, a2
        arc_color = "#C00000" if label == "?" else color
        ax.add_patch(
            Arc(
                v,
                2.0 * r,
                2.0 * r,
                angle=0.0,
                theta1=theta1,
                theta2=theta2,
                color=arc_color,
                linewidth=1.7,
            )
        )

    if label is not None:
        bx, by = _unit(u1[0] + u2[0], u1[1] + u2[1])
        dot = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
        half_sin = max(0.32, math.sin(0.5 * math.acos(dot)))
        dist = min(r + (0.048 * span) / half_sin, 0.44 * min(d1, d2))
        lx = v[0] + dist * bx
        ly = v[1] + dist * by
        txt_color = "#C00000" if label == "?" else "#111111"
        weight = "bold" if label == "?" else "normal"
        ax.text(
            lx,
            ly,
            label,
            fontsize=12,
            fontweight=weight,
            color=txt_color,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )


def _draw_vertex_name(
    ax,
    v: tuple[float, float],
    neighbors: list[tuple[float, float]],
    name: str,
    span: float,
) -> None:
    sx, sy = 0.0, 0.0
    for nb in neighbors:
        ux, uy = _unit(nb[0] - v[0], nb[1] - v[1])
        sx += ux
        sy += uy
    if math.hypot(sx, sy) < 1e-6:
        ox, oy = (0.0, -1.0)
    else:
        ux, uy = _unit(sx, sy)
        ox, oy = (-ux, -uy)
    offset = 0.055 * span
    ax.text(
        v[0] + offset * ox,
        v[1] + offset * oy,
        name,
        fontsize=13,
        fontweight="bold",
        color="#111111",
        ha="center",
        va="center",
    )


def _draw_tick_mark(ax, p1: tuple[float, float], p2: tuple[float, float], span: float) -> None:
    mx, my = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])
    tx, ty = _unit(p2[0] - p1[0], p2[1] - p1[1])
    nx, ny = -ty, tx
    h = 0.026 * span
    ax.plot([mx - h * nx, mx + h * nx], [my - h * ny, my + h * ny], color="#C00000", linewidth=2.0)


def _draw_parallel_mark(ax, p1: tuple[float, float], p2: tuple[float, float], span: float, pos: float = 0.5) -> None:
    mx = p1[0] + pos * (p2[0] - p1[0])
    my = p1[1] + pos * (p2[1] - p1[1])
    tx, ty = _unit(p2[0] - p1[0], p2[1] - p1[1])
    nx, ny = -ty, tx
    s = 0.024 * span
    w1 = (mx - s * tx + 0.75 * s * nx, my - s * ty + 0.75 * s * ny)
    w2 = (mx - s * tx - 0.75 * s * nx, my - s * ty - 0.75 * s * ny)
    ax.plot([w1[0], mx, w2[0]], [w1[1], my, w2[1]], color="#2E75B6", linewidth=1.8)


def _render_chart(idx: int, rng: random.Random, path: Path) -> tuple[str, float, str, float, dict]:
    t = idx % 6
    with matplotlib.rc_context(RC_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=128)

        if t == 0:
            regions = ["North", "South", "East", "West"]
            vals = rng.sample(range(15, 100, 5), 4)
            values = dict(zip(regions, vals))
            target_region = rng.choice(regions)
            ax.bar(
                regions,
                vals,
                width=0.58,
                color=["#4472C4", "#ED7D31", "#70AD47", "#7030A0"],
                edgecolor="#222222",
                linewidth=0.9,
            )
            _setup_grid5(ax, "y", 105)
            ax.set_ylabel("Units")
            ax.set_title("Regional Output (5-unit grid)")
            p1 = f"What is the value for {target_region} in the bar chart? Answer with a number in units."
            p2 = "What is the difference between the highest regional value and the lowest regional value? Answer with a number in units."
            direct = float(values[target_region])
            multi = float(max(vals) - min(vals))
            meta = {"subtype": "bar4_grid5", "values": values, "target_region": target_region}

        elif t == 1:
            sectors = ["Solar", "Wind", "Hydro", "Biomass"]
            vals = rng.sample(range(15, 100, 5), 4)
            values = dict(zip(sectors, vals))
            pair = rng.sample(sectors, 2)
            ax.barh(
                sectors,
                vals,
                height=0.56,
                color=["#F1C232", "#3D85C6", "#45818E", "#6AA84F"],
                edgecolor="#222222",
                linewidth=0.9,
            )
            _setup_grid5(ax, "x", 105)
            ax.invert_yaxis()
            ax.set_xlabel("Units")
            ax.set_title("Sector Generation (5-unit grid)")
            p1 = "What is the highest output among the four sectors? Answer with a number in units."
            p2 = f"What is the combined output of {pair[0]} and {pair[1]}? Answer with a number in units."
            direct = float(max(vals))
            multi = float(values[pair[0]] + values[pair[1]])
            meta = {"subtype": "barh4_grid5", "values": values, "sum_pair": pair}

        elif t == 2:
            months = ["Jan", "Feb", "Mar", "Apr", "May"]
            vals = rng.sample(range(15, 100, 5), 5)
            values = dict(zip(months, vals))
            target_month = rng.choice(months)
            pair = rng.sample(months, 2)
            ax.plot(
                months,
                vals,
                color="#1F4E79",
                linewidth=2.3,
                marker="o",
                markersize=8.5,
                markerfacecolor="#D9534F",
                markeredgecolor="#111111",
                markeredgewidth=1.0,
                zorder=3,
            )
            _setup_grid5(ax, "y", 105)
            ax.grid(which="major", axis="x", color="#DCE2EC", linestyle=":", linewidth=0.8)
            ax.set_ylabel("Units")
            ax.set_title("Monthly Index (5-unit grid)")
            p1 = f"What is the value recorded in {target_month}? Answer with a number in units."
            p2 = f"What is the combined value of {pair[0]} and {pair[1]} in the line chart? Answer with a number in units."
            direct = float(values[target_month])
            multi = float(values[pair[0]] + values[pair[1]])
            meta = {
                "subtype": "line5_grid5",
                "values": values,
                "target_month": target_month,
                "sum_pair": pair,
            }

        elif t == 3:
            plants = ["Plant A", "Plant B", "Plant C"]
            nums = rng.sample(range(18, 90), 6)
            q1 = dict(zip(plants, nums[:3]))
            q2 = dict(zip(plants, nums[3:]))
            series_map = {"Q1": q1, "Q2": q2}
            target_series = rng.choice(["Q1", "Q2"])
            target_plant = rng.choice(plants)
            sum_series = rng.choice(["Q1", "Q2"])
            x = np.arange(len(plants))
            width = 0.35
            b1 = ax.bar(x - width / 2, [q1[p] for p in plants], width, label="Q1", color="#4472C4", edgecolor="#222222", linewidth=0.8)
            b2 = ax.bar(x + width / 2, [q2[p] for p in plants], width, label="Q2", color="#ED7D31", edgecolor="#222222", linewidth=0.8)
            ax.bar_label(b1, padding=3, fontsize=10)
            ax.bar_label(b2, padding=3, fontsize=10)
            ax.set_xticks(x, plants)
            ax.set_ylim(0, 115)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_ylabel("Units")
            ax.set_title("Quarterly Plant Production")
            ax.legend(loc="upper right", frameon=True)
            p1 = f"What is the {target_series} production for {target_plant}? Answer with a number in units."
            p2 = f"What is the total {sum_series} production summed across all three plants? Answer with a number in units."
            direct = float(series_map[target_series][target_plant])
            multi = float(sum(series_map[sum_series].values()))
            meta = {
                "subtype": "grouped_bar_labeled",
                "series": series_map,
                "target_series": target_series,
                "target_plant": target_plant,
                "sum_series": sum_series,
            }

        elif t == 4:
            divisions = ["Div X", "Div Y", "Div Z"]
            core = {d: rng.randint(24, 50) for d in divisions}
            addon = {d: rng.randint(20, 42) for d in divisions}
            seg_map = {"Core": core, "Addon": addon}
            target_seg = rng.choice(["Core", "Addon"])
            target_div = rng.choice(divisions)
            pair = rng.sample(divisions, 2)
            x = np.arange(len(divisions))
            core_vals = [core[d] for d in divisions]
            addon_vals = [addon[d] for d in divisions]
            ax.bar(x, core_vals, width=0.52, label="Core", color="#2F5597", edgecolor="#222222", linewidth=0.8)
            ax.bar(x, addon_vals, width=0.52, bottom=core_vals, label="Addon", color="#70AD47", edgecolor="#222222", linewidth=0.8)
            for idx_d, d in enumerate(divisions):
                ax.text(idx_d, core[d] * 0.5, str(core[d]), ha="center", va="center", color="white", fontweight="bold", fontsize=11)
                ax.text(idx_d, core[d] + addon[d] * 0.5, str(addon[d]), ha="center", va="center", color="white", fontweight="bold", fontsize=11)
            ax.set_xticks(x, divisions)
            ax.set_ylim(0, 115)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_ylabel("Units")
            ax.set_title("Division Breakdown (Core + Addon)")
            ax.legend(loc="upper right", frameon=True)
            p1 = f"What is the {target_seg} value for {target_div}? Answer with a number in units."
            p2 = f"What is the combined total (Core plus Addon) of {pair[0]} and {pair[1]} together? Answer with a number in units."
            direct = float(seg_map[target_seg][target_div])
            multi = float((core[pair[0]] + addon[pair[0]]) + (core[pair[1]] + addon[pair[1]]))
            meta = {
                "subtype": "stacked_bar_labeled",
                "segments": seg_map,
                "target_segment": target_seg,
                "target_division": target_div,
                "sum_divisions": pair,
            }

        else:
            products = ["P1", "P2", "P3", "P4", "P5"]
            target = rng.choice([45, 50, 55, 60])
            above_k = rng.choice([2, 3])
            above_vals = rng.sample(range(target + 5, 100, 5), above_k)
            below_vals = rng.sample(range(15, target, 5), 5 - above_k)
            vals = above_vals + below_vals
            rng.shuffle(vals)
            values = dict(zip(products, vals))
            ax.bar(products, vals, width=0.56, color="#5B9BD5", edgecolor="#222222", linewidth=0.9)
            ax.axhline(target, color="#C00000", linestyle="--", linewidth=2.1, label="Target")
            _setup_grid5(ax, "y", 110)
            ax.set_ylabel("Units")
            ax.set_title("Product Output vs Target (5-unit grid)")
            ax.legend(loc="upper right", frameon=True)
            p1 = "What is the value of the dashed Target line? Answer with a number in units."
            p2 = "For all products strictly above the Target line, what is the sum of their excess amounts above the Target line? Answer with a number in units."
            direct = float(target)
            multi = float(sum(v - target for v in vals if v > target))
            meta = {"subtype": "threshold_bar5_grid5", "values": values, "target": target}

        fig.subplots_adjust(left=0.14, right=0.95, bottom=0.12, top=0.90)
        _save_rgb_figure(fig, path)

    return p1, direct, p2, multi, meta


def _render_geometry(idx: int, rng: random.Random, path: Path) -> tuple[str, float, str, float, dict]:
    t = idx % 6
    with matplotlib.rc_context(RC_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=128)
        ax.set_aspect("equal")
        ax.axis("off")

        if t == 0:
            angle_a = rng.randint(38, 66)
            angle_b = rng.randint(38, 66)
            angle_c = 180 - angle_a - angle_b
            pt_a = (0.0, 0.0)
            pt_b = (1.0, 0.0)
            ac = math.sin(math.radians(angle_b)) / math.sin(math.radians(angle_c))
            pt_c = (ac * math.cos(math.radians(angle_a)), ac * math.sin(math.radians(angle_a)))
            verts = {"A": pt_a, "B": pt_b, "C": pt_c}
            angles = {"A": angle_a, "B": angle_b, "C": angle_c}
            known1, known2, target = rng.sample(["A", "B", "C"], 3)

            cx, cy, span = _compute_geometry_span([pt_a, pt_b, pt_c])
            ax.plot([pt_a[0], pt_b[0], pt_c[0], pt_a[0]], [pt_a[1], pt_b[1], pt_c[1], pt_a[1]], color="#1F4E79", linewidth=2.2)
            neighbors = {"A": [pt_b, pt_c], "B": [pt_a, pt_c], "C": [pt_a, pt_b]}
            for name, pt in verts.items():
                _draw_vertex_name(ax, pt, neighbors[name], name, span)
            for name in (known1, known2):
                n1, n2 = neighbors[name]
                _draw_angle(ax, verts[name], n1, n2, span, label=f"{angles[name]}°")
            n1, n2 = neighbors[target]
            _draw_angle(ax, verts[target], n1, n2, span, label="?")

            p1 = f"In triangle ABC, what is the marked interior angle at vertex {known1}, in degrees?"
            p2 = f"In triangle ABC, find the interior angle marked '?' at vertex {target}, in degrees."
            direct = float(angles[known1])
            multi = float(angles[target])
            meta = {
                "subtype": "scalene_triangle",
                "angles": angles,
                "known": [known1, known2],
                "target": target,
                "vertices": {k: [v[0], v[1]] for k, v in verts.items()},
            }

        elif t == 1:
            angle_a = rng.randint(34, 56)
            angle_c = 90 - angle_a
            pt_a = (0.0, 0.0)
            pt_b = (1.0, 0.0)
            pt_c = (1.0, math.tan(math.radians(angle_a)))
            verts = {"A": pt_a, "B": pt_b, "C": pt_c}
            angles = {"A": angle_a, "B": 90, "C": angle_c}
            known_acute, target_acute = rng.sample(["A", "C"], 2)

            cx, cy, span = _compute_geometry_span([pt_a, pt_b, pt_c])
            ax.plot([pt_a[0], pt_b[0], pt_c[0], pt_a[0]], [pt_a[1], pt_b[1], pt_c[1], pt_a[1]], color="#1F4E79", linewidth=2.2)
            neighbors = {"A": [pt_b, pt_c], "B": [pt_a, pt_c], "C": [pt_a, pt_b]}
            for name, pt in verts.items():
                _draw_vertex_name(ax, pt, neighbors[name], name, span)
            _draw_angle(ax, pt_b, pt_a, pt_c, span, label=None, right_angle=True)
            _draw_angle(ax, verts[known_acute], *neighbors[known_acute], span, label=f"{angles[known_acute]}°")
            _draw_angle(ax, verts[target_acute], *neighbors[target_acute], span, label="?")

            p1 = f"In triangle ABC, what is the marked interior angle at vertex {known_acute}, in degrees?"
            p2 = f"In triangle ABC, find the interior angle marked '?' at vertex {target_acute}, in degrees."
            direct = float(angles[known_acute])
            multi = float(angles[target_acute])
            meta = {
                "subtype": "right_triangle_symbol",
                "angles": angles,
                "known_acute": known_acute,
                "target_acute": target_acute,
                "vertices": {k: [v[0], v[1]] for k, v in verts.items()},
            }

        elif t == 2:
            angle_c = rng.randrange(38, 84, 2)
            base_angle = (180 - angle_c) // 2
            pt_a = (0.0, 0.0)
            pt_b = (1.0, 0.0)
            pt_c = (0.5, 0.5 * math.tan(math.radians(base_angle)))
            verts = {"A": pt_a, "B": pt_b, "C": pt_c}
            angles = {"A": base_angle, "B": base_angle, "C": angle_c}
            mode = rng.choice(["given_apex", "given_base"])
            given_v, target_v = ("C", "B") if mode == "given_apex" else ("A", "C")

            cx, cy, span = _compute_geometry_span([pt_a, pt_b, pt_c])
            ax.plot([pt_a[0], pt_b[0], pt_c[0], pt_a[0]], [pt_a[1], pt_b[1], pt_c[1], pt_a[1]], color="#1F4E79", linewidth=2.2)
            _draw_tick_mark(ax, pt_a, pt_c, span)
            _draw_tick_mark(ax, pt_b, pt_c, span)
            neighbors = {"A": [pt_b, pt_c], "B": [pt_a, pt_c], "C": [pt_a, pt_b]}
            for name, pt in verts.items():
                _draw_vertex_name(ax, pt, neighbors[name], name, span)
            _draw_angle(ax, verts[given_v], *neighbors[given_v], span, label=f"{angles[given_v]}°")
            _draw_angle(ax, verts[target_v], *neighbors[target_v], span, label="?")

            p1 = f"In triangle ABC, sides marked with tick marks are equal in length. What is the marked interior angle at vertex {given_v}, in degrees?"
            p2 = f"In triangle ABC, sides marked with tick marks are equal in length. Find the interior angle marked '?' at vertex {target_v}, in degrees."
            direct = float(angles[given_v])
            multi = float(angles[target_v])
            meta = {
                "subtype": "isosceles_ticks",
                "angles": angles,
                "mode": mode,
                "given_vertex": given_v,
                "target_vertex": target_v,
                "vertices": {k: [v[0], v[1]] for k, v in verts.items()},
            }

        elif t == 3:
            angle_a = rng.randint(38, 62)
            angle_c = rng.randint(38, 62)
            angle_abc = 180 - angle_a - angle_c
            ext_cbd = angle_a + angle_c
            pt_a = (0.0, 0.0)
            pt_b = (1.0, 0.0)
            pt_d = (1.58, 0.0)
            ac = math.sin(math.radians(angle_abc)) / math.sin(math.radians(angle_c))
            pt_c = (ac * math.cos(math.radians(angle_a)), ac * math.sin(math.radians(angle_a)))
            verts = {"A": pt_a, "B": pt_b, "C": pt_c, "D": pt_d}

            cx, cy, span = _compute_geometry_span([pt_a, pt_b, pt_c, pt_d])
            ax.plot([pt_a[0], pt_d[0]], [pt_a[1], pt_d[1]], color="#1F4E79", linewidth=2.2)
            ax.plot([pt_a[0], pt_c[0], pt_b[0]], [pt_a[1], pt_c[1], pt_b[1]], color="#1F4E79", linewidth=2.2)
            _draw_vertex_name(ax, pt_a, [pt_b, pt_c], "A", span)
            _draw_vertex_name(ax, pt_b, [pt_a, pt_d, pt_c], "B", span)
            _draw_vertex_name(ax, pt_c, [pt_a, pt_b], "C", span)
            _draw_vertex_name(ax, pt_d, [pt_b], "D", span)
            _draw_angle(ax, pt_a, pt_b, pt_c, span, label=f"{angle_a}°")
            _draw_angle(ax, pt_c, pt_a, pt_b, span, label=f"{angle_c}°")
            _draw_angle(ax, pt_b, pt_d, pt_c, span, label="?")

            p1 = "Point B lies on straight line segment AD. What is the marked interior angle at vertex C, in degrees?"
            p2 = "Point B lies on straight line segment AD. Find the exterior angle CBD marked '?', in degrees."
            direct = float(angle_c)
            multi = float(ext_cbd)
            meta = {
                "subtype": "exterior_angle",
                "angle_A": angle_a,
                "angle_C": angle_c,
                "angle_ABC": angle_abc,
                "exterior_CBD": ext_cbd,
                "vertices": {k: [v[0], v[1]] for k, v in verts.items()},
            }

        elif t == 4:
            angle_a = rng.randint(42, 64)
            angle_b = rng.randint(42, 64)
            angle_acb = 180 - angle_a - angle_b
            pt_a = (0.0, 0.0)
            pt_b = (1.1, 0.0)
            ac = 1.1 * math.sin(math.radians(angle_b)) / math.sin(math.radians(angle_acb))
            pt_c = (ac * math.cos(math.radians(angle_a)), ac * math.sin(math.radians(angle_a)))
            pt_p = (pt_c[0] - 0.62, pt_c[1])
            pt_q = (pt_c[0] + 0.62, pt_c[1])
            verts = {"A": pt_a, "B": pt_b, "C": pt_c, "P": pt_p, "Q": pt_q}

            cx, cy, span = _compute_geometry_span(list(verts.values()))
            ax.plot([pt_a[0], pt_b[0], pt_c[0], pt_a[0]], [pt_a[1], pt_b[1], pt_c[1], pt_a[1]], color="#1F4E79", linewidth=2.2)
            ax.plot([pt_p[0], pt_q[0]], [pt_p[1], pt_q[1]], color="#1F4E79", linewidth=2.0)
            _draw_parallel_mark(ax, pt_a, pt_b, span, pos=0.50)
            _draw_parallel_mark(ax, pt_c, pt_q, span, pos=0.55)
            _draw_vertex_name(ax, pt_a, [pt_b, pt_c], "A", span)
            _draw_vertex_name(ax, pt_b, [pt_a, pt_c], "B", span)
            _draw_vertex_name(ax, pt_c, [pt_p, pt_q, pt_a, pt_b], "C", span)
            _draw_vertex_name(ax, pt_p, [pt_c], "P", span)
            _draw_vertex_name(ax, pt_q, [pt_c], "Q", span)
            _draw_angle(ax, pt_c, pt_p, pt_a, span, label=f"{angle_a}°")
            _draw_angle(ax, pt_b, pt_a, pt_c, span, label=f"{angle_b}°")
            _draw_angle(ax, pt_c, pt_a, pt_b, span, label="?")

            p1 = "Lines PQ and AB are parallel. What is the marked angle PCA, in degrees?"
            p2 = "Lines PQ and AB are parallel. Find the interior angle ACB marked '?', in degrees."
            direct = float(angle_a)
            multi = float(angle_acb)
            meta = {
                "subtype": "parallel_line_apex",
                "angle_PCA": angle_a,
                "angle_CBA": angle_b,
                "angle_ACB": angle_acb,
                "vertices": {k: [v[0], v[1]] for k, v in verts.items()},
            }

        else:
            angle_a = rng.randint(38, 52)
            angle_adb = rng.randint(76, 88)
            angle_cbd = rng.randint(35, 42)
            angle_abd = 180 - angle_a - angle_adb
            angle_bdc = 180 - angle_adb
            angle_c = angle_adb - angle_cbd

            pt_d = (0.0, 0.0)
            theta_b = math.radians(180 - angle_adb)
            pt_b = (math.cos(theta_b), math.sin(theta_b))
            len_ad = math.sin(math.radians(angle_abd)) / math.sin(math.radians(angle_a))
            len_cd = math.sin(math.radians(angle_cbd)) / math.sin(math.radians(angle_c))
            pt_a = (-len_ad, 0.0)
            pt_c = (len_cd, 0.0)
            verts = {"A": pt_a, "B": pt_b, "C": pt_c, "D": pt_d}

            cx, cy, span = _compute_geometry_span(list(verts.values()))
            ax.plot([pt_a[0], pt_c[0]], [pt_a[1], pt_c[1]], color="#1F4E79", linewidth=2.2)
            ax.plot([pt_a[0], pt_b[0], pt_c[0]], [pt_a[1], pt_b[1], pt_c[1]], color="#1F4E79", linewidth=2.2)
            ax.plot([pt_d[0], pt_b[0]], [pt_d[1], pt_b[1]], color="#1F4E79", linewidth=2.0)
            _draw_vertex_name(ax, pt_a, [pt_d, pt_b], "A", span)
            _draw_vertex_name(ax, pt_b, [pt_a, pt_d, pt_c], "B", span)
            _draw_vertex_name(ax, pt_c, [pt_d, pt_b], "C", span)
            _draw_vertex_name(ax, pt_d, [pt_a, pt_c, pt_b], "D", span)
            _draw_angle(ax, pt_a, pt_d, pt_b, span, label=f"{angle_a}°")
            _draw_angle(ax, pt_d, pt_a, pt_b, span, label=f"{angle_adb}°")
            _draw_angle(ax, pt_c, pt_d, pt_b, span, label=f"{angle_c}°")
            _draw_angle(ax, pt_b, pt_d, pt_c, span, label="?")

            p1 = "Point D lies on straight segment AC. What is the marked angle ADB, in degrees?"
            p2 = "Point D lies on straight segment AC. Find the angle CBD marked '?' in triangle CBD, in degrees."
            direct = float(angle_adb)
            multi = float(angle_cbd)
            meta = {
                "subtype": "cevian_two_triangles",
                "angle_A": angle_a,
                "angle_ADB": angle_adb,
                "angle_ABD": angle_abd,
                "angle_BDC": angle_bdc,
                "angle_C": angle_c,
                "angle_CBD": angle_cbd,
                "vertices": {k: [v[0], v[1]] for k, v in verts.items()},
            }

        ax.set_xlim(cx - 0.5 * span, cx + 0.5 * span)
        ax.set_ylim(cy - 0.5 * span, cy + 0.5 * span)
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
        _save_rgb_figure(fig, path)

    return p1, direct, p2, multi, meta


def generate_dataset(out_dir, seed: int = 20261123, n_per_type: int = 12) -> dict:
    if n_per_type <= 0:
        raise ValueError("n_per_type must be positive")
    out = Path(out_dir)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows: list[Question] = []

    for modality in ("chart", "geometry"):
        for i in range(n_per_type):
            image_id = f"{modality}-{i:03d}"
            template = f"{modality}-v1-{i % 6}"
            rel_path = f"images/{image_id}.png"
            full_path = out / rel_path
            if modality == "chart":
                p1, direct, p2, multi, common = _render_chart(i, rng, full_path)
                unit = "units"
            else:
                p1, direct, p2, multi, common = _render_geometry(i, rng, full_path)
                unit = "degrees"
            pix = pixel_sha256(full_path)
            for task, prompt, answer in (("direct", p1, direct), ("multi_step", p2, multi)):
                rows.append(
                    Question(
                        question_id=f"{image_id}-{task}",
                        image_id=image_id,
                        template_id=template,
                        modality=modality,
                        task_type=task,
                        prompt=prompt,
                        answer=answer,
                        unit=unit,
                        image_path=rel_path,
                        metadata={"seed": seed, **common, "pixel_sha256": pix},
                    )
                )

    manifest = {
        "schema_version": 2,
        "seed": seed,
        "image_count": n_per_type * 2,
        "question_count": len(rows),
        "questions": [q.to_dict() for q in rows],
    }
    atomic_write_text(out / "questions.json", json.dumps(manifest, indent=2) + "\n")
    return manifest
