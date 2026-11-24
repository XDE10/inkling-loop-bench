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


def _setup_signed_grid5(ax, lo: int = -50, hi: int = 55) -> None:
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(which="major", axis="x", color="#AEB6C2", linestyle="-", linewidth=0.9, alpha=0.9)
    ax.grid(which="minor", axis="x", color="#D0D7E2", linestyle="--", linewidth=0.7, alpha=0.9)
    ax.axvline(0, color="#222222", linewidth=1.4, zorder=3)
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


def _draw_tick_mark(ax, p1: tuple[float, float], p2: tuple[float, float], span: float, pos: float = 0.5) -> None:
    mx = p1[0] + pos * (p2[0] - p1[0])
    my = p1[1] + pos * (p2[1] - p1[1])
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


def _render_chart_v2(idx: int, rng: random.Random, path: Path) -> tuple[str, float, str, float, dict]:
    """Six harder multi-hop compositional chart sub-templates (v2) with crisp visual readability."""
    t = idx % 6
    with matplotlib.rc_context(RC_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=128)

        if t == 0:
            regions = ["North", "South", "East", "West"]
            quarters = ["Q1", "Q2", "Q3"]
            while True:
                nums = rng.sample(range(18, 92), 12)
                q_map = {
                    "Q1": dict(zip(regions, nums[0:4])),
                    "Q2": dict(zip(regions, nums[4:8])),
                    "Q3": dict(zip(regions, nums[8:12])),
                }
                hi_q3_reg = max(regions, key=lambda r: q_map["Q3"][r])
                lo_q1_reg = min(regions, key=lambda r: q_map["Q1"][r])
                if hi_q3_reg != lo_q1_reg:
                    break
            target_q = rng.choice(quarters)
            target_reg = rng.choice(regions)
            x = np.arange(len(regions))
            width = 0.25
            b1 = ax.bar(x - width, [q_map["Q1"][r] for r in regions], width, label="Q1", color="#2F5597", edgecolor="#222222", linewidth=0.8)
            b2 = ax.bar(x, [q_map["Q2"][r] for r in regions], width, label="Q2", color="#ED7D31", edgecolor="#222222", linewidth=0.8)
            b3 = ax.bar(x + width, [q_map["Q3"][r] for r in regions], width, label="Q3", color="#70AD47", edgecolor="#222222", linewidth=0.8)
            ax.bar_label(b1, padding=2, fontsize=8.5)
            ax.bar_label(b2, padding=2, fontsize=8.5)
            ax.bar_label(b3, padding=2, fontsize=8.5)
            ax.set_xticks(x, regions)
            ax.set_ylim(0, 118)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_ylabel("Units")
            ax.set_title("Regional Quarterly Output (Q1–Q3)")
            ax.legend(loc="upper right", ncol=3, frameon=True)
            p1 = f"What is the {target_q} output for {target_reg}? Answer with a number in units."
            p2 = (
                "Find the region with the highest Q3 output and the region with the lowest Q1 output. "
                "What is the three-quarter total (Q1 plus Q2 plus Q3) of that highest-Q3 region minus "
                "the three-quarter total (Q1 plus Q2 plus Q3) of that lowest-Q1 region? Answer with a number in units."
            )
            tot_hi = sum(q_map[q][hi_q3_reg] for q in quarters)
            tot_lo = sum(q_map[q][lo_q1_reg] for q in quarters)
            direct = float(q_map[target_q][target_reg])
            multi = float(tot_hi - tot_lo)
            meta = {
                "subtype": "grouped_bar3x4_net_diff",
                "series": q_map,
                "target_quarter": target_q,
                "target_region": target_reg,
                "highest_q3_region": hi_q3_reg,
                "lowest_q1_region": lo_q1_reg,
            }

        elif t == 1:
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
            while True:
                alpha_vals = rng.sample(range(15, 100, 5), 6)
                beta_vals = rng.sample(range(15, 100, 5), 6)
                if all(abs(a - b) >= 10 for a, b in zip(alpha_vals, beta_vals)):
                    pos_months = sum(a > b for a, b in zip(alpha_vals, beta_vals))
                    if pos_months in (2, 3, 4):
                        break
            alpha_map = dict(zip(months, alpha_vals))
            beta_map = dict(zip(months, beta_vals))
            series_map = {"Alpha": alpha_map, "Beta": beta_map}
            target_series = rng.choice(["Alpha", "Beta"])
            target_month = rng.choice(months)
            ax.plot(
                months,
                alpha_vals,
                label="Alpha",
                color="#1F4E79",
                linewidth=2.2,
                 linestyle="-",
                marker="o",
                markersize=8.0,
                markerfacecolor="#4472C4",
                markeredgecolor="#111111",
                zorder=3,
            )
            ax.plot(
                months,
                beta_vals,
                label="Beta",
                color="#C00000",
                linewidth=2.2,
                linestyle="--",
                marker="s",
                markersize=7.5,
                markerfacecolor="#ED7D31",
                markeredgecolor="#111111",
                zorder=3,
            )
            _setup_grid5(ax, "y", 110)
            ax.grid(which="major", axis="x", color="#DCE2EC", linestyle=":", linewidth=0.8)
            ax.set_ylabel("Units")
            ax.set_title("Alpha vs Beta Monthly Trajectories (5-unit grid)")
            ax.legend(loc="upper right", frameon=True)
            p1 = f"In {target_month}, what is the value of series {target_series} in the line chart? Answer with a number in units."
            p2 = (
                "Across all months where Alpha is strictly greater than Beta, "
                "what is the sum of the positive gaps (Alpha minus Beta)? Answer with a number in units."
            )
            direct = float(series_map[target_series][target_month])
            multi = float(sum(a - b for a, b in zip(alpha_vals, beta_vals) if a > b))
            meta = {
                "subtype": "dual_line6_crossover_grid5",
                "series": series_map,
                "target_series": target_series,
                "target_month": target_month,
            }

        elif t == 2:
            plants = ["Plant A", "Plant B", "Plant C", "Plant D"]
            while True:
                base = {p: rng.randint(20, 38) for p in plants}
                mid = {p: rng.randint(20, 38) for p in plants}
                peak = {p: rng.randint(18, 34) for p in plants}
                totals = [base[p] + mid[p] + peak[p] for p in plants]
                if len(set(totals)) == 4 and min(abs(totals[i] - totals[j]) for i in range(4) for j in range(i + 1, 4)) >= 3:
                    break
            seg_map = {"Base": base, "Mid": mid, "Peak": peak}
            target_seg = rng.choice(["Base", "Mid", "Peak"])
            target_plant = rng.choice(plants)
            x = np.arange(len(plants))
            b_vals = [base[p] for p in plants]
            m_vals = [mid[p] for p in plants]
            p_vals = [peak[p] for p in plants]
            bm_vals = [b_vals[i] + m_vals[i] for i in range(len(plants))]
            ax.bar(x, b_vals, width=0.52, label="Base", color="#2F5597", edgecolor="#222222", linewidth=0.8)
            ax.bar(x, m_vals, width=0.52, bottom=b_vals, label="Mid", color="#5B9BD5", edgecolor="#222222", linewidth=0.8)
            ax.bar(x, p_vals, width=0.52, bottom=bm_vals, label="Peak", color="#70AD47", edgecolor="#222222", linewidth=0.8)
            for idx_p, p in enumerate(plants):
                ax.text(idx_p, base[p] * 0.5, str(base[p]), ha="center", va="center", color="white", fontweight="bold", fontsize=10)
                ax.text(idx_p, base[p] + mid[p] * 0.5, str(mid[p]), ha="center", va="center", color="white", fontweight="bold", fontsize=10)
                ax.text(idx_p, base[p] + mid[p] + peak[p] * 0.5, str(peak[p]), ha="center", va="center", color="white", fontweight="bold", fontsize=10)
            ax.set_xticks(x, plants)
            ax.set_ylim(0, 126)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_ylabel("Units")
            ax.set_title("Three-Tier Plant Stack (Base + Mid + Peak)")
            ax.legend(loc="upper right", ncol=3, frameon=True)
            p1 = f"What is the {target_seg} segment value for {target_plant}? Answer with a number in units."
            p2 = (
                "Compute the combined total (Base plus Mid plus Peak) for each of the four plants. "
                "What is the highest combined plant total minus the lowest combined plant total? Answer with a number in units."
            )
            tot_map = {p: base[p] + mid[p] + peak[p] for p in plants}
            direct = float(seg_map[target_seg][target_plant])
            multi = float(max(tot_map.values()) - min(tot_map.values()))
            meta = {
                "subtype": "stacked3_bar4_range",
                "segments": seg_map,
                "target_segment": target_seg,
                "target_plant": target_plant,
            }

        elif t == 3:
            sectors = ["S1", "S2", "S3", "S4", "S5", "S6"]
            pos_vals = rng.sample([10, 15, 20, 25, 30, 35, 40, 45, 50], 3)
            neg_vals = rng.sample([-45, -40, -35, -30, -25, -20, -15, -10], 3)
            vals = pos_vals + neg_vals
            rng.shuffle(vals)
            values = dict(zip(sectors, vals))
            target_sector = rng.choice(sectors)
            colors = ["#2E75B6" if v > 0 else "#D9534F" for v in vals]
            ax.barh(sectors, vals, height=0.56, color=colors, edgecolor="#222222", linewidth=0.9)
            _setup_signed_grid5(ax, -50, 55)
            ax.invert_yaxis()
            ax.set_xlabel("Signed Net Change (Units)")
            ax.set_title("Sector Net Change (5-unit signed grid)")
            p1 = f"What is the signed net change value for sector {target_sector}? Answer with a number in units."
            p2 = (
                "What is the overall net sum across all six sectors "
                "(adding positive sector values and subtracting negative sector magnitudes)? Answer with a number in units."
            )
            direct = float(values[target_sector])
            multi = float(sum(vals))
            meta = {
                "subtype": "diverging_barh6_grid5",
                "values": values,
                "target_sector": target_sector,
            }

        elif t == 4:
            products = ["P1", "P2", "P3", "P4", "P5", "P6"]
            floor_val = rng.choice([30, 35, 40])
            cap_val = rng.choice([65, 70, 75])
            below_vals = rng.sample(range(10, floor_val, 5), 2)
            inside_vals = rng.sample(range(floor_val + 5, cap_val, 5), 2)
            above_vals = rng.sample(range(cap_val + 5, 105, 5), 2)
            vals = below_vals + inside_vals + above_vals
            rng.shuffle(vals)
            values = dict(zip(products, vals))
            target_prod = rng.choice(products)
            ax.bar(products, vals, width=0.56, color="#5B9BD5", edgecolor="#222222", linewidth=0.9)
            ax.axhline(cap_val, color="#C00000", linestyle="--", linewidth=2.1, label="Cap")
            ax.axhline(floor_val, color="#1F4E79", linestyle="-.", linewidth=2.1, label="Floor")
            _setup_grid5(ax, "y", 115)
            ax.set_ylabel("Units")
            ax.set_title("Product Output vs Operating Band (5-unit grid)")
            ax.legend(loc="upper right", frameon=True)
            p1 = f"What is the output value of product {target_prod} in the bar chart? Answer with a number in units."
            p2 = (
                "For every product strictly below the Floor line or strictly above the Cap line, "
                "compute how far outside the band it lies (Floor minus value if below Floor, or value minus Cap if above Cap). "
                "What is the sum of these outside deviations across all outside products? Answer with a number in units."
            )
            direct = float(values[target_prod])
            multi = float(
                sum(floor_val - v for v in vals if v < floor_val)
                + sum(v - cap_val for v in vals if v > cap_val)
            )
            meta = {
                "subtype": "band_bar6_grid5",
                "values": values,
                "floor": floor_val,
                "cap": cap_val,
                "target_product": target_prod,
            }

        else:
            rows_lbl = ["Line X", "Line Y", "Line Z"]
            cols_lbl = ["W1", "W2", "W3", "W4"]
            while True:
                nums = rng.sample(range(16, 86), 12)
                grid = [nums[0:4], nums[4:8], nums[8:12]]
                row_sums = [sum(grid[r]) for r in range(3)]
                col_sums = [sum(grid[r][c] for r in range(3)) for c in range(4)]
                if len(set(row_sums)) == 3 and len(set(col_sums)) == 4:
                    break
            matrix_map = {
                rows_lbl[r]: {cols_lbl[c]: grid[r][c] for c in range(4)}
                for r in range(3)
            }
            target_row = rng.choice(rows_lbl)
            target_col = rng.choice(cols_lbl)
            arr = np.array(grid, dtype=float)
            ax.imshow(arr, cmap="Blues", vmin=0, vmax=125, aspect="equal")
            for r in range(3):
                for c in range(4):
                    ax.text(
                        c,
                        r,
                        str(grid[r][c]),
                        ha="center",
                        va="center",
                        color="#111111",
                        fontweight="bold",
                        fontsize=13,
                    )
            for r in range(4):
                ax.axhline(r - 0.5, color="#222222", linewidth=1.2)
            for c in range(5):
                ax.axvline(c - 0.5, color="#222222", linewidth=1.2)
            ax.set_xticks(range(4), cols_lbl)
            ax.set_yticks(range(3), rows_lbl)
            ax.set_xlabel("Week")
            ax.set_ylabel("Production Line")
            ax.set_title("Weekly Line Output Matrix (Units)")
            p1 = f"In the matrix chart, what is the output value in row {target_row} and column {target_col}? Answer with a number in units."
            p2 = (
                "Find the column with the largest three-row column sum and the row with the smallest four-column row sum. "
                "What is that largest column sum minus that smallest row sum? Answer with a number in units."
            )
            direct = float(matrix_map[target_row][target_col])
            multi = float(max(col_sums) - min(row_sums))
            meta = {
                "subtype": "matrix_heatmap_3x4",
                "matrix": matrix_map,
                "target_row": target_row,
                "target_col": target_col,
            }

        fig.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.90)
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


def _render_geometry_v2(idx: int, rng: random.Random, path: Path) -> tuple[str, float, str, float, dict]:
    """Six harder 3-to-4-step Euclidean geometry sub-templates (v2) with exact coordinates."""
    t = idx % 6
    with matplotlib.rc_context(RC_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=128)
        ax.set_aspect("equal")
        ax.axis("off")

        if t == 0:
            # Quadrilateral ABCD split by diagonal AC: AB || DC and DA = DC
            a1 = rng.randint(46, 60)
            b1 = rng.randint(54, 70)
            c1 = 180 - a1 - b1
            adc_ang = 180 - 2 * a1

            pt_a = (0.0, 0.0)
            pt_c = (1.0, 0.0)
            ab = math.sin(math.radians(c1)) / math.sin(math.radians(b1))
            pt_b = (ab * math.cos(math.radians(a1)), ab * math.sin(math.radians(a1)))
            pt_d = (0.5, -0.5 * math.tan(math.radians(a1)))
            verts = {"A": pt_a, "B": pt_b, "C": pt_c, "D": pt_d}

            cx, cy, span = _compute_geometry_span(list(verts.values()))
            ax.plot(
                [pt_a[0], pt_b[0], pt_c[0], pt_d[0], pt_a[0]],
                [pt_a[1], pt_b[1], pt_c[1], pt_d[1], pt_a[1]],
                color="#1F4E79",
                linewidth=2.2,
            )
            ax.plot([pt_a[0], pt_c[0]], [pt_a[1], pt_c[1]], color="#1F4E79", linewidth=1.9)
            _draw_parallel_mark(ax, pt_a, pt_b, span, pos=0.55)
            _draw_parallel_mark(ax, pt_d, pt_c, span, pos=0.70)
            _draw_tick_mark(ax, pt_a, pt_d, span, pos=0.45)
            _draw_tick_mark(ax, pt_c, pt_d, span, pos=0.62)
            _draw_vertex_name(ax, pt_a, [pt_b, pt_c, pt_d], "A", span)
            _draw_vertex_name(ax, pt_b, [pt_a, pt_c], "B", span)
            _draw_vertex_name(ax, pt_c, [pt_b, pt_a, pt_d], "C", span)
            _draw_vertex_name(ax, pt_d, [pt_a, pt_c], "D", span)
            _draw_angle(ax, pt_b, pt_a, pt_c, span, label=f"{b1}°")
            _draw_angle(ax, pt_c, pt_a, pt_b, span, label=f"{c1}°")
            _draw_angle(ax, pt_d, pt_a, pt_c, span, label="?")

            p1 = (
                "In quadrilateral ABCD with diagonal AC, sides AB and DC are parallel and sides AD and CD "
                "have tick marks indicating equal length. What is the marked interior angle ABC at vertex B, in degrees?"
            )
            p2 = (
                "In quadrilateral ABCD with diagonal AC, sides AB and DC are parallel and sides AD and CD "
                "have tick marks indicating equal length. Find the interior angle ADC marked '?' at vertex D, in degrees."
            )
            direct = float(b1)
            multi = float(adc_ang)
            meta = {
                "subtype": "quad_parallel_isosceles",
                "angle_BAC": a1,
                "angle_ABC": b1,
                "angle_ACB": c1,
                "angle_ADC": adc_ang,
                "vertices": {k: [v[0], v[1]] for k, v in verts.items()},
            }

        elif t == 1:
            # Collinear A-D-C-E with DA = DB and exterior angle BCE given
            a_ang = rng.randint(34, 41)
            ext_c = rng.randint(116, 128)
            bdc_ang = 2 * a_ang
            bcd_ang = 180 - ext_c
            dbc_ang = ext_c - 2 * a_ang

            pt_d = (0.0, 0.0)
            pt_a = (-1.0, 0.0)
            th_b = math.radians(bdc_ang)
            pt_b = (math.cos(th_b), math.sin(th_b))
            len_dc = math.sin(math.radians(dbc_ang)) / math.sin(math.radians(bcd_ang))
            pt_c = (len_dc, 0.0)
            pt_e = (len_dc + 0.58, 0.0)
            verts = {"A": pt_a, "B": pt_b, "C": pt_c, "D": pt_d, "E": pt_e}

            cx, cy, span = _compute_geometry_span(list(verts.values()))
            ax.plot([pt_a[0], pt_e[0]], [pt_a[1], pt_e[1]], color="#1F4E79", linewidth=2.2)
            ax.plot([pt_a[0], pt_b[0], pt_c[0]], [pt_a[1], pt_b[1], pt_c[1]], color="#1F4E79", linewidth=2.2)
            ax.plot([pt_d[0], pt_b[0]], [pt_d[1], pt_b[1]], color="#1F4E79", linewidth=2.0)
            _draw_tick_mark(ax, pt_a, pt_d, span)
            _draw_tick_mark(ax, pt_d, pt_b, span)
            _draw_vertex_name(ax, pt_a, [pt_d, pt_b], "A", span)
            _draw_vertex_name(ax, pt_b, [pt_a, pt_d, pt_c], "B", span)
            _draw_vertex_name(ax, pt_c, [pt_d, pt_e, pt_b], "C", span)
            _draw_vertex_name(ax, pt_d, [pt_a, pt_c, pt_b], "D", span)
            _draw_vertex_name(ax, pt_e, [pt_c], "E", span)
            _draw_angle(ax, pt_a, pt_d, pt_b, span, label=f"{a_ang}°")
            _draw_angle(ax, pt_c, pt_e, pt_b, span, label=f"{ext_c}°")
            _draw_angle(ax, pt_b, pt_d, pt_c, span, label="?")

            p1 = (
                "Points A, D, C, and E lie in order on a straight line, and segments AD and BD have tick marks "
                "indicating equal length. What is the marked exterior angle BCE at vertex C, in degrees?"
            )
            p2 = (
                "Points A, D, C, and E lie in order on a straight line, and segments AD and BD have tick marks "
                "indicating equal length. Find the interior angle DBC marked '?' in triangle BDC, in degrees."
            )
            direct = float(ext_c)
            multi = float(dbc_ang)
            meta = {
                "subtype": "collinear_isosceles_exterior",
                "angle_BAD": a_ang,
                "exterior_BCE": ext_c,
                "angle_BDC": bdc_ang,
                "angle_DBC": dbc_ang,
                "vertices": {k: [v[0], v[1]] for k, v in verts.items()},
            }

        elif t == 2:
            # Parallel lines PCQ || ADB with obtuse exterior-side angles PCM and ADM, find bend CMD
            alpha = rng.randint(35, 47)
            beta = rng.randint(36, 48)
            pcm_ang = 180 - alpha
            adm_ang = 180 - beta
            cmd_ang = alpha + beta

            pt_c = (0.0, 1.0)
            xm = 0.52 / math.tan(math.radians(alpha))
            pt_m = (xm, 0.48)
            xd = xm - 0.48 / math.tan(math.radians(beta))
            pt_d = (xd, 0.0)
            x_left = min(0.0, xd) - 0.58
            x_right = xm + 0.48
            pt_p = (x_left, 1.0)
            pt_q = (x_right, 1.0)
            pt_a = (x_left, 0.0)
            pt_b = (x_right, 0.0)
            verts = {"P": pt_p, "C": pt_c, "Q": pt_q, "A": pt_a, "D": pt_d, "B": pt_b, "M": pt_m}

            cx, cy, span = _compute_geometry_span(list(verts.values()))
            ax.plot([pt_p[0], pt_q[0]], [pt_p[1], pt_q[1]], color="#1F4E79", linewidth=2.1)
            ax.plot([pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]], color="#1F4E79", linewidth=2.1)
            ax.plot([pt_c[0], pt_m[0], pt_d[0]], [pt_c[1], pt_m[1], pt_d[1]], color="#1F4E79", linewidth=2.2)
            _draw_parallel_mark(ax, pt_c, pt_q, span, pos=0.68)
            _draw_parallel_mark(ax, pt_d, pt_b, span, pos=0.68)
            _draw_vertex_name(ax, pt_p, [pt_c], "P", span)
            _draw_vertex_name(ax, pt_c, [pt_p, pt_q, pt_m], "C", span)
            _draw_vertex_name(ax, pt_q, [pt_c], "Q", span)
            _draw_vertex_name(ax, pt_a, [pt_d], "A", span)
            _draw_vertex_name(ax, pt_d, [pt_a, pt_b, pt_m], "D", span)
            _draw_vertex_name(ax, pt_b, [pt_d], "B", span)
            _draw_vertex_name(ax, pt_m, [pt_c, pt_d], "M", span)
            _draw_angle(ax, pt_c, pt_p, pt_m, span, label=f"{pcm_ang}°")
            _draw_angle(ax, pt_d, pt_a, pt_m, span, label=f"{adm_ang}°")
            _draw_angle(ax, pt_m, pt_c, pt_d, span, label="?")

            p1 = "Straight line PCQ is parallel to straight line ADB. What is the marked angle PCM at point C, in degrees?"
            p2 = (
                "Straight line PCQ is parallel to straight line ADB, and vertex M lies between the two parallel lines. "
                "Find the interior bend angle CMD marked '?' at vertex M, in degrees."
            )
            direct = float(pcm_ang)
            multi = float(cmd_ang)
            meta = {
                "subtype": "parallel_zigzag_obtuse",
                "angle_PCM": pcm_ang,
                "angle_ADM": adm_ang,
                "angle_CMD": cmd_ang,
                "vertices": {k: [v[0], v[1]] for k, v in verts.items()},
            }

        elif t == 3:
            # Bowtie crossing AC x BD at O: right angle ABO = 90 and OC = OD
            theta = rng.randrange(42, 56, 2)
            a_ang = 90 - theta
            ocd_ang = (180 - theta) // 2

            pt_o = (0.0, 0.0)
            pt_a = (-1.0, 0.0)
            pt_c = (1.05, 0.0)
            th_r = math.radians(theta)
            rb = math.cos(th_r)
            pt_b = (-rb * math.cos(th_r), rb * math.sin(th_r))
            pt_d = (1.05 * math.cos(th_r), -1.05 * math.sin(th_r))
            verts = {"A": pt_a, "B": pt_b, "C": pt_c, "D": pt_d, "O": pt_o}

            cx, cy, span = _compute_geometry_span(list(verts.values()))
            ax.plot([pt_a[0], pt_c[0]], [pt_a[1], pt_c[1]], color="#1F4E79", linewidth=2.2)
            ax.plot([pt_b[0], pt_d[0]], [pt_b[1], pt_d[1]], color="#1F4E79", linewidth=2.2)
            ax.plot([pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]], color="#1F4E79", linewidth=2.2)
            ax.plot([pt_c[0], pt_d[0]], [pt_c[1], pt_d[1]], color="#1F4E79", linewidth=2.2)
            _draw_tick_mark(ax, pt_o, pt_c, span)
            _draw_tick_mark(ax, pt_o, pt_d, span)
            _draw_vertex_name(ax, pt_a, [pt_o, pt_b], "A", span)
            _draw_vertex_name(ax, pt_b, [pt_a, pt_o], "B", span)
            _draw_vertex_name(ax, pt_c, [pt_o, pt_d], "C", span)
            _draw_vertex_name(ax, pt_d, [pt_o, pt_c], "D", span)
            _draw_vertex_name(ax, pt_o, [pt_a, pt_d], "O", span)
            _draw_angle(ax, pt_b, pt_a, pt_o, span, label=None, right_angle=True)
            _draw_angle(ax, pt_a, pt_o, pt_b, span, label=f"{a_ang}°")
            _draw_angle(ax, pt_c, pt_o, pt_d, span, label="?")

            p1 = (
                "Straight segments AC and BD intersect at point O, angle ABO is a right angle, and segments OC and OD "
                "have tick marks indicating equal length. What is the marked interior angle BAO at vertex A, in degrees?"
            )
            p2 = (
                "Straight segments AC and BD intersect at point O, angle ABO is a right angle, and segments OC and OD "
                "have tick marks indicating equal length. Find the interior angle OCD marked '?' at vertex C, in degrees."
            )
            direct = float(a_ang)
            multi = float(ocd_ang)
            meta = {
                "subtype": "bowtie_right_isosceles",
                "angle_BAO": a_ang,
                "angle_AOB": theta,
                "angle_OCD": ocd_ang,
                "vertices": {k: [v[0], v[1]] for k, v in verts.items()},
            }

        elif t == 4:
            # Three-triangle partition of ABC with D on AC, E on BC, and BD = BE
            a_ang = rng.randint(44, 54)
            c_ang = rng.randint(36, 44)
            abd_ang = rng.randint(24, 30)
            if (a_ang + c_ang + abd_ang) % 2 == 1:
                abd_ang += 1
            abc_ang = 180 - a_ang - c_ang
            dbe_ang = abc_ang - abd_ang
            bed_ang = (180 - dbe_ang) // 2
            dec_ang = 180 - bed_ang

            pt_a = (0.0, 0.0)
            pt_c = (1.3, 0.0)
            ab = 1.3 * math.sin(math.radians(c_ang)) / math.sin(math.radians(abc_ang))
            pt_b = (ab * math.cos(math.radians(a_ang)), ab * math.sin(math.radians(a_ang)))
            ad = ab * math.sin(math.radians(abd_ang)) / math.sin(math.radians(180 - a_ang - abd_ang))
            pt_d = (ad, 0.0)
            bd = math.hypot(pt_b[0] - pt_d[0], pt_b[1] - pt_d[1])
            u_bc = _unit(pt_c[0] - pt_b[0], pt_c[1] - pt_b[1])
            pt_e = (pt_b[0] + bd * u_bc[0], pt_b[1] + bd * u_bc[1])
            verts = {"A": pt_a, "B": pt_b, "C": pt_c, "D": pt_d, "E": pt_e}

            cx, cy, span = _compute_geometry_span(list(verts.values()))
            ax.plot([pt_a[0], pt_b[0], pt_c[0], pt_a[0]], [pt_a[1], pt_b[1], pt_c[1], pt_a[1]], color="#1F4E79", linewidth=2.2)
            ax.plot([pt_b[0], pt_d[0], pt_e[0]], [pt_b[1], pt_d[1], pt_e[1]], color="#1F4E79", linewidth=2.0)
            _draw_tick_mark(ax, pt_b, pt_d, span, pos=0.62)
            _draw_tick_mark(ax, pt_b, pt_e, span, pos=0.52)
            _draw_vertex_name(ax, pt_a, [pt_c, pt_b], "A", span)
            _draw_vertex_name(ax, pt_b, [pt_a, pt_d, pt_c], "B", span)
            _draw_vertex_name(ax, pt_c, [pt_a, pt_b], "C", span)
            _draw_vertex_name(ax, pt_d, [pt_a, pt_c, pt_b, pt_e], "D", span)
            _draw_vertex_name(ax, pt_e, [pt_b, pt_c, pt_d], "E", span)
            _draw_angle(ax, pt_a, pt_c, pt_b, span, label=f"{a_ang}°")
            _draw_angle(ax, pt_b, pt_a, pt_d, span, label=f"{abd_ang}°")
            _draw_angle(ax, pt_c, pt_a, pt_e, span, label=f"{c_ang}°")
            _draw_angle(ax, pt_e, pt_d, pt_c, span, label="?")

            p1 = (
                "Point D lies on straight segment AC, point E lies on straight segment BC, and segments BD and BE "
                "have tick marks indicating equal length. What is the marked interior angle ACB at vertex C, in degrees?"
            )
            p2 = (
                "Point D lies on straight segment AC, point E lies on straight segment BC, and segments BD and BE "
                "have tick marks indicating equal length. Find the angle DEC marked '?' at point E, in degrees."
            )
            direct = float(c_ang)
            multi = float(dec_ang)
            meta = {
                "subtype": "three_triangle_isosceles_chain",
                "angle_BAC": a_ang,
                "angle_ABD": abd_ang,
                "angle_ACB": c_ang,
                "angle_DEC": dec_ang,
                "vertices": {k: [v[0], v[1]] for k, v in verts.items()},
            }

        else:
            # Parallel lines PCQE || ADB with right angle ACD = 90 and wedge CDE = d_ang, find obtuse QED
            a_ang = rng.randint(48, 60)
            d_ang = rng.randint(68, 82)
            edb_ang = 90 + a_ang - d_ang
            qed_ang = 180 - edb_ang

            pt_a = (0.0, 0.0)
            xc = 1.0 / math.tan(math.radians(a_ang))
            pt_c = (xc, 1.0)
            xd = xc + math.tan(math.radians(a_ang))
            pt_d = (xd, 0.0)
            xe = xd + 1.0 / math.tan(math.radians(edb_ang))
            pt_e = (xe, 1.0)
            pt_p = (-0.42, 1.0)
            pt_q = (xe + 0.78, 1.0)
            pt_b = (xe + 0.78, 0.0)
            verts = {"A": pt_a, "B": pt_b, "C": pt_c, "D": pt_d, "E": pt_e, "P": pt_p, "Q": pt_q}

            cx, cy, span = _compute_geometry_span(list(verts.values()))
            ax.plot([pt_p[0], pt_q[0]], [pt_p[1], pt_q[1]], color="#1F4E79", linewidth=2.1)
            ax.plot([pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]], color="#1F4E79", linewidth=2.1)
            ax.plot([pt_a[0], pt_c[0], pt_d[0], pt_e[0]], [pt_a[1], pt_c[1], pt_d[1], pt_e[1]], color="#1F4E79", linewidth=2.2)
            _draw_parallel_mark(ax, pt_c, pt_e, span, pos=0.50)
            _draw_parallel_mark(ax, pt_a, pt_d, span, pos=0.50)
            _draw_vertex_name(ax, pt_p, [pt_c], "P", span)
            _draw_vertex_name(ax, pt_c, [pt_p, pt_e, pt_a, pt_d], "C", span)
            _draw_vertex_name(ax, pt_e, [pt_c, pt_q, pt_d], "E", span)
            _draw_vertex_name(ax, pt_q, [pt_e], "Q", span)
            _draw_vertex_name(ax, pt_a, [pt_d, pt_c], "A", span)
            _draw_vertex_name(ax, pt_d, [pt_a, pt_b, pt_c, pt_e], "D", span)
            _draw_vertex_name(ax, pt_b, [pt_d], "B", span)
            _draw_angle(ax, pt_a, pt_d, pt_c, span, label=f"{a_ang}°")
            _draw_angle(ax, pt_c, pt_a, pt_d, span, label=None, right_angle=True)
            _draw_angle(ax, pt_d, pt_c, pt_e, span, label=f"{d_ang}°")
            _draw_angle(ax, pt_e, pt_q, pt_d, span, label="?")

            p1 = (
                "Straight line PCQE is parallel to straight line ADB, and angle ACD is a right angle. "
                "What is the marked angle CDE at point D, in degrees?"
            )
            p2 = (
                "Straight line PCQE is parallel to straight line ADB, and angle ACD is a right angle. "
                "Find the angle QED marked '?' at point E, in degrees."
            )
            direct = float(d_ang)
            multi = float(qed_ang)
            meta = {
                "subtype": "parallel_right_w_chain",
                "angle_CAD": a_ang,
                "angle_CDE": d_ang,
                "angle_QED": qed_ang,
                "vertices": {k: [v[0], v[1]] for k, v in verts.items()},
            }

        ax.set_xlim(cx - 0.5 * span, cx + 0.5 * span)
        ax.set_ylim(cy - 0.5 * span, cy + 0.5 * span)
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
        _save_rgb_figure(fig, path)

    return p1, direct, p2, multi, meta


def generate_dataset(
    out_dir,
    seed: int = 20261123,
    n_per_type: int = 12,
    *,
    profile: str = "smoke",
    seed_mode: str = "sequential",
) -> dict:
    if n_per_type <= 0:
        raise ValueError("n_per_type must be positive")
    if profile not in ("smoke", "synth"):
        raise ValueError(f"Unsupported dataset profile: {profile}")
    if seed_mode not in ("sequential", "per_item"):
        raise ValueError(f"Unsupported seed_mode: {seed_mode}")

    out = Path(out_dir)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    seq_rng = random.Random(seed)
    rows: list[Question] = []

    for modality in ("chart", "geometry"):
        for i in range(n_per_type):
            image_id = f"{modality}-{i:03d}"
            rel_path = f"images/{image_id}.png"
            full_path = out / rel_path
            item_rng = (
                seq_rng
                if seed_mode == "sequential"
                else random.Random(f"{seed}:{profile}:{modality}:{i}")
            )

            if profile == "smoke":
                sub_idx = i % 6
                template = f"{modality}-v1-{sub_idx}"
                difficulty = "standard"
                if modality == "chart":
                    p1, direct, p2, multi, common = _render_chart(sub_idx, item_rng, full_path)
                    unit = "units"
                else:
                    p1, direct, p2, multi, common = _render_geometry(sub_idx, item_rng, full_path)
                    unit = "degrees"
            else:
                sub_idx = (i // 2) % 6
                is_hard = (i % 2) == 1
                version = "v2" if is_hard else "v1"
                difficulty = "hard" if is_hard else "standard"
                template = f"{modality}-{version}-{sub_idx}"
                if modality == "chart":
                    renderer = _render_chart_v2 if is_hard else _render_chart
                    p1, direct, p2, multi, common = renderer(sub_idx, item_rng, full_path)
                    unit = "units"
                else:
                    renderer = _render_geometry_v2 if is_hard else _render_geometry
                    p1, direct, p2, multi, common = renderer(sub_idx, item_rng, full_path)
                    unit = "degrees"

            pix = pixel_sha256(full_path)
            extra_meta = {"difficulty": difficulty} if profile == "synth" else {}
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
                        metadata={"seed": seed, **extra_meta, **common, "pixel_sha256": pix},
                    )
                )

    manifest = {
        "schema_version": 2,
        "seed": seed,
        "image_count": n_per_type * 2,
        "question_count": len(rows),
        "questions": [q.to_dict() for q in rows],
    }
    if profile != "smoke" or seed_mode != "sequential":
        manifest["profile"] = profile
        manifest["seed_mode"] = seed_mode
    atomic_write_text(out / "questions.json", json.dumps(manifest, indent=2) + "\n")
    return manifest
