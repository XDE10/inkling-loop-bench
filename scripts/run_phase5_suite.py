"""Execute Phase 5 Suite:
1. Study 5A (`effort48`): Small Reasoning Effort × Pass-Depth Matrix
   (`reasoning_effort` in {none, low, medium, high} × `loop_num_passes` in {1, 3, 5})
   on a deterministic 48-question balanced synthetic subset (1 image per all 24 v1+v2 templates).
2. Study 5B (`chartqa80`): One-source real-world holdout (`HuggingFaceM4/ChartQA` @ `b605b6e08b`,
   40 real-world chart clusters × 2 human-authored numeric questions = 80 questions) across
   `loop_num_passes` in {1, 2, 3, 5}, reported strictly separately per `docs/FINAL_PLAN.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from loopbench.backends import create_backend
from loopbench.backends.transformers_backend import get_model_cache_stats
from loopbench.config import load_config
from loopbench.data.generate import generate_dataset
from loopbench.data.holdout import build_chartqa_holdout, validate_holdout_dataset
from loopbench.data.validate import validate_dataset
from loopbench.metrics import summarize
from loopbench.report import build_phase5_reports
from loopbench.runner import run
from loopbench.storage import atomic_write_text, load_rows
from run_phase3_smoke48 import (
    _make_progress_printer,
    collect_env_fingerprint,
    write_warmup_subset,
)

DEFAULT_EFFORTS = ("none", "low", "medium", "high")
DEFAULT_EFFORT_PASSES = (1, 3, 5)
DEFAULT_HOLDOUT_PASSES = (1, 2, 3, 5)


def prepare_effort_dataset(out_dir: Path, seed: int = 20261124, n_per_type: int = 12) -> Path:
    ds_dir = out_dir / f"dataset_effort{n_per_type * 4}"
    if not (ds_dir / "questions.json").is_file():
        generate_dataset(
            ds_dir,
            seed=seed,
            n_per_type=n_per_type,
            profile="synth",
            seed_mode="per_item",
        )
    valid = validate_dataset(ds_dir, expected_images=n_per_type * 2, expected_questions=n_per_type * 4)
    print(f"EFFORT_DATASET_VALID (n={n_per_type*4}):", json.dumps(valid), flush=True)
    return ds_dir


def prepare_holdout_dataset(out_dir: Path, seed: int = 20261125, n_images: int = 40, prebuilt_dir: Path | None = None) -> Path:
    ds_dir = prebuilt_dir if prebuilt_dir is not None else (out_dir / f"dataset_chartqa{n_images * 2}")
    if not (ds_dir / "questions.json").is_file():
        build_chartqa_holdout(ds_dir, seed=seed, n_images=n_images)
    valid = validate_holdout_dataset(ds_dir, expected_images=n_images, expected_questions=n_images * 2)
    print(f"HOLDOUT_DATASET_VALID (n={n_images*2}):", json.dumps(valid), flush=True)
    return ds_dir


def _verify_layers(rows: list[dict], passes: int, backend_name: str, label: str) -> None:
    if backend_name != "transformers":
        return
    expected = 24 + 12 * (int(passes) - 1)
    bad = [r["question_id"] for r in rows if r.get("effective_layers_verified") != expected]
    if bad:
        raise RuntimeError(f"Layer verification mismatch on {label} (expected {expected}): {bad[:5]}")


def run_effort_cell(
    base_cfg: dict,
    ds_dir: Path,
    out_dir: Path,
    effort: str,
    passes: int,
    scale_tag: str = "effort48",
    backend=None,
) -> Path:
    cfg = {
        **base_cfg,
        "reasoning_effort": effort,
        "loop_num_passes": int(passes),
        "image_mode": "correct",
    }
    jsonl_path = out_dir / f"{scale_tag}_{effort}_loop{passes}.jsonl"
    res = run(
        cfg,
        ds_dir,
        jsonl_path,
        backend=backend,
        progress_callback=_make_progress_printer(f"{scale_tag} effort={effort} loop={passes}"),
    )
    rows = load_rows(jsonl_path)
    _verify_layers(rows, passes, cfg["backend"], f"{scale_tag}_{effort}_loop{passes}")
    summ = summarize(rows)
    print(
        f"EFFORT_CELL_DONE effort={effort} loop={passes}:",
        json.dumps({
            "run": res,
            "accuracy": summ["accuracy"],
            "truncated": summ["truncated"],
            "e2e_p50_ms": summ["e2e_p50_ms"],
            "r_tok_mean": summ["reasoning_tokens_mean"],
        }),
        flush=True,
    )
    return jsonl_path


def run_holdout_pass_stage(
    holdout_cfg: dict,
    ds_dir: Path,
    out_dir: Path,
    passes: int,
    holdout_tag: str = "chartqa80",
    backend=None,
) -> Path:
    cfg = {
        **holdout_cfg,
        "loop_num_passes": int(passes),
        "image_mode": "correct",
    }
    jsonl_path = out_dir / f"{holdout_tag}_loop{passes}.jsonl"
    res = run(
        cfg,
        ds_dir,
        jsonl_path,
        backend=backend,
        progress_callback=_make_progress_printer(f"{holdout_tag} loop={passes}"),
    )
    rows = load_rows(jsonl_path)
    _verify_layers(rows, passes, cfg["backend"], f"{holdout_tag}_loop{passes}")
    summ = summarize(rows)
    print(
        f"HOLDOUT_STAGE_DONE {holdout_tag} loop={passes}:",
        json.dumps({
            "run": res,
            "strict_exact_acc": summ["accuracy"],
            "e2e_p50_ms": summ["e2e_p50_ms"],
            "visual_tokens_unique_cnt": len(summ["visual_tokens_unique"]),
        }),
        flush=True,
    )
    return jsonl_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase 5 Effort Matrix & Real-World Holdout Suite")
    ap.add_argument("--effort-config", default=str(ROOT / "configs/phase5_effort.yaml"))
    ap.add_argument("--holdout-config", default=str(ROOT / "configs/phase5_holdout.yaml"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/phase5"))
    ap.add_argument("--effort-n-per-type", type=int, default=12)
    ap.add_argument("--holdout-n-images", type=int, default=40)
    ap.add_argument("--efforts", default="none,low,medium,high")
    ap.add_argument("--effort-passes", default="1,3,5")
    ap.add_argument("--holdout-passes", default="1,2,3,5")
    ap.add_argument("--skip-effort", action="store_true")
    ap.add_argument("--skip-holdout", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    effort_cfg = load_config(args.effort_config)
    holdout_cfg = load_config(args.holdout_config)

    env_path = out_dir / "env_fingerprint.json"
    if not env_path.is_file():
        env_info = collect_env_fingerprint(effort_cfg)
        atomic_write_text(env_path, json.dumps(env_info, indent=2) + "\n")
    else:
        env_info = json.loads(env_path.read_text(encoding="utf-8"))

    active_backend = create_backend(effort_cfg)

    efforts = tuple(x.strip() for x in args.efforts.split(",") if x.strip())
    effort_passes = tuple(int(x.strip()) for x in args.effort_passes.split(",") if x.strip())
    holdout_passes = tuple(int(x.strip()) for x in args.holdout_passes.split(",") if x.strip())

    effort_tag = f"effort{args.effort_n_per_type * 4}"
    holdout_tag = f"chartqa{args.holdout_n_images * 2}"

    effort_grid_map: dict[tuple[str, int], Path] = {}
    holdout_pass_map: dict[int, Path] = {}
    effort_ds = None
    if not args.skip_effort:
        effort_ds = prepare_effort_dataset(out_dir, seed=int(effort_cfg["seed"]), n_per_type=args.effort_n_per_type)
        if not args.no_warmup and effort_cfg["backend"] != "mock":
            w_ds = write_warmup_subset(effort_ds, out_dir / "subset_warmup1")
            run({**effort_cfg, "loop_num_passes": 1, "reasoning_effort": "none", "max_new_tokens": 16}, w_ds, out_dir / "_warmup.jsonl", backend=active_backend)

    # Execute fast suppressed-CoT cells ('none', 'low') first, then real-world ChartQA-80 holdout,
    # then heavier 'medium' and 'high' cells so checkpoints deliver early scientific signal cleanly.
    fast_efforts = [e for e in efforts if e in ("none", "low")]
    heavy_efforts = [e for e in efforts if e not in ("none", "low")]

    if not args.skip_effort and effort_ds is not None:
        for eff in fast_efforts:
            for p in effort_passes:
                effort_grid_map[(eff, p)] = run_effort_cell(
                    effort_cfg, effort_ds, out_dir, effort=eff, passes=p, scale_tag=effort_tag, backend=active_backend
                )

    if not args.skip_holdout:
        holdout_ds = prepare_holdout_dataset(out_dir, seed=int(holdout_cfg["seed"]), n_images=args.holdout_n_images)
        if args.skip_effort and not args.no_warmup and holdout_cfg["backend"] != "mock":
            w_ds = write_warmup_subset(holdout_ds, out_dir / "subset_warmup_holdout")
            run({**holdout_cfg, "loop_num_passes": 1, "max_new_tokens": 16}, w_ds, out_dir / "_warmup_holdout.jsonl", backend=active_backend)
        for p in holdout_passes:
            holdout_pass_map[p] = run_holdout_pass_stage(
                holdout_cfg, holdout_ds, out_dir, passes=p, holdout_tag=holdout_tag, backend=active_backend
            )

    if not args.skip_effort and effort_ds is not None:
        for eff in heavy_efforts:
            for p in effort_passes:
                effort_grid_map[(eff, p)] = run_effort_cell(
                    effort_cfg, effort_ds, out_dir, effort=eff, passes=p, scale_tag=effort_tag, backend=active_backend
                )

    cache_stats = get_model_cache_stats()
    if effort_cfg["backend"] == "transformers" and cache_stats["loads"] > 1:
        raise RuntimeError(f"Zero-reload invariant violated in Phase 5: {cache_stats}")
    atomic_write_text(out_dir / "model_cache_stats.json", json.dumps(cache_stats, indent=2) + "\n")

    build_phase5_reports(
        effort_jsonl_map=effort_grid_map,
        holdout_jsonl_map=holdout_pass_map,
        out_dir=out_dir / "report",
        effort_tag=effort_tag,
        holdout_tag=holdout_tag,
        env_info=env_info,
        cache_stats=cache_stats,
    )
    print("PHASE5_REPORTS_READY:", json.dumps({"cache_stats": cache_stats}), flush=True)


if __name__ == "__main__":
    main()
