import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from loopbench.backends import create_backend
from loopbench.backends.transformers_backend import get_model_cache_stats
from loopbench.config import load_config
from loopbench.data.generate import generate_dataset
from loopbench.data.validate import validate_dataset
from loopbench.metrics import (
    evaluate_synth320_extension_rule,
    paired_cluster_bootstrap_diff_ci,
    summarize,
)
from loopbench.report import build_phase4_report
from loopbench.runner import run
from loopbench.storage import atomic_write_text, load_rows
from run_phase3_smoke48 import (
    _make_progress_printer,
    collect_env_fingerprint,
    write_warmup_subset,
)


def prepare_phase4_workspace(
    config_path: str | Path,
    out_dir: str | Path,
    n_per_type: int = 60,
    do_warmup: bool = True,
    backend=None,
) -> tuple[dict, Path, Path, object]:
    base_cfg = load_config(config_path)
    seed = int(base_cfg["seed"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    total_q = n_per_type * 4
    ds_dir = out / f"dataset_synth{total_q}"
    if not (ds_dir / "questions.json").is_file():
        generate_dataset(
            ds_dir,
            seed=seed,
            n_per_type=n_per_type,
            profile="synth",
            seed_mode="per_item",
        )
    ds_valid = validate_dataset(
        ds_dir,
        expected_images=n_per_type * 2,
        expected_questions=total_q,
    )
    print(f"DATASET_VALID (synth-{total_q}):", json.dumps(ds_valid), flush=True)

    env_path = out / "env_fingerprint.json"
    if not env_path.is_file():
        env_info = collect_env_fingerprint(base_cfg)
        atomic_write_text(env_path, json.dumps(env_info, indent=2) + "\n")
    else:
        env_info = json.loads(env_path.read_text(encoding="utf-8"))
    print("ENV_FINGERPRINT:", json.dumps(env_info), flush=True)

    active_backend = backend if backend is not None else create_backend(base_cfg)
    if do_warmup and base_cfg["backend"] != "mock":
        warmup_ds = write_warmup_subset(ds_dir, out / "subset_warmup1")
        warmup_cfg = {**base_cfg, "loop_num_passes": 1, "max_new_tokens": 32, "image_mode": "correct"}
        run(warmup_cfg, warmup_ds, out / "_warmup.jsonl", backend=active_backend)
        print("WARMUP_DONE:", json.dumps(get_model_cache_stats()), flush=True)

    return base_cfg, out, ds_dir, active_backend


def run_phase4_pass_stage(
    base_cfg: dict,
    ds_dir: Path,
    out_dir: Path,
    passes: int,
    scale_tag: str = "synth240",
    backend=None,
) -> dict:
    cfg = {**base_cfg, "loop_num_passes": int(passes), "image_mode": "correct"}
    jsonl_path = out_dir / f"{scale_tag}_loop{passes}.jsonl"
    res = run(
        cfg,
        ds_dir,
        jsonl_path,
        backend=backend,
        progress_callback=_make_progress_printer(f"{scale_tag} loop={passes}"),
    )
    rows = load_rows(jsonl_path)
    if cfg["backend"] == "transformers":
        expected_layers = 24 + 12 * (int(passes) - 1)
        bad = [
            r["question_id"]
            for r in rows
            if r.get("effective_layers_verified") != expected_layers
        ]
        if bad:
            raise RuntimeError(
                f"Effective layer verification mismatch for {scale_tag} loop={passes} (expected {expected_layers}): {bad[:5]}"
            )
    summ = summarize(rows)
    print(
        f"STAGE_DONE {scale_tag} loop={passes}:",
        json.dumps(
            {
                "run": res,
                "n": summ["n"],
                "accuracy": summ["accuracy"],
                "ci95": summ["accuracy_cluster_bootstrap_95ci"],
                "by_difficulty": summ["by_difficulty"],
                "e2e_p50_ms": summ["e2e_p50_ms"],
            }
        ),
        flush=True,
    )
    return {"run": res, "summary": summ, "jsonl": str(jsonl_path)}


def compute_stage_extension_decision(
    out_dir: Path,
    passes_list: tuple[int, ...] = (1, 2, 3, 5),
    scale_tag: str = "synth240",
    max_ci_half_width: float = 0.05,
    ambiguous_min_abs_diff: float = 0.025,
) -> dict:
    rows_map = {
        p: load_rows(out_dir / f"{scale_tag}_loop{p}.jsonl")
        for p in passes_list
        if (out_dir / f"{scale_tag}_loop{p}.jsonl").is_file()
    }
    base_p = min(rows_map)
    paired_diffs: dict[str, dict] = {}
    for p in sorted(rows_map):
        if p != base_p:
            paired_diffs[f"loop{p}_minus_loop{base_p}"] = paired_cluster_bootstrap_diff_ci(
                rows_map[base_p], rows_map[p]
            )
    if 3 in rows_map and 5 in rows_map:
        paired_diffs["loop5_minus_loop3"] = paired_cluster_bootstrap_diff_ci(rows_map[3], rows_map[5])
    return evaluate_synth320_extension_rule(
        paired_diffs,
        max_ci_half_width=max_ci_half_width,
        ambiguous_min_abs_diff=ambiguous_min_abs_diff,
    )


def finalize_phase4(
    out_dir: str | Path,
    passes_list: tuple[int, ...] = (1, 2, 3, 5),
    scale_tag: str = "synth240",
    initial_decision: dict | None = None,
    max_ci_half_width: float = 0.05,
    ambiguous_min_abs_diff: float = 0.025,
) -> dict:
    out = Path(out_dir)
    env_path = out / "env_fingerprint.json"
    env_info = json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else None
    cache_stats = get_model_cache_stats()
    atomic_write_text(out / "model_cache_stats.json", json.dumps(cache_stats, indent=2) + "\n")

    pass_map = {
        p: out / f"{scale_tag}_loop{p}.jsonl"
        for p in passes_list
        if (out / f"{scale_tag}_loop{p}.jsonl").is_file()
    }
    summary = build_phase4_report(
        pass_jsonl_map=pass_map,
        out_dir=out / "report",
        scale_tag=scale_tag,
        env_info=env_info,
        cache_stats=cache_stats,
        initial_synth240_decision=initial_decision,
        max_ci_half_width=max_ci_half_width,
        ambiguous_min_abs_diff=ambiguous_min_abs_diff,
    )
    print(
        "PHASE4_REPORT_READY:",
        json.dumps(
            {
                "scale_tag": scale_tag,
                "cache_stats": cache_stats,
                "passes": list(pass_map),
                "extension_decision": summary["uncertainty_extension_decision"],
            }
        ),
        flush=True,
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Execute Phase 4 synth-240 main synthetic study (with predeclared synth-320 rule)")
    p.add_argument("--config", default=str(ROOT / "configs/synth240.yaml"))
    p.add_argument("--out", default=str(ROOT / "artifacts/phase4"))
    p.add_argument("--n-per-type", type=int, default=60)
    p.add_argument("--extension-n-per-type", type=int, default=80)
    p.add_argument("--auto-extend", action="store_true", help="Automatically extend in-place to synth-320 if predeclared CI rule triggers")
    p.add_argument("--max-ci-half-width", type=float, default=0.05)
    p.add_argument("--ambiguous-min-abs-diff", type=float, default=0.025)
    p.add_argument("--passes", default="1,2,3,5")
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args()

    passes_list = tuple(int(x.strip()) for x in args.passes.split(",") if x.strip())
    base_cfg, out_dir, ds_dir, active_backend = prepare_phase4_workspace(
        config_path=args.config,
        out_dir=args.out,
        n_per_type=args.n_per_type,
        do_warmup=not args.no_warmup,
    )

    scale_tag = f"synth{args.n_per_type * 4}"
    for passes in passes_list:
        run_phase4_pass_stage(
            base_cfg,
            ds_dir,
            out_dir,
            passes=passes,
            scale_tag=scale_tag,
            backend=active_backend,
        )

    decision_240 = compute_stage_extension_decision(
        out_dir,
        passes_list=passes_list,
        scale_tag=scale_tag,
        max_ci_half_width=args.max_ci_half_width,
        ambiguous_min_abs_diff=args.ambiguous_min_abs_diff,
    )
    atomic_write_text(
        out_dir / "synth240_extension_decision.json",
        json.dumps(decision_240, indent=2) + "\n",
    )
    print("SYNTH240_EXTENSION_DECISION:", json.dumps(decision_240), flush=True)

    final_scale_tag = scale_tag
    if args.auto_extend and decision_240["trigger_synth320"] and args.extension_n_per_type > args.n_per_type:
        ext_total = args.extension_n_per_type * 4
        final_scale_tag = f"synth{ext_total}"
        print(f"PREDECLARED_UNCERTAINTY_RULE_TRIGGERED -> Extending to {final_scale_tag} reusing prefix rows!", flush=True)
        _, _, ext_ds_dir, _ = prepare_phase4_workspace(
            config_path=args.config,
            out_dir=args.out,
            n_per_type=args.extension_n_per_type,
            do_warmup=False,
            backend=active_backend,
        )
        for passes in passes_list:
            src_jl = out_dir / f"{scale_tag}_loop{passes}.jsonl"
            dst_jl = out_dir / f"{final_scale_tag}_loop{passes}.jsonl"
            if src_jl.is_file() and not dst_jl.is_file():
                shutil.copy2(src_jl, dst_jl)
            run_phase4_pass_stage(
                base_cfg,
                ext_ds_dir,
                out_dir,
                passes=passes,
                scale_tag=final_scale_tag,
                backend=active_backend,
            )

    if base_cfg["backend"] == "transformers":
        stats = get_model_cache_stats()
        if stats["loads"] > 1:
            raise RuntimeError(f"Zero-reload invariant violated: expected <= 1 model load, got {stats}")

    finalize_phase4(
        out_dir=out_dir,
        passes_list=passes_list,
        scale_tag=final_scale_tag,
        initial_decision=decision_240,
        max_ci_half_width=args.max_ci_half_width,
        ambiguous_min_abs_diff=args.ambiguous_min_abs_diff,
    )


if __name__ == "__main__":
    main()
