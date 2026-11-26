import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loopbench.backends import create_backend
from loopbench.backends.transformers_backend import get_model_cache_stats
from loopbench.config import load_config
from loopbench.data.generate import generate_dataset
from loopbench.data.validate import validate_dataset
from loopbench.metrics import summarize
from loopbench.report import build_phase3_report
from loopbench.runner import resolve_image_path, run
from loopbench.storage import atomic_write_text, load_rows


def _pkg_ver(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_env_fingerprint(config: dict) -> dict:
    gpu_info = {}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_info = {
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": f"{props.major}.{props.minor}",
                "total_vram_mb": round(props.total_memory / (1024 * 1024), 2),
                "cuda_runtime": torch.version.cuda,
            }
    except ImportError:
        pass

    smi = ""
    if shutil.which("nvidia-smi"):
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                text=True,
                timeout=10,
            ).strip()
        except Exception as e:
            smi = f"error: {e!r}"

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": gpu_info,
        "nvidia_smi": smi,
        "packages": {
            "torch": _pkg_ver("torch"),
            "transformers": _pkg_ver("transformers"),
            "accelerate": _pkg_ver("accelerate"),
            "kernels": _pkg_ver("kernels"),
        },
        "pinned_model_id": config["model_id"],
        "pinned_model_revision": config["model_revision"],
    }


def write_warmup_subset(full_dataset: Path, subset_dir: Path) -> Path:
    subset_dir.mkdir(parents=True, exist_ok=True)
    full_data = json.loads((full_dataset / "questions.json").read_text(encoding="utf-8"))
    first_q = dict(full_data["questions"][0])
    first_q["image_path"] = str(resolve_image_path(full_dataset, first_q["image_path"]))
    manifest = {
        "schema_version": full_data.get("schema_version", 2),
        "seed": full_data.get("seed", 0),
        "image_count": 1,
        "question_count": 1,
        "questions": [first_q],
    }
    atomic_write_text(subset_dir / "questions.json", json.dumps(manifest, indent=2) + "\n")
    return subset_dir


def _make_progress_printer(tag: str):
    def _cb(idx: int, total: int, row: dict) -> None:
        e2e_s = (row.get("e2e_ms") or 0.0) / 1000.0
        ttft = row.get("ttft_ms") or 0.0
        layers = row.get("effective_layers_verified")
        r_tok = row.get("reasoning_tokens")
        a_tok = row.get("answer_tokens")
        print(
            f"[{tag} {idx:02d}/{total:02d}] {row['question_id']} -> {row['outcome']} "
            f"(pred={row.get('predicted')}, exp={row['expected']}, "
            f"ttft={ttft:.1f}ms, e2e={e2e_s:.2f}s, r_tok={r_tok}, a_tok={a_tok}, layers={layers})",
            flush=True,
        )

    return _cb


def prepare_phase3_workspace(
    config_path: str | Path,
    out_dir: str | Path,
    seed: int = 20261123,
    n_per_type: int = 12,
    do_warmup: bool = True,
    backend=None,
) -> tuple[dict, Path, Path, object]:
    base_cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds_dir = out / "dataset_smoke48"
    if not (ds_dir / "questions.json").is_file():
        generate_dataset(ds_dir, seed=seed, n_per_type=n_per_type)
    ds_valid = validate_dataset(
        ds_dir,
        expected_images=n_per_type * 2,
        expected_questions=n_per_type * 4,
    )
    print("DATASET_VALID:", json.dumps(ds_valid), flush=True)

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


def run_pass_depth_stage(
    base_cfg: dict,
    ds_dir: Path,
    out_dir: Path,
    passes: int,
    backend=None,
) -> dict:
    cfg = {**base_cfg, "loop_num_passes": int(passes), "image_mode": "correct"}
    jsonl_path = out_dir / f"smoke48_loop{passes}.jsonl"
    res = run(
        cfg,
        ds_dir,
        jsonl_path,
        backend=backend,
        progress_callback=_make_progress_printer(f"loop={passes} correct"),
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
                f"Effective layer verification mismatch for loop={passes} (expected {expected_layers}): {bad}"
            )
    summ = summarize(rows)
    print(f"STAGE_DONE loop={passes}:", json.dumps({"run": res, "summary": summ}), flush=True)
    return {"run": res, "summary": summ, "jsonl": str(jsonl_path)}


def run_leakage_stage(
    base_cfg: dict,
    ds_dir: Path,
    out_dir: Path,
    image_mode: str,
    passes: int = 3,
    backend=None,
) -> dict:
    if image_mode not in ("blank", "mismatched"):
        raise ValueError(f"Leakage stage expects blank or mismatched, got {image_mode}")
    cfg = {**base_cfg, "loop_num_passes": int(passes), "image_mode": image_mode}
    jsonl_path = out_dir / f"leakage_{image_mode}_loop{passes}.jsonl"
    res = run(
        cfg,
        ds_dir,
        jsonl_path,
        backend=backend,
        progress_callback=_make_progress_printer(f"loop={passes} {image_mode}"),
    )
    rows = load_rows(jsonl_path)
    summ = summarize(rows)
    print(
        f"LEAKAGE_DONE mode={image_mode} loop={passes}:",
        json.dumps({"run": res, "summary": summ}),
        flush=True,
    )
    return {"run": res, "summary": summ, "jsonl": str(jsonl_path)}


def finalize_phase3(
    out_dir: str | Path,
    passes_list: tuple[int, ...] = (1, 2, 3, 5),
    leakage_passes: int = 3,
    include_leakage: bool = True,
) -> dict:
    out = Path(out_dir)
    env_path = out / "env_fingerprint.json"
    env_info = json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else None
    cache_stats = get_model_cache_stats()
    atomic_write_text(out / "model_cache_stats.json", json.dumps(cache_stats, indent=2) + "\n")

    pass_map = {p: out / f"smoke48_loop{p}.jsonl" for p in passes_list if (out / f"smoke48_loop{p}.jsonl").is_file()}
    leak_map = {}
    if include_leakage:
        for mode in ("blank", "mismatched"):
            cand = out / f"leakage_{mode}_loop{leakage_passes}.jsonl"
            if cand.is_file():
                leak_map[mode] = cand

    summary = build_phase3_report(
        pass_jsonl_map=pass_map,
        leakage_jsonl_map=leak_map,
        out_dir=out / "report",
        env_info=env_info,
        cache_stats=cache_stats,
        leakage_passes=leakage_passes,
    )
    print("PHASE3_REPORT_READY:", json.dumps({"cache_stats": cache_stats, "passes": list(pass_map)}), flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Execute Phase 3 smoke-48 pass-depth scaling & visual leakage suite")
    p.add_argument("--config", default=str(ROOT / "configs/smoke.yaml"))
    p.add_argument("--out", default=str(ROOT / "artifacts/phase3"))
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--n-per-type", type=int, default=12)
    p.add_argument("--passes", default="1,2,3,5")
    p.add_argument("--leakage-passes", type=int, default=3)
    p.add_argument("--skip-leakage", action="store_true")
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args()

    passes_list = tuple(int(x.strip()) for x in args.passes.split(",") if x.strip())
    base_cfg, out_dir, ds_dir, active_backend = prepare_phase3_workspace(
        config_path=args.config,
        out_dir=args.out,
        seed=args.seed,
        n_per_type=args.n_per_type,
        do_warmup=not args.no_warmup,
    )

    for passes in passes_list:
        run_pass_depth_stage(base_cfg, ds_dir, out_dir, passes=passes, backend=active_backend)

    if not args.skip_leakage:
        for mode in ("blank", "mismatched"):
            run_leakage_stage(
                base_cfg,
                ds_dir,
                out_dir,
                image_mode=mode,
                passes=args.leakage_passes,
                backend=active_backend,
            )

    if base_cfg["backend"] == "transformers":
        stats = get_model_cache_stats()
        if stats["loads"] > 1:
            raise RuntimeError(f"Zero-reload invariant violated: expected <= 1 model load, got {stats}")

    finalize_phase3(
        out_dir=out_dir,
        passes_list=passes_list,
        leakage_passes=args.leakage_passes,
        include_leakage=not args.skip_leakage,
    )


if __name__ == "__main__":
    main()
