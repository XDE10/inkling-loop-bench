import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loopbench.config import load_config
from loopbench.data.generate import generate_dataset
from loopbench.data.validate import validate_dataset
from loopbench.metrics import summarize
from loopbench.runner import resolve_image_path, run
from loopbench.storage import atomic_write_text, load_rows

DIAGNOSTIC_IMAGE_IDS = (
    "chart-000",
    "chart-002",
    "chart-005",
    "geometry-001",
    "geometry-003",
    "geometry-005",
)


def _pkg_ver(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_env_fingerprint(config: dict) -> dict:
    import torch

    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_mb": round(props.total_memory / (1024 * 1024), 2),
            "cuda_runtime": torch.version.cuda,
        }
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
            "nvidia-modelopt": _pkg_ver("nvidia-modelopt"),
            "compressed-tensors": _pkg_ver("compressed-tensors"),
            "kernels": _pkg_ver("kernels"),
            "vllm": _pkg_ver("vllm"),
            "sglang": _pkg_ver("sglang"),
        },
        "pinned_model_id": config["model_id"],
        "pinned_model_revision": config["model_revision"],
    }


def run_serving_backend_gate(time_budget_s: int = 180) -> dict:
    """Time-boxed feasibility check for vLLM / SGLang serving on Colab G4 Blackwell."""
    t0 = time.perf_counter()
    report: dict = {
        "time_budget_s": time_budget_s,
        "docker_available": shutil.which("docker") is not None,
        "candidates": {},
    }

    for pkg, arch_symbol in (("vllm", "Inkling2LoopForConditionalGeneration"), ("sglang", "inkling2")):
        c_start = time.perf_counter()
        ver = _pkg_ver(pkg)
        entry: dict = {"installed_version": ver, "architecture_registered": False, "usable": False, "notes": []}
        if ver is None:
            # Quick dry-run index probe (bounded timeout) without burning heavy compile minutes
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pip", "index", "versions", pkg],
                    capture_output=True,
                    text=True,
                    timeout=min(25, max(5, int(time_budget_s - (time.perf_counter() - t0)))),
                )
                first_line = (proc.stdout or proc.stderr or "").splitlines()[:2]
                entry["pypi_index_probe"] = first_line
            except Exception as e:
                entry["pypi_index_probe"] = [repr(e)]
            if pkg == "sglang":
                entry["notes"].append(
                    "Official SGLang Inkling-2-Mini cookbook specifies unmerged PR branch "
                    "(sgl-project/sglang PR #51926) or dedicated Docker container (lmsysorg/sglang:inkling-2-mini); "
                    "Docker daemon is unavailable inside Colab runtime."
                )
            else:
                entry["notes"].append(
                    "vLLM not preinstalled; checking wheel installability only if within time-box."
                )
        else:
            try:
                if pkg == "vllm":
                    from vllm.model_executor.models import ModelRegistry  # type: ignore

                    supported = ModelRegistry.get_supported_archs()
                    entry["architecture_registered"] = arch_symbol in supported
                    entry["usable"] = bool(entry["architecture_registered"])
                else:
                    import sglang  # type: ignore

                    entry["architecture_registered"] = True
            except Exception as e:
                entry["notes"].append(f"Import/registry check failed: {e!r}")
        entry["elapsed_s"] = round(time.perf_counter() - c_start, 3)
        report["candidates"][pkg] = entry

    any_usable = any(v.get("usable") for v in report["candidates"].values())
    report["selected_backend"] = (
        next(k for k, v in report["candidates"].items() if v.get("usable"))
        if any_usable
        else "transformers"
    )
    report["decision"] = (
        f"Use '{report['selected_backend']}' serving backend."
        if any_usable
        else "Lock official HuggingFace Transformers reference backend for Phase 2/3 correctness and single-stream latency reporting."
    )
    report["total_gate_elapsed_s"] = round(time.perf_counter() - t0, 3)
    return report


def write_subset_manifest(full_dataset: Path, subset_dir: Path, image_ids: tuple[str, ...]) -> Path:
    subset_dir.mkdir(parents=True, exist_ok=True)
    full_data = json.loads((full_dataset / "questions.json").read_text(encoding="utf-8"))
    selected = []
    for q in full_data["questions"]:
        if q["image_id"] in image_ids:
            q_copy = dict(q)
            q_copy["image_path"] = str(resolve_image_path(full_dataset, q["image_path"]))
            selected.append(q_copy)
    manifest = {
        "schema_version": full_data.get("schema_version", 2),
        "seed": full_data["seed"],
        "image_count": len(image_ids),
        "question_count": len(selected),
        "questions": selected,
    }
    atomic_write_text(subset_dir / "questions.json", json.dumps(manifest, indent=2) + "\n")
    return subset_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Execute Phase 2 Colab G4 GPU probe & 12-question diagnostic")
    p.add_argument("--config", default=str(ROOT / "configs/smoke.yaml"))
    p.add_argument("--out", default=str(ROOT / "artifacts/phase2"))
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--skip-diagnostic", action="store_true")
    args = p.parse_args()

    base_cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 0. Dataset generation + validation
    ds_dir = out / "dataset_smoke48"
    if not (ds_dir / "questions.json").is_file():
        generate_dataset(ds_dir, seed=args.seed, n_per_type=12)
    ds_valid = validate_dataset(ds_dir, expected_images=24, expected_questions=48)
    print("DATASET_VALID:", ds_valid, flush=True)

    # 1. Environment fingerprint & Serving gate
    env_info = collect_env_fingerprint(base_cfg)
    atomic_write_text(out / "env_fingerprint.json", json.dumps(env_info, indent=2) + "\n")
    print("ENV_FINGERPRINT:", json.dumps(env_info), flush=True)

    gate_info = run_serving_backend_gate(time_budget_s=120)
    atomic_write_text(out / "serving_gate.json", json.dumps(gate_info, indent=2) + "\n")
    print("SERVING_GATE:", json.dumps(gate_info), flush=True)

    # 2. Single-image probe (chart-000 & geometry-000 at loop=1 vs loop=3)
    probe_ds = write_subset_manifest(ds_dir, out / "subset_probe4", ("chart-000", "geometry-000"))

    # Warmup 1 short call to initialize CUDA allocator/kernels cleanly before timed probe
    warmup_ds = write_subset_manifest(ds_dir, out / "subset_warmup", ("chart-000",))
    warmup_cfg = {**base_cfg, "loop_num_passes": 1, "max_new_tokens": 32}
    run(warmup_cfg, warmup_ds, out / "_warmup.jsonl")
    print("WARMUP_DONE", flush=True)

    probe_summaries = {}
    for passes in (1, 3):
        p_cfg = {**base_cfg, "loop_num_passes": passes, "max_new_tokens": 2048}
        p_jsonl = out / f"probe_loop{passes}.jsonl"
        res = run(p_cfg, probe_ds, p_jsonl)
        rows = load_rows(p_jsonl)
        summ = summarize(rows)
        summ["avg_reasoning_tokens"] = (
            sum(r["reasoning_tokens"] or 0 for r in rows) / len(rows) if rows else 0.0
        )
        summ["avg_answer_tokens"] = (
            sum(r["answer_tokens"] or 0 for r in rows) / len(rows) if rows else 0.0
        )
        summ["effective_layers_verified"] = [r.get("effective_layers_verified") for r in rows]
        summ["visual_tokens_sample"] = rows[0].get("visual_tokens") if rows else None
        summ["image_grid_thw_sample"] = rows[0].get("image_grid_thw") if rows else None
        probe_summaries[f"loop_{passes}"] = {"run": res, "summary": summ}
        print(f"PROBE_LOOP_{passes}:", json.dumps(summ), flush=True)

    atomic_write_text(out / "single_image_probe.json", json.dumps(probe_summaries, indent=2) + "\n")

    # 3. 12-question decoding-policy diagnostic (at loop_num_passes=3, medium effort)
    if not args.skip_diagnostic:
        diag_ds = write_subset_manifest(ds_dir, out / "subset_diag12", DIAGNOSTIC_IMAGE_IDS)
        policies = {
            "greedy_512": {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0, "max_new_tokens": 512},
            "greedy_2048": {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0, "max_new_tokens": 2048},
            "sampled_t1_2048": {"temperature": 1.0, "top_p": 0.95, "repetition_penalty": 1.0, "max_new_tokens": 2048},
        }
        diag_report = {}
        for pol_name, overrides in policies.items():
            d_cfg = {**base_cfg, "loop_num_passes": 3, **overrides}
            d_jsonl = out / f"diag12_{pol_name}.jsonl"
            d_res = run(d_cfg, diag_ds, d_jsonl)
            d_rows = load_rows(d_jsonl)
            d_summ = summarize(d_rows)
            d_summ["mean_reasoning_tokens"] = (
                round(sum(r["reasoning_tokens"] or 0 for r in d_rows) / len(d_rows), 1)
                if d_rows
                else 0.0
            )
            d_summ["max_reasoning_tokens"] = (
                max((r["reasoning_tokens"] or 0) for r in d_rows) if d_rows else 0
            )
            diag_report[pol_name] = {"overrides": overrides, "run": d_res, "summary": d_summ}
            print(f"DIAG_{pol_name}:", json.dumps(d_summ), flush=True)

        atomic_write_text(out / "decoding_diagnostic_12q.json", json.dumps(diag_report, indent=2) + "\n")


if __name__ == "__main__":
    main()
