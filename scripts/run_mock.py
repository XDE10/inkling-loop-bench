import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loopbench.config import load_config
from loopbench.data.generate import generate_dataset
from loopbench.data.validate import validate_dataset
from loopbench.report import build_report
from loopbench.runner import run


def main() -> None:
    p = argparse.ArgumentParser(description="Run Phase 1 CPU mock LoopBench pipeline")
    p.add_argument("--config", default=str(ROOT / "configs/mock.yaml"))
    p.add_argument("--out", default=str(ROOT / "artifacts/mock"))
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--n-per-type", type=int, default=12)
    args = p.parse_args()

    out = Path(args.out)
    dataset = out / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    generate_dataset(dataset, seed=args.seed, n_per_type=args.n_per_type)
    print(
        "DATA:",
        validate_dataset(
            dataset,
            expected_images=args.n_per_type * 2,
            expected_questions=args.n_per_type * 4,
        ),
    )
    cfg = load_config(args.config)
    result = run(cfg, dataset, out / "results.jsonl")
    print("RUN:", result)
    print("REPORT:", build_report(out / "results.jsonl", out / "report"))


if __name__ == "__main__":
    main()
