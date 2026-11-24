import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from loopbench.storage import append_jsonl
from run_phase6_ablations import build_offline_oracle_suite


def test_phase6_offline_oracle_suite_builder(tmp_path):
    p4_dir = tmp_path / "p4"
    p5_dir = tmp_path / "p5"
    p4_dir.mkdir()
    p5_dir.mkdir()

    # Create synthetic Phase 4 rows across loop=1,2,3,5
    for p in (1, 2, 3, 5):
        jl = p4_dir / f"synth240_loop{p}.jsonl"
        for i in range(8):
            is_hard = i >= 4
            ok = (not is_hard) or (p >= 3)
            append_jsonl(jl, {
                "config_id": f"cfg-{p}",
                "question_id": f"q-{i}",
                "image_id": f"img-{i//2}",
                "difficulty": "hard" if is_hard else "standard",
                "task_type": "multi_step" if i % 2 else "direct",
                "outcome": "correct" if ok else "wrong_answer",
                "correct": ok,
                "e2e_ms": 1000.0 * p,
                "reasoning_tokens": 200,
                "is_mock": True,
            })

    out_dir = tmp_path / "p6_out"
    res = build_offline_oracle_suite(p4_dir, p5_dir, out_dir)
    assert res["mock"] is True
    assert "phase4_synth240_pass_oracle" in res["suites"]
    ro = res["suites"]["phase4_synth240_pass_oracle"]["router_oracle"]
    assert ro["accuracy"] == 1.0
    # Half solved at loop=1 (1000ms), half solved at loop=3 (3000ms) -> mean 2000ms vs loop=3 fixed 3000ms (1.5x speedup)
    assert abs(ro["e2e_mean_ms"] - 2000.0) < 1e-6
    assert abs(ro["mean_speedup_vs_best_fixed"] - 1.5) < 1e-4
    assert (out_dir / "phase6_oracle_summary.json").is_file()
    assert (out_dir / "phase6_oracle_summary.csv").is_file()
    assert (out_dir / "pareto_phase6_oracle_mock.png").is_file()
