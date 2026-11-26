import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase7_gpu_systems import render_gpu_systems_figure


def test_gpu_systems_figure_render(tmp_path: Path):
    sample = {
        "model_id": "thinkingmachines/TML-d24-varloop-research",
        "model_revision": "c90921bd5f78df6c8168173b9f8a3a850320c594",
        "gpu": "Mock Blackwell",
        "stage_breakdown": [
            {
                "loop_num_passes": 1,
                "effective_layers_expected": 24,
                "decode_tpot_wall_ms": 34.0,
                "decode_prelude_ms": 8.0,
                "decode_core_ms": 17.0,
                "decode_coda_ms": 7.5,
                "decode_overhead_ms": 1.5,
                "core_share_of_decode_pct": 50.0,
                "kv_cache_after_decode_mb": 48.0,
                "kv_cache_delta_per_tok_kb": 96.0,
            },
            {
                "loop_num_passes": 3,
                "effective_layers_expected": 48,
                "decode_tpot_wall_ms": 83.0,
                "decode_prelude_ms": 8.2,
                "decode_core_ms": 64.8,
                "decode_coda_ms": 7.8,
                "decode_overhead_ms": 2.2,
                "core_share_of_decode_pct": 78.1,
                "kv_cache_after_decode_mb": 96.0,
                "kv_cache_delta_per_tok_kb": 192.0,
            },
            {
                "loop_num_passes": 5,
                "effective_layers_expected": 72,
                "decode_tpot_wall_ms": 132.0,
                "decode_prelude_ms": 8.3,
                "decode_core_ms": 113.5,
                "decode_coda_ms": 7.9,
                "decode_overhead_ms": 2.3,
                "core_share_of_decode_pct": 86.0,
                "kv_cache_after_decode_mb": 144.0,
                "kv_cache_delta_per_tok_kb": 288.0,
            },
        ],
        "batch_throughput": [
            {"loop_num_passes": 1, "batch_size": 1, "agg_decode_tok_per_s": 29.0, "oom": False},
            {"loop_num_passes": 1, "batch_size": 2, "agg_decode_tok_per_s": 54.0, "oom": False},
            {"loop_num_passes": 1, "batch_size": 4, "agg_decode_tok_per_s": 98.0, "oom": False},
            {"loop_num_passes": 3, "batch_size": 1, "agg_decode_tok_per_s": 12.0, "oom": False},
            {"loop_num_passes": 3, "batch_size": 2, "agg_decode_tok_per_s": 23.0, "oom": False},
            {"loop_num_passes": 3, "batch_size": 4, "agg_decode_tok_per_s": 42.0, "oom": False},
        ],
    }
    out_png = tmp_path / "gpu_systems_microbench.png"
    render_gpu_systems_figure(sample, out_png)
    assert out_png.is_file() and out_png.stat().st_size > 10_000
