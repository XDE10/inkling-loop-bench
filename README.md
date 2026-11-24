# LoopBench

**Measuring Accuracy–Latency–Memory Trade-offs of Latent Looping and Explicit Reasoning in Multimodal Inference.**

- **Phase 1 (Signed off & merged)**: Deterministic CPU/mock pipeline (`schema_version: 2`, 12 chart/geometry sub-templates, crash-resilient JSONL resumability, 5-way error taxonomy, cluster bootstrap CI).
- **Phase 2 (Signed off & merged)**: Pinned `thinkingmachines/TML-d24-varloop-research` (`c90921bd5f78df6c8168173b9f8a3a850320c594`), verified `TransformersBackend` on `NVIDIA RTX PRO 6000 Blackwell Server Edition` (96 GB), causally verified `loop=1` vs `loop=3`, completed the Serving Backend Gate and 12-question decoding-policy diagnostic (`reports/phase2_g4_probe.md`).
- **Phase 3 (Signed off on Colab G4 Blackwell)**: Completed full `smoke-48` causal pass-depth sweep across `loop_num_passes ∈ {1, 2, 3, 5}` (`24, 36, 48, 72` verified effective layers; `192` evaluations) plus `blank` and `mismatched` visual leakage controls (`96` evaluations) with `image_id` cluster bootstrap CIs and multi-panel Pareto analysis (`reports/phase3_smoke48.md`).

> Mock outputs under `artifacts/mock/` are synthetic test fixtures, not model measurements. Real GPU evaluation summaries are recorded in `reports/phase2_g4_probe.md` and `reports/phase3_smoke48.md` (with full raw JSONL traces and logs archived to `mydrive:inkling_loop_bench/phase3` via `rclone`).

## Setup

~~~bash
cd /d/Projects/inkling-loop-bench
python -m pip install -e ".[dev]"
~~~

## Run local mock pipeline & unit tests

~~~bash
python scripts/run_mock.py --config configs/mock.yaml
PYTHONPATH=src python -m pytest -q
~~~

## Run Phase 2 & Phase 3 GPU Suites (Colab G4 Blackwell)

~~~bash
python scripts/run_phase2_colab_suite.py --config configs/smoke.yaml --out artifacts/phase2
python scripts/run_phase3_smoke48.py --config configs/smoke.yaml --out artifacts/phase3
~~~

## Locked Configuration

- **Checkpoint**: `thinkingmachines/TML-d24-varloop-research` @ `c90921bd5f78df6c8168173b9f8a3a850320c594`
- **Backend**: Official HuggingFace `Transformers` (`Inkling2LoopForConditionalGeneration`, zero-reload in-place `loop_num_passes` switching with forward-hook layer count verification)
- **Pass Depths**: `1, 2, 3, 5` (`Inkling-2-Mini` kept strictly separate as external 3-pass reference)
- **Decoding Policy**: `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.0`, `max_new_tokens=2048`, `reasoning_effort=medium`, `concurrency=1`
- **Leakage Controls**: `image_mode` in `("correct", "blank", "mismatched")`
