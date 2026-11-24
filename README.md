# LoopBench

**Measuring Accuracy–Latency–Memory Trade-offs of Latent Looping and Explicit Reasoning in Multimodal Inference.**

- **Phase 1 (Signed off & merged)**: Deterministic CPU/mock pipeline (`schema_version: 2`, 12 chart/geometry sub-templates, crash-resilient JSONL resumability, 5-way error taxonomy, cluster bootstrap CI).
- **Phase 2 (Signed off on Colab G4 Blackwell)**: Pinned `thinkingmachines/TML-d24-varloop-research` (`c90921bd5f78df6c8168173b9f8a3a850320c594`), verified `TransformersBackend` on `NVIDIA RTX PRO 6000 Blackwell Server Edition` (96 GB), causally verified `loop=1` (24 layers) vs `loop=3` (48 layers), completed the Serving Backend Gate and 12-question decoding-policy diagnostic (`reports/phase2_g4_probe.md`).

> Mock outputs under `artifacts/mock/` are synthetic test fixtures, not model measurements. Real GPU probe outputs are recorded under `artifacts/phase2/` and summarized in `reports/phase2_g4_probe.md`.

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

## Run Phase 2 GPU Probe & Diagnostic (Colab G4 Blackwell)

~~~bash
python scripts/run_phase2_colab_suite.py --config configs/smoke.yaml --out artifacts/phase2
~~~

## Locked Configuration for Phase 3 (`smoke-48` Pass-Depth Curve)

- **Checkpoint**: `thinkingmachines/TML-d24-varloop-research` @ `c90921bd5f78df6c8168173b9f8a3a850320c594`
- **Backend**: Official HuggingFace `Transformers` (`Inkling2LoopForConditionalGeneration`, in-place `loop_num_passes` switching with forward-hook layer count verification)
- **Pass Depths**: `1, 2, 3, 5` (`Inkling-2-Mini` kept strictly separate as external 3-pass reference)
- **Decoding Policy**: `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.0`, `max_new_tokens=2048`, `reasoning_effort=medium`
- **Leakage Controls**: `image_mode` in `("correct", "blank", "mismatched")`
