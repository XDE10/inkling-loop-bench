# LoopBench

**Measuring Accuracy–Latency–Memory Trade-offs of Latent Looping and Explicit Reasoning in Multimodal Inference.**

Phase 1 is CPU-only and mock-first (signed off). It does not download model weights, start a GPU, use a local proxy, or reuse Geometry3K.

> Mock outputs are synthetic test fixtures, not model measurements.

## Setup

~~~bash
cd /d/Projects/inkling-loop-bench
python -m pip install -e ".[dev]"
~~~

## Run the full mock pipeline & tests

~~~bash
python scripts/run_mock.py --config configs/mock.yaml
PYTHONPATH=src python -m pytest -q
~~~

Outputs are written under `artifacts/mock/`: generated 768×768 RGB images and portable `questions.json` (`schema_version: 2`), append-only `results.jsonl`, `summary.json`, `summary.csv`, `report.md`, and `pareto_mock.png`. All mock reports are stamped `MOCK / TEST DATA — NOT REAL MODEL PERFORMANCE`.

## Dataset (`smoke-48` & `synth-240` ready)

The deterministic `smoke-48` set contains 24 images (12 charts across 6 distinct chart sub-templates, 12 geometry diagrams across 6 Euclidean-verified geometry sub-templates) and 48 questions (`direct` + `multi_step` per image).
- Grid-read charts align values to exact multiples of 5 with 10-unit major solid gridlines and 5-unit minor dashed gridlines; multi-series/stacked charts include explicit series labels and legends.
- Geometry diagrams compute exact Euclidean vertex coordinates (verified by vector dot-product invariants in `validate_dataset`), interior angle bisector arcs, right-angle squares, parallel markers, and congruence tick marks with zero prompt numeric leakage.
- Image dimensions are 768×768 true 3-channel RGB; pixel SHA256 hashes are recorded independently of PNG metadata, and `image_path` entries are stored as portable POSIX relative paths (`images/<id>.png`).

## Result schema & error taxonomy

Each JSONL row carries `config_id + question_id` for idempotent resumability and automatic trailing partial-record crash recovery. Every prediction is classified into five mutually exclusive outcomes (`correct`, `wrong_answer`, `parse_failure`, `truncated`, `backend_error`) plus `image_mode` (`correct`, `blank`, `mismatched`) for visual leakage checks. Forward-compatible GPU fields (`reasoning_tokens`, `answer_tokens`, `visual_tokens`, `image_grid_thw`, `ttft_ms`, `tpot_ms`, `e2e_ms`, `peak_vram_mb`, `speculative_decoding`, `loop_share_cache`) are reserved as nullable.

## Phase 2 — GPU smoke (next phase)

Before using `configs/smoke.yaml`, pin an exact HuggingFace commit SHA in `model_revision` (unpinned revisions are rejected by `validate_runtime_ready`) and record the Colab GPU, driver, CUDA, PyTorch, Transformers, and processor versions. The initial GPU probe loads `thinkingmachines/TML-d24-varloop-research` and compares single-image execution at `loop=1` and `loop=3`.

## Research boundaries

- The causal pass-depth comparison uses one fixed `TML-d24-varloop-research` revision across passes `1, 2, 3, 5`.
- `Inkling-2-Mini` is a separate fixed-3-pass external reference, never mixed into the pass-depth curve.
- 48 questions are smoke/engineering validation only, not statistical evidence.
- Main synthetic study target: 120 images × 2 questions = 240 questions (`--n-per-type 60`); expand to 320 only under a predeclared uncertainty rule.
- Real-world holdout (60–100 questions) is a later, separately reported phase.
- No online router or dashboard in the core deliverable.
