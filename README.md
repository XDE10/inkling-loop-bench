# LoopBench

**Measuring Accuracy–Latency–Memory Trade-offs of Latent Looping and Explicit Reasoning in Multimodal Inference.**

Phase 1 is CPU-only and mock-first. It does not download model weights, start a GPU, use a local proxy, reuse Geometry3K, or push to `main`.

> Mock outputs are synthetic test fixtures, not model measurements.

## Setup (PowerShell)

~~~powershell
cd D:\Projects\inkling-loop-bench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
~~~

## Run the full mock pipeline

~~~powershell
python scripts\run_mock.py --config configs\mock.yaml
pytest -q
~~~

Outputs are written under `artifacts/mock/`: generated images/questions, append-only JSONL, JSON summary, Markdown report and static plot. All mock reports are stamped `MOCK / TEST DATA — NOT REAL MODEL PERFORMANCE`.

## Dataset

The deterministic smoke set contains 24 images (12 charts, 12 geometry diagrams) and 48 questions (one direct and one multi-step question per image). Answers are recomputed from generator metadata. Image dimensions are 768×768 RGB; pixel hashes are recorded independently of PNG metadata.

## Result schema

Each JSONL row carries `config_id + question_id` for resumability. Fields not available in mock mode are retained as null for forward compatibility, including reasoning/answer/visual tokens, image_grid_thw, TTFT, TPOT, E2E, peak VRAM, speculative decoding and loop cache mode.

## Phase 2 — GPU smoke (not implemented yet)

Before using `configs/smoke.yaml`, pin an exact model revision and record the Colab GPU, driver, CUDA, PyTorch, Transformers and processor versions. The initial GPU probe should load the same research checkpoint and compare one image at loop=1 and loop=3. Do not treat this placeholder config as a runnable GPU command.

## Research boundaries

- The causal pass-depth comparison uses one fixed `TML-d24-varloop-research` revision.
- `Inkling-2-Mini` is a separate fixed-3-pass external reference, never mixed into the pass-depth curve.
- 48 questions are smoke/engineering validation only, not statistical evidence.
- Main synthetic study target: 120 images × 2 questions = 240 questions; expand to 320 only under a predeclared uncertainty rule.
- Real-world holdout (60–100 questions) is a later, separately reported phase.
- No online router or dashboard in the core deliverable.
