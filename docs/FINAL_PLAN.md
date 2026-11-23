# Final research and implementation plan

This plan is the closed decision from AI Debate Room room-001 (messages 001–005).

## Research question

Measure the accuracy–latency–memory trade-off of inference-time loop scaling and explicit reasoning effort for one fixed variable-loop multimodal checkpoint. Do not assume more passes are faster or more accurate.

## Locked choices

- Primary checkpoint: `thinkingmachines/TML-d24-varloop-research`, exact HF revision SHA to be pinned before GPU use.
- Passes: 1, 2, 3, 5; same checkpoint and all other settings fixed for the causal pass-depth comparison.
- `Inkling-2-Mini`: separate fixed-3-pass external reference only; never mix into the pass-depth curve.
- First GPU correctness path: official Transformers reference path.
- Backend gate: time-box vLLM/SGLang feasibility on Colab G4. If usable and correctness aligned, use it for serving latency; otherwise report Transformers reference-backend latency honestly.
- Decoding parameters are explicit: temperature, top_p, repetition_penalty, max_new_tokens, stop.
- Visual inputs: fixed rendering settings; record processor-derived image_grid_thw/visual tokens where available.
- Report only observable TTFT boundaries. Do not infer TTFT internals from pass count.
- Phase 1 is CPU/mock only: no large model download, GPU launch, local proxy, Geometry3K reuse, or push to main.

## Phases

1. CPU engineering baseline: deterministic smoke-48, validation, mock end-to-end, tests, frozen data/result schemas.
2. Colab G4 single-image Transformers probe: loop 1 vs 3, validate config/processor/output/memory; then serving backend feasibility gate and 12-question decoding-policy diagnostic.
3. Real smoke-48: loop 1/2/3/5, fixed effort and decoding, single concurrency; include correct/blank/mismatched-image leakage checks.
4. Main synthetic study: synth-240 (120 images × 2 questions); extend to synth-320 only under a predeclared uncertainty rule. Use image_id cluster bootstrap.
5. Small effort matrix and one-source real-world holdout (60–100 questions), separately reported.
6. Optional systems ablations only if main signal is stable: DSpark, KV cache, concurrency/throughput, offline Oracle. No required router or dashboard.

## Phase 1 acceptance

- 24 images and 48 questions, 12 chart + 12 geometry, each image paired with direct + multi-step.
- Deterministic seed, pixel-matrix SHA256, answer recomputation and image validation.
- Mock pipeline: generate → validate → run → grade → metrics → report.
- Resume by config_id + question_id; recover an incomplete trailing JSONL record without losing completed rows.
- Separate parse failures, truncation, wrong answers and backend errors.
- Explicit config schema; future result fields reserved as nullable.
- JSONL, CSV/JSON summary, static plot and Markdown report; mock artifacts are visibly labeled as test data.
