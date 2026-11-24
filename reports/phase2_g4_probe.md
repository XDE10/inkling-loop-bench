# Phase 2 Execution & Sign-off Report — Colab G4 Blackwell Probe

Date: 2026-11-24  
Branch: `work/phase2-gpu-probe`

---

## 1. Environment & Checkpoint Fingerprint

- **Primary Checkpoint**: `thinkingmachines/TML-d24-varloop-research`
- **Pinned HF Revision SHA**: `c90921bd5f78df6c8168173b9f8a3a850320c594`
- **Colab Runtime**: Isolated G4 session (`loopbench-p2`, stopped immediately upon completion; `2.51` compute units consumed)
- **GPU**: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`97,887 MiB` total VRAM, Compute Capability `12.0`)
- **Driver / CUDA**: NVIDIA Driver `580.82.07`, CUDA `13.0`
- **Software Stack**:
  - OS: `Linux-6.6.159+-x86_64-with-glibc2.39`
  - Python: `3.13.16`
  - PyTorch: `2.14.1+cu130`
  - Transformers: `5.24.0` (`AutoProcessor` + `Inkling2LoopForConditionalGeneration`)
  - Accelerate: `1.16.0`, Kernels: `0.19.0`

---

## 2. Serving Backend Feasibility Gate (Time-Boxed)

Per `docs/FINAL_PLAN.md`, `vLLM` and `SGLang` were evaluated on the Colab G4 Blackwell runtime before locking the Phase 2/3 backend:

| Candidate | PyPI Latest on Colab | Feasibility Status | Technical Finding |
| :--- | :---: | :---: | :--- |
| **vLLM** | `0.34.0` | Not usable on stock PyPI | Official Inkling-2-Mini recipe requires `vllm >= 0.35.0+` nightly (`vllm/vllm-openai:nightly` image or unmerged nightly wheel); Docker daemon is unavailable in Colab VM. |
| **SGLang** | `0.5.23` | Not usable on Colab G4 | Official cookbook specifies unmerged PR (`sgl-project/sglang` PR `#51926`) / Docker image (`lmsysorg/sglang:inkling-2-mini`) targeting `SM103` (`fa4`). |
| **Transformers** | `5.24.0` | **Selected & Verified** | Native `Inkling2LoopForConditionalGeneration` & `Inkling2Processor` support verified end-to-end with deterministic greedy decoding and exact forward-hook layer validation. |

**Gate Decision**: Lock the official **HuggingFace `Transformers` reference backend** for Phase 2/3 single-stream accuracy, observable TTFT/TPOT/E2E latency, and peak VRAM reporting.

> **Runtime Note**: In `transformers 5.24.0`, `quant_method: "modelopt"` (`NVFP4`) is skipped with a warning and weights run cleanly in native `BF16/FP32` within `~50.7 GB` VRAM (`51.8%` of the 96 GB RTX PRO 6000 Blackwell), using the reference PyTorch `causal_conv1d` path consistently across all pass depths.

---

## 3. Single-Image Transformers Probe (`loop=1` vs `loop=3`)

Tested on 2 images × 2 tasks = **4 questions** (`chart-000` direct & multi-step, `geometry-000` direct & multi-step) at `reasoning_effort="medium"`, `temperature=0.0`, `max_new_tokens=2048`.

### Critical Implementation Validation
1. **Config Shadowing Guard**: `config.json` contains `"loop_config": {"num_passes": 3}`, which overrides `loop_num_passes` during `Inkling2LoopTextConfig.__post_init__`. `TransformersBackend.apply_loop_num_passes()` updates both `model.config.text_config.loop_num_passes` and `model.model.language_model.config.loop_num_passes` in-place (zero weight reload across pass depths).
2. **Causal Forward-Hook Proof**: Forward pre-hooks attached to `model.model.language_model.layers` verified **exact `24` layer evaluations/step at `loop=1`** and **exact `48` layer evaluations/step at `loop=3`**.
3. **Processor & Token Telemetry**: Every `768×768` RGB image produces `image_grid_thw = [1, 20, 20]` (**`400` visual tokens**). Model outputs natively separate `<|content_thinking|>...<|end_message|>` and `<|content_text|>...<|end_message|>`.

### Probe Quantitative Comparison

| Metric | `loop = 1` (24 effective layers) | `loop = 3` (48 effective layers) | Scaling Ratio (`3 / 1`) |
| :--- | :---: | :---: | :---: |
| **Effective Layers Verified** | `[24, 24, 24, 24]` | `[48, 48, 48, 48]` | **`2.00×`** |
| **Accuracy (`n=4`)** | `4/4 (100.0%)` | `4/4 (100.0%)` | `1.00×` |
| **Observable TTFT P50** | `113.88 ms` | `230.06 ms` | **`2.02×`** |
| **Observable TTFT P90** | `115.43 ms` | `242.19 ms` | **`2.10×`** |
| **Decode TPOT (mean)** | `35.08 ms/tok` | `84.98 ms/tok` | **`2.42×`** |
| **E2E Latency P50** | `5,042.47 ms` | `11,778.35 ms` | **`2.34×`** |
| **Peak VRAM (max)** | `50,629.97 MB` | `50,696.49 MB` | **`+66.52 MB (+0.13%)`** |
| **Mean Reasoning Tokens** | `111.5` | `121.8` | `1.09×` |
| **Mean Answer Tokens** | `18.0` | `18.5` | `1.03×` |

---

## 4. 12-Question Decoding-Policy Diagnostic (`loop=3`, `effort=medium`)

Evaluated across 6 diverse templates × 2 tasks = **12 questions** (`chart-000`, `chart-002`, `chart-005`, `geometry-001`, `geometry-003`, `geometry-005`):

| Policy | Decoding Settings | Raw Grader Acc | Post-Aside-Fix Acc | Truncated (`length`) | Mean / Max Reasoning Toks | TTFT P50 | E2E P50 | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`greedy_512`** | `temp=0.0, top_p=1.0, max=512` | `10/12 (83.3%)` | `10/12 (83.3%)` | **`2/12 (16.7%)`** | `196.4 / 510` | `227.15 ms` | `13,124.15 ms` | `50,730.9 MB` |
| **`greedy_2048`** | `temp=0.0, top_p=1.0, max=2048` | `10/12 (83.3%)` | **`12/12 (100.0%)`** | **`0/12 (0.0%)`** | `216.7 / 635` | `226.60 ms` | `13,100.61 ms` | `50,750.7 MB` |
| **`sampled_t1_2048`** | `temp=1.0, top_p=0.95, max=2048` | `10/12 (83.3%)` | **`12/12 (100.0%)`** | **`0/12 (0.0%)`** | `237.2 / 866` | `226.54 ms` | `15,999.99 ms` | `50,765.0 MB` |

### Key Diagnostic Findings & Locked Fixes for Phase 3

1. **Token Budget Gate (`512` → `2048` Locked)**:
   - With `max_new_tokens=512`, `2/12` questions (`16.7%`) truncated inside `<|content_thinking|>` before emitting `<|content_text|>`. Raising `max_new_tokens` to **`2048`** reduced truncation to **`0%`** (`max_reasoning_tokens = 635` under greedy, `866` under `temp=1.0`).
2. **Greedy vs Sampling Gate (`temperature=0.0` Locked)**:
   - Deterministic greedy decoding (`temperature=0.0`) exhibited **zero repetition loops**, identical mathematical accuracy to `temperature=1.0`, **18.1% lower P50 E2E latency** (`13.10s` vs `16.00s`), and zero sampling variance across pass depths.
3. **Grader Aside-Contamination Fix (`grading.py`)**:
   - Inspection of the 2 apparent `wrong_answer` rows under `greedy_2048` (`geometry-003-direct` and `geometry-005-direct`) revealed that the model solved **all 12/12 questions correctly** (`48°` and `80°`), but appended helpful disambiguation asides about the red `?` from the paired multi-step prompt:
     - `The marked interior angle at vertex **C** is **48°**. (If you meant the exterior angle at **B**, it would be **98°**.)`
     - `The marked angle **∠ADB** is **80°** as shown in the figure.\n\nNote: the red unknown angle at **B** would be **35°**.`
   - `extract_number()` in `src/loopbench/grading.py` has been updated to strip trailing parenthetical/Note asides before fallback numeric extraction while keeping the Phase 1 `questions.json` schema 100% frozen. Re-scoring `diag12_greedy_2048.jsonl` with the updated grader yields **`12/12 (100.0%)`**.

---

## 5. Phase 2 Sign-off

All Phase 2 exit criteria in `docs/FINAL_PLAN.md` are satisfied:
- Exact HF revision SHA (`c90921bd5f78df6c8168173b9f8a3a850320c594`) pinned in `configs/smoke.yaml`.
- Single-image `loop=1` vs `loop=3` probe verified config, processor (`[1, 20, 20]`, `400` visual tokens), forward layer counts (`24` vs `48`), latency (`TTFT` `2.02×`, `E2E` `2.34×`), and shared-weight VRAM stability (`+66.5 MB`).
- Serving backend feasibility gate completed and documented.
- 12-question decoding-policy diagnostic completed, locking `temperature=0.0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=2048` for Phase 3 (`smoke-48` across `loop=1, 2, 3, 5` + blank/mismatched leakage checks).
