# Phase 3 Execution & Sign-off Report — Real `smoke-48` Pass-Depth Scaling & Visual Leakage Gate

Date: 2026-11-24  
Branch: `work/phase3-smoke48`

---

## 1. Experimental Setup & Environment Fingerprint

- **Primary Checkpoint**: `thinkingmachines/TML-d24-varloop-research`
- **Pinned HF Revision SHA**: `c90921bd5f78df6c8168173b9f8a3a850320c594`
- **Backend**: Official HuggingFace `Transformers` reference backend (`AutoProcessor` + `Inkling2LoopForConditionalGeneration`), single-stream (`concurrency = 1`)
- **Hardware (Colab G4)**: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`97,887 MiB` total VRAM, Compute Capability `12.0`, Driver `580.82.07`, CUDA `13.0`)
- **Software Stack**: Linux `6.6.159+`, Python `3.13.16`, `torch 2.14.1+cu130`, `transformers 5.24.0`, `accelerate 1.16.0`, `kernels 0.19.0`
- **Locked Decoding & Visual Inputs**:
  - `reasoning_effort = "medium"`, `temperature = 0.0`, `top_p = 1.0`, `repetition_penalty = 1.0`, `max_new_tokens = 2048`, `stop = []`, `seed = 20261123`
  - `24` deterministic `768×768` RGB images (`12` chart + `12` geometry across `12` sub-templates) × `2` paired tasks (`direct`, `multi_step`) = **`48` questions** (`image_grid_thw = [1, 20, 20]`, **`400` visual tokens** per image)
- **Zero-Reload In-Place Pass Switching & Causal Hook Proof**:
  - Within each GPU worker run, `TransformersBackend` loaded the checkpoint once (`loads = 1`) and switched `loop_num_passes ∈ {1, 2, 3, 5}` in-place via `apply_loop_num_passes()`.
  - Forward pre-hooks attached to `model.model.language_model.layers` verified exact prefill layer invocations on **every single question (`48/48` per stage)**: **`24`, `36`, `48`, and `72` effective layers** for `loop = 1, 2, 3, 5`.
- **Artifact & Cloud Storage Layout**:
  - Full raw `.jsonl` traces (`288` total model evaluations), worker logs, Colab session logs, and archive bundles are synced via `rclone` to `mydrive:inkling_loop_bench/phase3`, while compact summary JSON/CSV/PNG artifacts reside locally under `artifacts/phase3/report/`.

---

## 2. Causal Pass-Depth Scaling Results (`smoke-48`, `image_mode="correct"`)

All settings except `loop_num_passes` were held strictly fixed across `4 × 48 = 192` evaluations:

| `loop_num_passes` | Effective Layers Verified | Accuracy (`n=48`) | `image_id` Cluster 95% CI | 5-Way Outcomes (`C / W / PF / T / BE`) | TTFT P50 (`Ratio`) | Decode TPOT P50 (`Ratio`) | E2E P50 (`Ratio`) | E2E P90 | Peak VRAM Max (`Δ vs loop=1`) | Mean Reasoning / Answer Toks |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1`** | `24` (`1.00×`) | **`100.0%` (`48/48`)** | `[100.0%, 100.0%]` | `48 / 0 / 0 / 0 / 0` | `115.1 ms` (`1.00×`) | `34.02 ms` (`1.00×`) | `5,322.7 ms` (`1.00×`) | `25,013.0 ms` | `50,672.1 MB` (`+0.0 MB`) | `257.5 / 32.4` |
| **`2`** | `36` (`1.50×`) | **`95.8%` (`46/48`)** | `[89.6%, 100.0%]` | `46 / 2 / 0 / 0 / 0` | `170.7 ms` (`1.48×`) | `57.49 ms` (`1.69×`) | `8,781.3 ms` (`1.65×`) | `43,637.4 ms` | `50,728.8 MB` (`+56.7 MB`) | `251.6 / 31.1` |
| **`3`** | `48` (`2.00×`) | **`97.9%` (`47/48`)** | `[93.8%, 100.0%]` | `47 / 1 / 0 / 0 / 0` | `228.1 ms` (`1.98×`) | `82.04 ms` (`2.41×`) | `11,900.8 ms` (`2.24×`) | `55,940.3 ms` | `50,786.5 MB` (`+114.4 MB`) | `238.1 / 32.4` |
| **`5`** | `72` (`3.00×`) | **`95.8%` (`46/48`)** | `[89.6%, 100.0%]` | `46 / 2 / 0 / 0 / 0` | `340.5 ms` (`2.96×`) | `131.93 ms` (`3.88×`) | `20,635.2 ms` (`3.88×`) | `88,773.5 ms` | `50,899.1 MB` (`+227.0 MB`) | `238.4 / 33.7` |

*(Legend for 5-Way Outcomes: `C` = Correct, `W` = Wrong Answer, `PF` = Parse Failure, `T` = Truncated `finish_reason=length`, `BE` = Backend Error.)*

### Subgroup Accuracy Breakdown (`Modality × Task Type`)

| Subgroup (`n=12` each) | `loop = 1` (`24L`) | `loop = 2` (`36L`) | `loop = 3` (`48L`) | `loop = 5` (`72L`) |
| :--- | :---: | :---: | :---: | :---: |
| **`chart / direct`** | `100.0%` (`12/12`) | `100.0%` (`12/12`) | `100.0%` (`12/12`) | `100.0%` (`12/12`) |
| **`chart / multi_step`** | `100.0%` (`12/12`) | `100.0%` (`12/12`) | `100.0%` (`12/12`) | `100.0%` (`12/12`) |
| **`geometry / direct`** | `100.0%` (`12/12`) | `83.3%` (`10/12`) | `91.7%` (`11/12`) | `83.3%` (`10/12`) |
| **`geometry / multi_step`** | `100.0%` (`12/12`) | `100.0%` (`12/12`) | `100.0%` (`12/12`) | `100.0%` (`12/12`) |

### Paired `image_id` Cluster Bootstrap Accuracy Differences (`2,000` resamples, `24` image clusters)

| Comparison | Point Δ Accuracy | Paired Cluster Bootstrap 95% CI | Statistical Interpretation |
| :--- | :---: | :---: | :--- |
| `loop=2 − loop=1` | `-4.17%` | `[-10.42%, 0.00%]` | Indistinguishable from ceiling (2 `geometry/direct` over-thinking items) |
| `loop=3 − loop=1` | `-2.08%` | `[-6.25%, 0.00%]` | Indistinguishable from ceiling (1 `geometry/direct` over-thinking item) |
| `loop=5 − loop=1` | `-4.17%` | `[-10.42%, 0.00%]` | Indistinguishable from ceiling (2 `geometry/direct` over-thinking items) |
| `loop=3 − loop=2` | `+2.08%` | `[-4.17%, +8.33%]` | Zero-spanning CI |
| `loop=5 − loop=3` | `-2.08%` | `[-6.25%, 0.00%]` | Zero-spanning CI |

---

## 3. Visual Leakage Control Gate (`loop = 3`, `n = 48` per condition)

To verify that `smoke-48` strictly requires reading the matched visual input rather than exploiting prompt priors or numeric heuristics, all `48` questions were evaluated at `loop_num_passes = 3` under three `image_mode` settings:

| `image_mode` | Accuracy (`n=48`) | Cluster 95% CI | Correct | Wrong Answer | Parse Failure / Refusal | Truncated | Backend Error | Accuracy Drop vs `correct` | E2E P50 | Mean Reasoning Toks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`correct`** | **`97.9%` (`47/48`)** | `[93.8%, 100.0%]` | `47` | `1` | `0` | `0` | `0` | — | `11,900.8 ms` | `238.1` |
| **`blank`** | **`0.0%` (`0/48`)** | `[0.0%, 0.0%]` | **`0`** | `24` | `24` | `0` | `0` | **`-97.92 pp`** | `27,447.7 ms` | `303.8` |
| **`mismatched`** | **`2.1%` (`1/48`)** | `[0.0%, 6.2%]` | **`1`** | `24` | `23` | `0` | `0` | **`-95.83 pp`** | `32,535.6 ms` | `399.9` |

- **`blank` (`768×768` pure white RGB image)**: Accuracy collapsed to **`0.0%` (`0/48`)**. On chart prompts the model reported `0` (`wrong_answer`), and on geometry prompts it predominantly refused to fabricate unseen angles (`parse_failure`) or stated generic `90°/180°` geometric constants (`wrong_answer`).
- **`mismatched` (cross-modality chart ↔ geometry image swap)**: Accuracy collapsed to **`2.1%` (`1/48`)** — `47/48` items either refused due to modality mismatch or extracted values from the wrong image; the single `correct` row (`geometry-004-direct`, `expected = 50.0`) was an isolated `50°` chance coincidence under the swapped `chart-004` image (`95% CI [0.0%, 6.2%]`).

---

## 4. Key Empirical Findings & Phase 4 Readiness

1. **Latency vs Effective Depth Follows Distinct Prefill and Decode Regimes**:
   - **Observable TTFT P50** (`115.1 → 170.7 → 228.1 → 340.5 ms`, i.e. `1.00× → 1.48× → 1.98× → 2.96×`) tracks total effective layer count (`24 → 36 → 48 → 72`, i.e. `1.00× → 1.50× → 2.00× → 3.00×`) almost 1-for-1 because vision encoding overhead is small relative to full-sequence decoder prefill.
   - **Decode TPOT P50** (`34.02 → 57.49 → 82.04 → 131.93 ms/tok`, i.e. `1.00× → 1.69× → 2.41× → 3.88×`) scales steeper than total layer ratio (`3.00×`) and approaches the looped-core ratio (`1 → 2 → 3 → 5`) because the repeated core block (`layers 6..17`) dominates autoregressive memory-bandwidth/state-update cost per generated token under unshared per-pass KV/conv state (`loop_share_cache = False`).
2. **Near-Constant VRAM Scaling Confirming Weight-Tied Latent Looping**:
   - Scaling from `24` effective layers (`loop=1`) to `72` effective layers (`loop=5`, `3×` depth) increased peak VRAM by only **`+227.0 MB` (`+0.45%`, `~56.7 MB/pass`)** on top of the `50.67 GB` model footprint, empirically confirming weight sharing across passes with linear growth limited strictly to per-pass KV/conv cache tensors.
3. **Accuracy Ceiling & Over-Thinking Mechanism on `smoke-48`**:
   - With `reasoning_effort = "medium"` (~`240–258` explicit CoT tokens), even `loop = 1` achieved **`100.0%` (`48/48`)** across all chart and multi-step geometry tasks (`0%` truncation under `max_new_tokens = 2048`).
   - All `5` errors out of `192` evaluations (`2` at `loop=2`, `1` at `loop=3`, `2` at `loop=5`) occurred on `geometry/direct` items where deeper latent passes fixated on the red `?` arc drawn for the paired `multi_step` task and solved the harder unknown angle instead of reporting the literally labeled angle (e.g. `geometry-004-direct`: model solved `∠ACB = 69°` as its primary answer and noted `∠PCA = 50°` at the end). More passes are therefore **neither faster nor automatically more accurate** when `effort="medium"` already saturates task difficulty (`docs/FINAL_PLAN.md` core hypothesis test).
4. **Sign-off for Phase 4 (`synth-240`)**:
   - All Phase 3 exit criteria (`loop ∈ {1, 2, 3, 5}` causal pass curve, zero-reload verification, `0%` truncation, `correct / blank / mismatched` leakage gate, cluster bootstrap CIs, and Pareto visualization) are complete and signed off. The pipeline is ready to scale to Phase 4 (`synth-240`, `120` images × `2` questions) and Phase 5 (`none / low / medium / high` reasoning-effort matrix, where `effort="none"` and `effort="low"` will test whether deeper latent passes substitute for suppressed explicit CoT).
