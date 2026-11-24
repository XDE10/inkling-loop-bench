# Phase 5 Execution & Sign-off Report — Reasoning-Effort Matrix (`effort48`) & One-Source Real-World Holdout (`ChartQA-80`)

Date: 2026-11-25  
Branch: `work/phase5-effort-holdout`

---

## 1. Experimental Setup & Strict Separation Guarantee

Per `docs/FINAL_PLAN.md` (Phase 5), the two studies below were executed and reported **strictly separately** on `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`97,887 MiB`, CUDA `13.0`, `torch 2.14.1+cu130`, `transformers 5.24.0`, `accelerate 1.16.0`, `kernels 0.19.0`) using the locked primary checkpoint **`thinkingmachines/TML-d24-varloop-research`** @ HF revision SHA `c90921bd5f78df6c8168173b9f8a3a850320c594` (`temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.0`, `max_new_tokens=2048`, `concurrency=1`, forward-hook verified effective layers `24 / 36 / 48 / 72` on `100%` of evaluations, zero-reload `loads = 1, hits = 656` on the main session):

1. **Study 5A — Reasoning Effort × Latent Pass-Depth Matrix (`effort48`)**:
   - **Dataset**: Deterministic balanced `48`-question synthetic subset (`seed=20261124`, `profile="synth"`, `seed_mode="per_item"`, `n_per_type=12`: `24` images × `2` paired tasks, covering **1 image per all 24 `v1 standard` + `v2 hard` chart/geometry sub-templates**, `768×768` RGB, `400` visual tokens/image).
   - **Grid**: `reasoning_effort ∈ {none, low, medium, high}` × `loop_num_passes ∈ {1, 3, 5}` = **`12` cells × `48` questions = `576` GPU evaluations**.
2. **Study 5B — One-Source Real-World Holdout (`ChartQA-80`)**:
   - **Dataset**: Curated deterministically (`seed=20261125`) from `HuggingFaceM4/ChartQA` (`split="test"`, pinned revision SHA `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`), sampling `40` distinct real-world native-resolution chart clusters (`200–858 × 306–836` px, processor-derived `50–336` visual tokens, median `280`) × `2` human-authored finite numeric questions = **`80` questions**.
   - **Sweep**: `loop_num_passes ∈ {1, 2, 3, 5}` at fixed `reasoning_effort="medium"` = **`4` depths × `80` questions = `320` GPU evaluations** (`896` total Phase 5 GPU evaluations).
3. **Cloud-First Storage**: All `16` raw JSONL traces (`896` evaluations), worker logs, dataset archive (`dataset_chartqa80.tar.gz`), and bundles are archived to `mydrive:inkling_loop_bench/phase5` via `rclone`; compact summary JSON/CSV/PNG artifacts reside locally under `artifacts/phase5/report/`.

---

## 2. Study 5A: Latent Loop Depth × Explicit Reasoning Effort (`effort48`)

### Full 4 × 3 Empirical Matrix (`n = 48` questions, `24` `image_id` clusters per cell)

| `reasoning_effort` | `loop_num_passes` | Effective Layers | Accuracy (`n=48`) | `image_id` Cluster 95% CI | Standard (`v1`) | Hard (`v2`) | Direct | Multi-Step | Outcomes (`C / W / T`) | Mean CoT Toks | TTFT P50 (ms) | TPOT P50 (ms) | E2E P50 (s) | E2E Mean (s) | Peak VRAM Max (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`none`** | `1` | `24` | `83.3%` (`40/48`) | `[70.8%, 93.8%]` | `87.5%` | `79.2%` | `79.2%` | `87.5%` | `40 / 4 / 4` | `539.5` | `114.1` | `33.80` | `9.57` | `19.82` | `50,676.7` |
| **`none`** | `3` | `48` | `89.6%` (`43/48`) | `[77.1%, 97.9%]` | `95.8%` | `83.3%` | `83.3%` | `95.8%` | `43 / 3 / 2` | `460.0` | `227.8` | `82.27` | `21.14` | `42.20` | `50,800.4` |
| **`none`** | `5` | `72` | `87.5%` (`42/48`) | `[75.0%, 95.8%]` | `95.8%` | `79.2%` | `79.2%` | `95.8%` | `42 / 3 / 3` | `519.1` | `340.4` | `131.13` | `30.43` | `74.56` | `50,923.6` |
| **`low`** | `1` | `24` | `89.6%` (`43/48`) | `[81.2%, 97.9%]` | `87.5%` | `91.7%` | `83.3%` | `95.8%` | `43 / 4 / 1` | `422.2` | `112.7` | `34.34` | **`8.09`** | `16.51` | `50,676.7` |
| **`low`** | **`3`** | **`48`** | **`97.9%` (`47/48`)** | **`[93.8%, 100.0%]`** | **`100.0%`** | **`95.8%`** | **`100.0%`** | **`95.8%`** | **`47 / 0 / 1`** | **`348.9`** | `226.2` | `83.29` | **`15.50`** | `33.16` | `50,802.3` |
| **`low`** | `5` | `72` | `95.8%` (`46/48`) | `[89.6%, 100.0%]` | `95.8%` | `95.8%` | `95.8%` | `95.8%` | `46 / 2 / 0` | **`335.5`** | `341.8` | `132.64` | `23.71` | `51.48` | `50,908.1` |
| **`medium`** | `1` | `24` | `91.7%` (`44/48`) | `[81.2%, 100.0%]` | `95.8%` | `87.5%` | `87.5%` | `95.8%` | `44 / 2 / 2` | `486.6` | `116.3` | `34.62` | `10.52` | `18.76` | `50,690.0` |
| **`medium`** | `3` | `48` | `93.8%` (`45/48`) | `[83.3%, 100.0%]` | `95.8%` | `91.7%` | `91.7%` | `95.8%` | `45 / 2 / 1` | `483.2` | `229.6` | `83.53` | `20.27` | `45.03` | `50,802.3` |
| **`medium`** | `5` | `72` | `91.7%` (`44/48`) | `[81.2%, 100.0%]` | `100.0%` | `83.3%` | `91.7%` | `91.7%` | `44 / 2 / 2` | `512.1` | `346.0` | `131.22` | `30.87` | `73.83` | `50,935.6` |
| **`high`** | `1` | `24` | `87.5%` (`42/48`) | `[75.0%, 97.9%]` | `91.7%` | `83.3%` | `79.2%` | `95.8%` | `42 / 3 / 3` | `589.9` | `114.2` | `34.28` | `11.64` | `22.01` | `50,691.1` |
| **`high`** | `3` | `48` | `85.4%` (`41/48`) | `[72.9%, 95.8%]` | `95.8%` | `75.0%` | `75.0%` | `95.8%` | `41 / 3 / 4` | `567.6` | `227.0` | `82.93` | `22.70` | `50.98` | `50,816.8` |
| **`high`** | `5` | `72` | `91.7%` (`44/48`) | `[83.3%, 97.9%]` | `100.0%` | `83.3%` | `91.7%` | `91.7%` | `44 / 0 / 4` | `568.3` | `345.2` | `131.32` | `37.57` | `81.05` | `50,944.6` |

*(All `576` runs had `0` parse failures and `0` backend errors; `C / W / T` = Correct / Wrong Answer / Truncated at `max_new_tokens=2048`.)*

### Key Paired `image_id` Cluster Bootstrap Contrasts (`2,000` resamples, `24` clusters)

| Axis | Comparison | Point Δ Acc | 95% Cluster Bootstrap CI | Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Pass Depth @ `low`** | **`low_loop3 − low_loop1`** | **`+8.33%`** | **`[+2.08%, +16.67%]`** | **Statistically significant gain (`CI > 0`)**; eliminates all `4` wrong answers (`0` wrong) & trims CoT `-17.4%` |
| **Pass Depth @ `low`** | `low_loop5 − low_loop1` | `+6.25%` | `[0.00%, +12.50%]` | Zero truncations (`0/48`) and shortest CoT (`335.5` tok), but `1.53×` slower than `low_loop3` |
| **Pass Depth @ `none`** | `none_loop3 − none_loop1` | `+6.25%` | `[-4.17%, +18.75%]` | Latent looping compensates for unguided CoT (`+6.3 pp`, halving truncations `4 → 2`) |
| **Effort @ `loop=3`** | **`low_loop3 − none_loop3`** | **`+8.33%`** | **`[+2.08%, +16.67%]`** | **Statistically significant gain (`CI > 0`)** AND **`-26.7%` faster E2E P50 (`15.50s` vs `21.14s`)** |
| **Effort @ `loop=3`** | `high_loop3 − none_loop3` | `-4.17%` | `[-10.42%, 0.00%]` | Excessive explicit effort (`high`) inflates CoT (`567.6` tok) & truncations (`4/48`), hurting accuracy |
| **Effort @ `loop=5`** | **`low_loop5 − none_loop5`** | **`+8.33%`** | **`[+2.08%, +16.67%]`** | **Statistically significant gain (`CI > 0`)** with `-35.4%` fewer CoT tokens (`335.5` vs `519.1`) |

### Pareto Frontier & Offline Adaptive Oracle (`effort48`)

- **Non-Dominated Fixed-Policy Pareto Frontier** (both P50 and mean E2E): **`low_loop1` (`89.6%`, `8.09s` P50) → `medium_loop1` (`91.7%`, `10.52s` P50) → `low_loop3` (`97.9%`, `15.50s` P50)**. Every `none`, `high`, `medium_loop3/5`, and `low_loop5` configuration is strictly dominated.
- **Offline Router Oracle**: Achieves **`97.9%` (`47/48`)** at **`7.89s` P50 / `15.84s` mean E2E (`2.09×` mean speedup over best fixed policy `low_loop3` at `33.16s` mean!)** by routing **`46/48` (`95.8%`)** items to `loop=1` (`28` `low_loop1`, `14` `none_loop1`, `3` `medium_loop1`, `1` `high_loop1`) and escalating only `2/48` items to `loop=3` (`medium_loop3`, `high_loop3`).

---

## 3. Study 5B: One-Source Real-World Holdout (`ChartQA-80`, Strictly Separate)

Evaluated at locked `reasoning_effort = "medium"`, `temperature = 0.0`, `max_new_tokens = 2048` on native-resolution real-world charts (`50–336` visual tokens):

| `loop_num_passes` | Effective Layers | Standard Relaxed `±5%` (`n=80`) | Relaxed 95% Cluster CI | Strict `±0.01` | Pct-Norm `±5%` | Truncated | Mean CoT Toks | TTFT P50 (`Ratio`) | TPOT P50 (`Ratio`) | E2E P50 (`Ratio`) | Peak VRAM Max (`Δ`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1`** | `24` (`1.00×`) | **`90.0%` (`72/80`)** | `[82.5%, 96.2%]` | `86.2%` (`69/80`) | **`92.5%` (`74/80`)** | `0` | `228.1` | `110.7 ms` (`1.00×`) | `34.41 ms` (`1.00×`) | **`5.09 s` (`1.00×`)** | `50,661.7 MB` (`+0.0`) |
| **`2`** | `36` (`1.50×`) | `88.8%` (`71/80`) | `[81.2%, 95.0%]` | `86.2%` (`69/80`) | `91.2%` (`73/80`) | `1` | `252.9` | `169.3 ms` (`1.53×`) | `59.00 ms` (`1.71×`) | `8.88 s` (`1.74×`) | `50,727.3 MB` (`+65.6`) |
| **`3`** | `48` (`2.00×`) | `88.8%` (`71/80`) | `[81.2%, 95.0%]` | `86.2%` (`69/80`) | `91.2%` (`73/80`) | `0` | `247.2` | `225.4 ms` (`2.04×`) | `83.71 ms` (`2.43×`) | `12.73 s` (`2.50×`) | `50,787.8 MB` (`+126.2`) |
| **`5`** | `72` (`3.00×`) | **`90.0%` (`72/80`)** | `[82.5%, 96.2%]` | `86.2%` (`69/80`) | **`92.5%` (`74/80`)** | `0` | `230.2` | `339.2 ms` (`3.06×`) | `132.74 ms` (`3.86×`) | `19.79 s` (`3.89×`) | `50,869.9 MB` (`+208.2`) |
| *Ext. Ref: `Gemini 4 Argon` (`MEDIUM`)* | *N/A* | *`91.25%` (`73/80`)* | *`[83.8%, 97.5%]`* | *`87.5%` (`70/80`)* | *`93.75%` (`75/80`)* | *`0`* | *—* | *—* | *—* | *—* | *External Ref Only* |

- **Paired `image_id` Cluster Bootstrap Contrasts (`40` chart clusters)**:
  - `loop3 − loop1` (Relaxed `±5%`): **`-1.25%`**, 95% CI `[-6.25%, +2.50%]`; `loop5 − loop1`: **`0.00%`**, 95% CI `[-3.75%, +3.75%]`.
  - Under strict `±0.01`, all four depths score identically **`86.25%` (`69/80`)**.
- **Pareto Frontier & Offline Oracle (`ChartQA-80`)**:
  - **`loop=1` is the sole fixed-policy Pareto-optimal operating point** (`90.0%` relaxed / `92.5%` pct-norm at **`5.09s` P50**, **`3.89×` faster** than `loop=5` at identical accuracy).
  - The offline Router Oracle reaches `92.5%` (`74/80`, `+2.50 pp`, 95% CI `[0.00%, +6.25%]`) by assigning `71/80` (`88.8%`) queries to `loop=1`, while `6/80` items remain unsolved across all pass depths (largely matching upstream ChartQA label anomalies flagged in `reports/argon_multimodal_audit.md`, e.g., `chartqa-008` decimal `0.30/0.07` vs `30%/7%`, `chartqa-010-q2`, `chartqa-011-q2`).

---

## 4. Scientific Synthesis of Phase 5

1. **Co-Design Law Between Explicit Effort and Latent Looping**: Explicit reasoning effort and latent pass depth are **neither redundant nor monotonically additive**. Rather than pushing explicit effort to `high` (which causes verbose `570+`-token reasoning loops, `2048`-token truncations, and lower accuracy), pairing **concise explicit CoT (`effort="low"`) with moderate latent depth (`loop=3`, `48L`)** yields the global accuracy peak (**`97.9%`**, statistically significant `+8.33 pp` over both `low_loop1` and `none_loop3`) while keeping explicit CoT short (`348.9` tokens) and latency low (`15.50s` P50).
2. **External Validity on Native-Resolution Real Charts (`ChartQA-80`)**: The exact same hardware scaling laws replicate out-of-distribution on native-resolution real charts (`TTFT 1.00× → 3.06×`, `TPOT 1.00× → 3.86×`, `Peak VRAM +208.2 MB / +0.41%`). Because real-world chart VQA is dominated by visual extraction and concise 1–2 step arithmetic (`~230` CoT tokens), `loop=1` already matches frontier-level accuracy (`90.0%` relaxed / `92.5%` pct-norm) at **`3.89×` lower latency** than `loop=5`—reinforcing the core thesis of `docs/FINAL_PLAN.md`: **latent looping must be deployed selectively where multi-step reasoning benefits, never unconditionally**.
