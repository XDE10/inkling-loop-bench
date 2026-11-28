# Phase 8 Report: Official `Inkling-2-Mini` Cross-Model Check, Native-Rollback Self-Speculative Drafting, and True-Coda Cross-Pass Probes

> **Verdict (Post-Revision Verified on Blackwell G4)**: Phase 8 evaluates **`329` Official `Inkling-2-Mini` runs** (plus `bs=1 vs bs=4` and weight-load checks), **`50` Blackwell G4 self-speculative decoding runs** upgraded to native `transformers 5.24.1` O(1) cache cropping (plus a `draft=3` perfect-draft control and first-mismatch logit-margin audit), and **`1,920` True-Coda + `logit_softcap` teacher-forced cross-pass probes** (`20` stratified `MathVista-Hard-83` items × `5` passes on the deployed `loop=3` trajectory):
> 1. **Phase 8A (Official `Inkling-2-Mini` vs. `TML-d24-varloop-research`, `48L`, both BF16 fallback)**: Under HuggingFace `transformers 5.24.1` (skipping `modelopt` quantization on **both** checkpoints, verified **`0` missing / `0` unexpected / `0` mismatched keys** in [model_load_reports.json](../artifacts/phase8a_mini/model_load_reports.json)), Official Mini reaches **`87.95%` (`73/83`)** on `MathVista-Hard-83 (low)` (`+2.41 pp` vs. varloop `85.54%`, McNemar exact `p=0.727`), **`84.34%` (`70/83`)** at `medium` (`+1.20 pp`, `p=1.000`), **`6.02%` (`5/83`)** blank-image control (`+81.93 pp` visual gain), and **`90.00%` relaxed / `92.50%` pct-norm** on `ChartQA-80` (`+1.25 pp`, `p=1.000`). Point estimates favor `low` over `medium` (`+3.61 pp`, `p=0.375`, truncations `11 → 4`), while Mini's CoT is `10%–12%` *longer* on MathVista (`400.0` vs. `364.1`; `555.0` vs. `495.2`) and `22.5%` shorter on ChartQA (`191.7` vs. `247.2`). Per the official model card, we do **not** claim statistical equivalence between `varloop` and `Mini` (no TOST margin test; `bs=4` left-padding also induces BF16 trajectory drift: `7/8` numeric predictions match `bs=1`, `2/8` exact text match).
> 2. **Phase 8B (Self-Speculative Loop Drafting `loop=1 → loop=3`, Native `activate_past_recording()` + `crop(-n)`)**: Replacing our initial manual snapshot+replay rollback (`0.931×–0.831×`, `9.5%–14.3%` rollback tax) with `transformers 5.24.1` native `DynamicCache.activate_past_recording()` + `crop(-n)` collapses rollback overhead **`140×`** (`14.9–25.1 s → 0.13–0.26 s`, `0.13%` of wall time) and **flips net E2E wall-clock speedup positive**: **`1.050×` (`K=2`), `1.077×` (`K=4`, peak `1.28×` on `chartqa-000-q2`), `1.049×` (`K=6`), and `0.994×` (`K=8`)** at `86.2%–65.3%` acceptance. Crucially, speculative decoding is **not bit-exact** (`exact=2/10–4/10` on `256`-token runs): our **`draft_passes=3 == target_passes=3` perfect-draft control** (`4/4` items diverge despite `96.7%–98.7%` acceptance) and **teacher-forced logit margins** prove mismatches stem from BF16 `seqlen>1` (`causal_conv1d_fn` + batched GEMM) vs. `seqlen==1` (`causal_conv1d_update` + GEMV) + MoE top-k routing non-determinism, not a rollback bug.
> 3. **Phase 8C (True-Coda + `final_input` + `logit_softcap` Teacher-Forced Cross-Pass Probes)**: Running each pass (`p ∈ {1..5}`) through the full Coda (layers `18–23`), `final_input` residual blend, `norm`, `lm_head`, and `logit_softcap` on the deployed `loop=3` trajectory shows smooth next-token entropy reduction from Pass 1→3 (`0.518 → 0.485 → 0.459 nats`) followed by a plateau (`0.458–0.459` at Pass 4–5), with `p1→p3` top-1 agreement rising `89.06% → 93.65% → 100.00%` (and `p3` teacher-forced vs. autoregressive `model.generate()` matching on `97.81%` of steps). **We retract our earlier skipped-Coda Banach-contraction and `T->F` overthinking claims**: Pass 2+ restarts from Prelude with a gated message and separate per-pass cache (so `adjacent_delta_ratio` is empirical, not a Banach constant), and with true Coda logits `F->T` (`p1_ent=0.463`) and `T->F` (`p1_ent=0.442`) are virtually indistinguishable while `F->F` (`0.627`) has the highest entropy. Offline teacher-forced entropy early-exit (`entropy <= 0.30`) yields **`27.23%` total layer-visit saving** (`36.30%` core) at **`99.84%`** single-step fidelity vs. `loop=3` (`37.52%` total saving vs. `loop=5`), subject to the critical caveat that offline teacher-forced probes do not model multi-step autoregressive error accumulation or skipped-pass cache staleness.

---

## 1. Execution Provenance, Weight-Load Audit & Precision Disclosure

| Sub-Phase | Model & Revision | Hardware | Scope | Cloud Archive (`rclone`) |
| :--- | :--- | :--- | :--- | :--- |
| **8A** | `thinkingmachines/Inkling-2-Mini` @ `c9d028302cd0f035fdf8e7e91f089a4ce94d2195` (`is_varloop=False`, fixed `48L`) | NVIDIA RTX PRO 6000 Blackwell (96 GB, `sm_120`, CUDA 13.0, `torch 2.14.1`, `transformers 5.24.1`) | `MathVista-Hard-83` (`low`, `medium`, `blank`, `bs=4`) + `ChartQA-80` (`medium`, `bs=4`) + `bs=1 vs bs=4` check (`8` items) + Systems Probe (`329` evals) | `mydrive:inkling_loop_bench/phase8a/` & `phase8_rev/` |
| **8B** | `thinkingmachines/TML-d24-varloop-research` @ `c90921bd5f78df6c8168173b9f8a3a850320c594` | NVIDIA RTX PRO 6000 Blackwell (96 GB, `sm_120`) | `10` items (`6` MathVista + `4` ChartQA) × (`baseline` + `K ∈ {2,4,6,8}` native `crop(-n)`) + `4`-item `draft=3` control + logit-margin audit (`max_new_tokens=256`) | `mydrive:inkling_loop_bench/phase8_rev/` |
| **8C** | `thinkingmachines/TML-d24-varloop-research` @ `c90921bd5f78df6c8168173b9f8a3a850320c594` | NVIDIA RTX PRO 6000 Blackwell (96 GB, `sm_120`) | `20` stratified `MathVista-Hard-83` items (`5` `F->T`, `3` `T->F`, `6` `F->F`, `6` `T->T`) × `p ∈ {1..5}` True-Coda + softcap × `96` tokens (`1,920` steps) | `mydrive:inkling_loop_bench/phase8_rev/` |

> **Dual-Model Precision & Loader Audit (`model_load_reports.json`)**: Both pinned HuggingFace checkpoints (`TML-d24-varloop-research` and `Inkling-2-Mini`) natively ship with NVFP4/MXFP8 + partial BF16 weights (~`19.5 GB`). Under stock HuggingFace `transformers 5.24.1` (without external NVIDIA ModelOpt kernels), **both models—across every Phase 2–8 run—skip `modelopt` quantization and execute on the non-native BF16 fallback path (~`50.6–51.9 GB` VRAM)**. Direct `from_pretrained(..., output_loading_info=True)` inspection confirms **`0` missing keys, `0` unexpected keys, and `0` mismatched keys** for both checkpoints (`load_s = 4.73 s` for `varloop`, `64.89 s` for `Mini`; [model_load_reports.json](../artifacts/phase8a_mini/model_load_reports.json)).

> **Batching Consistency Check (`bs=1` vs. `bs=4`, `bs1_vs_bs4_check.json`)**: Because Phase 8A evaluated `Inkling-2-Mini` at `bs=4` (left-padded) while Phase 7B evaluated `varloop` at `bs=1`, we directly compared `bs=1` vs. `bs=4` on `8` MathVista items (`low`, `loop=3`, `TML-d24-varloop-research`). Although attention masks correctly mask padding, BF16 batched kernels change intermediate rounding: **`7/8` final numeric predictions match**, whereas only **`2/8` exact answer strings** and **`1/8` exact reasoning traces** are bit-identical, and `mathvista-0037-q1` flips numeric output (`-1.0` at `bs=1` vs. `4.0` at `bs=4`). Accordingly, `bs=4` amortized latencies are reported strictly separately from `bs=1` single-stream latencies.

---

## 2. Phase 8A: Official `Inkling-2-Mini` vs. `TML-d24-varloop-research` (`48L`)

### 2.1 Head-to-Head Accuracy, CoT Length, and Exact Statistical Tests

| Benchmark & Effort (Locked `48L` / `3`-pass, BF16 Fallback) | Official `Inkling-2-Mini` (`bs=4`) | Research `varloop` (`loop=3`, `bs=1`) | Paired Δ (`Mini − Varloop`), 95% Cluster CI | Flips (`V→M`: `F→T / T→F`) | Exact McNemar `p` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`MathVista-Hard-83` (`low`)** | **`87.95%`** (`73/83`, `W=5, P=1, T=4`, CoT `400.0`) | `85.54%` (`71/83`, `W=11, P=0, T=1`, CoT `364.1`) | `+2.41 pp` `[-3.61%, +9.64%]` (n.s.) | `5 / 3` | `0.727` |
| **`MathVista-Hard-83` (`medium`)** | **`84.34%`** (`70/83`, `W=1, P=1, T=11`, CoT `555.0`) | `83.13%` (`69/83`, `W=6, P=0, T=8`, CoT **`495.2`**) | `+1.20 pp` `[-6.02%, +8.43%]` (n.s.) | `5 / 4` | `1.000` |
| **`MathVista-Hard-83` (`blank` control)** | `6.02%` (`5/83`, CI `[1.20%, 10.84%]`) | `6.02%` (`5/83`, CI `[1.20%, 10.84%]`) | `0.00 pp` (Visual gain **`+81.93 pp`** `[+73.49%, +90.36%]`) | — | — |
| **`ChartQA-80` (`medium`, strict)** | `86.25%` (`69/80`, `T=0`, CoT **`191.7`**) | `86.25%` (`69/80`, `T=0`, CoT `247.2`) | `0.00 pp` `[-3.75%, +3.75%]` (n.s.) | `1 / 1` | `1.000` |
| **`ChartQA-80` (`medium`, relaxed ±5%)** | **`90.00%`** (`72/80`) | `88.75%` (`71/80`) | `+1.25 pp` `[-2.50%, +6.25%]` (n.s.) | `2 / 1` | `1.000` |
| **`ChartQA-80` (`medium`, pct-norm)** | **`92.50%`** (`74/80`) | `91.25%` (`73/80`) | `+1.25 pp` `[-2.50%, +6.25%]` (n.s.) | `2 / 1` | `1.000` |

*Per-source breakdown on `MathVista-Hard-83` (`Inkling-2-Mini` `low` / `medium`)*: `FunctionQA 24/26 | 24/26`, `GEOS 5/5 | 5/5`, `PaperQA 6/6 | 6/6`, `TabMWP 26/26 | 26/26`, `IQTest 8/13 | 7/13`, `TheoremQA 4/7 | 2/7`. Full failure breakdown at `medium`: `11` truncations at `2048` tokens, `1` wrong answer, `1` parse failure (`70/83` overall).

![Phase 8A Official Mini vs Research Varloop](../artifacts/phase8a_mini/mini_vs_varloop.png)

### 2.2 Key Takeaways & Methodological Caveats
1. **No Statistically Significant Accuracy Gap — Without Claiming Equivalence**: All pairwise `Mini − Varloop` comparisons are statistically non-significant (exact McNemar `p = 0.727–1.000`). Consistent with the official `TML-d24-varloop-research` model card (which explicitly advises against treating the research checkpoint as equivalent to production `Inkling-2-Mini`, and already publishes official `1–5`-pass MathVision scaling `74.8 → 88.1`), non-significance at `n=80–83` does **not** prove statistical equivalence (which would require a predeclared TOST equivalence margin) and also reflects `bs=4` vs. `bs=1` batching differences.
2. **CoT Length Direction Depends on Domain**: Official Mini generates **`10%–12%` longer CoT** than `varloop` on `MathVista-Hard-83` (`400.0 vs. 364.1` at `low`; `555.0 vs. 495.2` at `medium`, correcting an earlier `545.8` table typo) and **`22.5%` shorter CoT** on `ChartQA-80` (`191.7 vs. 247.2`).
3. **`low` vs. `medium` on Official Mini**: Point estimates replicate the `low`-effort advantage (`87.95%` vs. `84.34%`, `Δ = +3.61 pp`, 95% CI `[-1.20%, +8.43%]`, McNemar `p = 0.375`, `4` wins vs. `1` loss), cutting mean CoT `-27.9%` (`555.0 → 400.0`), truncations `-63.6%` (`11 → 4`), and `bs=4` amortized mean latency `-20.9%` (`29.74 s → 23.53 s`; strictly `bs=4` amortized, not directly comparable to Phase 7B's `bs=1` latencies).
4. **Identical HF Reference Systems Footprint**: Official Mini's Blackwell G4 systems probe ([mini_systems_probe.json](../artifacts/phase8a_mini/mini_systems_probe.json)) matches Phase 7C: **`192.0 KB/tok`** (`328` `DynamicCache` tensors within the `<=1024`-token sliding window), `decode_core = 64.4 ms` (`74.4%` of TPOT), and batch scaling from **`12.09 tok/s` (`bs=1`) → `48.03 tok/s` (`bs=4`, `3.97×`) → `70.81 tok/s` (`bs=8`, `5.86×`, `51.8 GB` VRAM, `0` OOM)**.

---

## 3. Phase 8B: Self-Speculative Loop Drafting (`loop=1 → loop=3`) on Blackwell G4

### 3.1 Native `activate_past_recording()` + `crop(-n)` Upgrade vs. Snapshot-Replay Ablation

Runtime introspection showed that all `48` `DynamicCache` layers in `Inkling2LoopForConditionalGeneration` are hybrid `LinearAttentionAndFullAttentionLayer` / `LinearAttentionAndSlidingWindowAttentionLayer` slots holding `conv_states`, `recurrent_states`, and `sliding_window=1024` KV buffers. In `transformers 5.24.1`, `DynamicCache(config=text_cfg)` natively supports O(1) hybrid-state rollback via **`cache.activate_past_recording()`** before the first forward pass followed by **`cache.crop(-n_rejected)`** (and `cache.crop(0)` on full acceptance to release the recording buffer). Upgrading `src/loopbench/speculative.py` from manual deep-clone snapshot + prefix replay to native `crop(-n)` eliminated **`99%` of rollback overhead** and flipped net E2E speedup positive across `10` items (`max_new_tokens=256`, `1,762` baseline tokens, naive `loop=3` baseline = `149.03 s`, `84.6 ms/tok`):

| Draft Horizon `K` | Rollback Mode | Overall Accept % | Thinking / Answer % (`Ans Share`) | Tok / Verify Round | Exact Match (`256`-tok) | Total Spec Time (vs Base) | **Net E2E Speedup** (Peak Item) | Time Split (`Draft / Verify / Rollback`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`K = 2`** | **Native `crop(-n)`** | **`86.19%`** | `85.06% / 100.0%` (`5.6%`) | `2.70` | `2/10` | `141.89 s` (`149.03 s`) | **`1.050×`** (`1.18×`) | `44.4% / 53.0% / 0.18%` (`0.26 s`) |
| **`K = 4`** | **Native `crop(-n)`** | `77.69%` | `76.03% / 100.0%` (`5.8%`) | `4.07` | `3/10` | **`138.34 s`** (`149.03 s`) | **`1.077×`** (**`1.28×`**) | `59.7% / 37.6% / 0.13%` (`0.18 s`) |
| **`K = 6`** | **Native `crop(-n)`** | `72.04%` | `70.84% / 87.6%` (`5.3%`) | `5.24` | `3/10` | `142.11 s` (`149.03 s`) | **`1.049×`** (`1.30×`) | `67.6% / 29.8% / 0.11%` (`0.15 s`) |
| **`K = 8`** | **Native `crop(-n)`** | `65.31%` | `63.81% / 84.8%` (`5.6%`) | `6.06` | `4/10` | `149.92 s` (`149.03 s`) | **`0.994×`** (`1.33×`) | `72.6% / 25.0% / 0.09%` (`0.13 s`) |
| *Ablation: `K=2..8`* | *Manual snapshot+replay* | *`86.8%→64.4%`* | *`85.7%→62.6%`* | *`2.74→6.38`* | *`3/10–4/10`* | *`156.8→175.6 s` (`146.0 s`)* | *`0.931×→0.831×`* | *Rollback = **`9.5%–14.3%`** (`14.9–25.1 s`)* |

![Phase 8B Self-Speculative Speedup & Acceptance](../artifacts/phase8b_speculative/speculative_speedup.png)

### 3.2 Causal Diagnosis of Non-Bit-Exactness (`draft=3` Perfect-Draft Control + Logit Margins)

Contrary to our initial draft (which inaccurately labeled greedy self-speculation "lossless / bit-exact" and claimed divergence occurred at identical positions across all `K`), `6/10–8/10` items diverge over `256` tokens, and mismatch positions vary across `K` (e.g., `mathvista-0088-q1`: `mm = 118, 66, 118, 66`; `mathvista-0098-q1`: `mm = 179, 58, 179, 151`). To separate cache-rollback bugs from BF16 kernel numerical drift, we ran a **`draft_passes=3 == target_passes=3` (`K=4`, native `crop`) perfect-draft control** and computed **full-sequence teacher-forced prefill logit margins** at the exact first mismatch token ([speculative_summary.json](../artifacts/phase8b_speculative/speculative_summary.json)):

| Control Item (`draft=3, target=3, K=4`) | Accept % | First Mismatch `idx` | TF `top1 − top2` Logit Margin | `baseline` Rank in TF | `spec` Rank in TF | Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `mathvista-0037-q1` | `96.67%` | `57` | `0.457` | **`3`** | **`1`** | Spec matches full-sequence teacher-forced prefill; autoregressive `generate()` drifted |
| `mathvista-0088-q1` | `98.08%` | `118` | `0.324` | `1` | `2` | `seqlen>1` vs `seqlen==1` BF16 + MoE routing flip |
| `mathvista-0241-q1` | `98.08%` | `176` | **`0.023`** | `1` | `2` | Razor-thin BF16 argmax tie (`Δ = 0.023` logit) |
| `chartqa-001-q2` | `98.73%` | `50` | `0.147` | **`2`** | **`1`** | Spec matches full-sequence teacher-forced prefill; autoregressive `generate()` drifted |

Even when drafting with the **identical `loop=3` model** using native `crop()`, `4/4` control items still diverge from autoregressive `model.generate()`, and on `2/4` items the speculative token is actually **rank-1** under full-sequence teacher-forced prefill while the autoregressive baseline is rank-2/3. Combined with Phase 8C showing `loop=3` teacher-forced prefill agrees with `loop=3` autoregressive `model.generate()` on only `97.81%` of tokens, this causally attributes the mismatches to **BF16 `seqlen>1` (`causal_conv1d_fn` + batched GEMM) vs. `seqlen==1` (`causal_conv1d_update` + GEMV) numerical differences amplified by 8-of-128 MoE top-k routing**, rather than a rollback bug.

### 3.3 Why Eager Speedup Caps at `1.08×` & Realistic Serving Roofline
1. **Heavy `24L/48L` Draft Cost Dominates (`44%–73%` of Wall Time)**: Once native `crop(-n)` removes rollback (`0.13%`), drafting alone consumes `82.6 s` of `138.3 s` at `K=4` (`~45–48 ms/tok` vs. `~84.6 ms/tok` baseline), while `verify` takes `~116–129 ms/round` (pointing to eager-PyTorch MoE expert-dispatch overhead that warrants kernel profiling).
2. **Honest Idealized Ceiling**: With rollback already near zero, **making drafting completely free** would cap speedup at **`1.54×` (`K=2`), `1.89×` (`K=4`), `2.09×` (`K=6`), and `2.18×` (`K=8`)**—not `3×+`—because parallel verification of `K+1` tokens still pays `~1.4–1.5×` single-token decode latency in eager PyTorch.
3. **Small End-to-End Scope of Answer-Stage Downshifting**: Although `loop=1` achieves `84.8%–100.0%` agreement on final-answer tokens, final answers comprise only **`5.3%–5.8%`** of drafted tokens (`answer_token_share`), bounding whole-response E2E speedup from answer-stage downshifting to **`~3%`**.
4. **High-Leverage Next Steps & Production Roofline Context**: Real serving speedups require (a) benchmarking the repository's official **`mtp.safetensors`** multi-token prediction drafter, (b) **Prelude + Pass-1 KV/state sharing** between `loop=1` draft and `loop=3` verify (since Prelude and Pass 1 are invariant to `loop_num_passes`), and (c) quantized fused engines—for context, the official HuggingFace inference blog reports **`157 tok/s`** (vLLM, no speculation) and **`190.6 tok/s`** (DSpark) on the same RTX PRO 6000 Blackwell (`~13–16×` above our `12.1–12.3 tok/s` HF BF16 eager baseline, though under native quantized precision and different prompt/output workloads).

---

## 4. Phase 8C: True-Coda + `logit_softcap` Cross-Pass Convergence & Offline Early-Exit

To fix our earlier layer-17 Logit-Lens shortcut (which skipped Coda layers `18–23`, `final_input` blending, and `logit_softcap` and never compared against deployed `loop=3`), we upgraded `src/loopbench/probes.py` to generate each question's real deployed **`loop=3`** trajectory (`20` stratified `MathVista-Hard-83` items × `96` tokens = `1,920` steps on Blackwell G4) and run exact single-shot **full-model teacher-forced forwards** for every `p ∈ {1, 2, 3, 4, 5}` through the true Coda, `final_input`, `norm`, `lm_head`, and `logit_softcap` ([probes_summary.json](../artifacts/phase8c_probes/probes_summary.json)):

### 4.1 Overall Per-Pass Convergence (True Coda + `logit_softcap`, `n = 1,920` Steps)

| Pass (`Eff. Layers`) | Agree w/ `loop=3` | Agree w/ `loop=5` | Agree w/ Actual Gen | Mean Entropy (nats) | Mean KL to `loop=3` / `loop=5` | Cos Sim vs. Prev | Rel L2 Delta | Empirical `adjacent_delta_ratio` (Mean / P50, `<1` %) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pass 1 (`24L`)** | `89.06%` | `89.01%` | `89.27%` | `0.5181` | `0.0857 / 0.1058` | — | — | — |
| **Pass 2 (`36L`)** | `93.65%` | `93.13%` | `94.06%` | `0.4854` | `0.0354 / 0.0564` | `0.9713` | `0.2307` | — |
| **Pass 3 (`48L`)** | **`100.00%`** | `96.20%` | **`97.81%`** | **`0.4592`** | `0.0000 / 0.0187` | `0.9777` | `0.2019` | `0.911 / 0.895` (`70.3%`) |
| **Pass 4 (`60L`)** | `97.34%` | `98.18%` | `96.51%` | `0.4578` | `0.0148 / 0.0061` | `0.9927` | `0.1123` | **`0.563 / 0.544`** (`98.6%`) |
| **Pass 5 (`72L`)** | `96.20%` | `100.00%` | `95.42%` | `0.4585` | `0.0226 / 0.0000` | `0.9956` | `0.0858` | `0.769 / 0.745` (`90.2%`) |

- **Why We Retract the Banach Fixed-Point Claim**: In `modeling_inkling2_loop.py`, Pass 2+ does **not** iterate a single map $h_{p}=F(h_{p-1})$; each pass re-enters from the Prelude hidden state combined with a gated message from the previous pass and reads/writes an independent per-pass cache. Thus `adjacent_delta_ratio` ($\|h_p - h_{p-1}\|_2 / \|h_{p-1} - h_{p-2}\|_2$) is strictly an **empirical step-size decay metric**, not a Banach contraction proof. Likewise, the sharp flattening after Pass 3 (`entropy 0.518 → 0.459 → 0.458`) is consistent with—rather than a causal explanation of—the official training mix (`1:2:6:2:1` passes + `3`-pass post-training).
- **Independent Confirmation of BF16 Prefill vs. Autoregressive Drift**: Even at `p=3`, single-shot teacher-forced logits agree with the actual autoregressive `loop=3` `model.generate()` tokens on `97.81%` (differing on `2.19%` of tokens), independently corroborating Phase 8B's finding.

### 4.2 Transition-Group Audit (`F->T` vs. `T->F` — Honest Retraction of Overthinking Claim)

| Phase 7B Transition Group (`loop=1 → loop=3`) | Probed Steps | Mean First-Stable Pass | Pass 1 Agree w/ `loop=3` | Pass 1 Entropy | Pass 3 Entropy | Pass 1 KL to `loop=3` | Empirical `delta_ratio` `p3 / p4` (`p4 < 1` %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`F->T`** (Repaired by `loop=3`, `n=5`) | `480` | `1.258` | `90.00%` | `0.4633` | `0.3820` | `0.0917` | `0.918 / 0.567` (`97.7%`) |
| **`T->F`** (Regressed by `loop=3`, `n=3`) | `288` | `1.257` | `90.97%` | `0.4421` | `0.3857` | `0.0557` | `0.946 / 0.565` (`98.6%`) |
| **`F->F`** (Unsolved at both depths, `n=6`) | `576` | `1.311` | **`86.81%`** | **`0.6273`** | **`0.5799`** | `0.0980` | `0.891 / 0.560` (`99.5%`) |
| **`T->T`** (Solved at both depths, `n=6`) | `576` | `1.280` | `89.58%` | `0.4924` | `0.4397` | `0.0836` | `0.909 / 0.563` (`98.4%`) |

> **Self-Correction**: Once true Coda layers and `logit_softcap` are applied, the previously reported `T->F` entropy rebound disappears completely: both `F->T` (`0.463 → 0.382`) and `T->F` (`0.442 → 0.386`) show nearly identical Pass 1→3 entropy decay and first-stable pass (`1.258` vs. `1.257`). At `n=3` post-hoc items, no reliable `T->F` overthinking signature exists in true-logit entropy; instead, **`F->F` persistently unsolved items carry the highest token uncertainty (`p1_ent=0.627`, `p1_ag3=86.8%`)**.

### 4.3 Offline Teacher-Forced Early-Exit Frontier (vs. Deployed `loop=3` and `loop=5`)

| Offline Early-Exit Rule | Target `loop=3` (`48L`) Mean Passes | **Total Layer Visits Saved (`loop=3`)** (Core Saved) | Offline Fidelity vs. `loop=3` | Target `loop=5` (`72L`) Total Layer Saved (Core) | Offline Fidelity vs. `loop=5` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`entropy <= 0.05`** | `2.126` | **`21.85%`** (`29.13%`) | **`99.95%`** | `30.28%` (`36.33%`) | `100.00%` |
| **`entropy <= 0.15`** | `2.015` | **`24.64%`** (`32.85%`) | **`99.95%`** | `33.86%` (`40.64%`) | `99.95%` |
| **`entropy <= 0.30`** | `1.911` | **`27.23%`** (`36.30%`) | **`99.84%`** | **`37.52%`** (`45.02%`) | **`99.84%`** |
| **`entropy <= 0.50`** | `1.776` | **`30.61%`** (`40.82%`) | **`99.84%`** | `42.09%` (`50.51%`) | `99.84%` |
| `entropy <= 1.00` | `1.413` | `39.69%` (`52.92%`) | `96.51%` | `54.35%` (`65.22%`) | `96.30%` |
| `cos_sim >= 0.980` | `2.592` | `10.20%` (`13.59%`) | `97.55%` | `32.90%` (`39.48%`) | `95.31%` |
| `cos_sim >= 0.940` | `2.068` | `23.31%` (`31.08%`) | `94.17%` | `48.47%` (`58.17%`) | `93.44%` |

![Phase 8C True-Coda Cross-Pass Convergence & Offline Early-Exit](../artifacts/phase8c_probes/latent_probes_early_exit.png)

> **Critical Offline-Evaluation Caveat**: These savings represent an **offline single-step teacher-forced upper bound** (evaluating true post-Coda logits at each pass along a fixed `loop=3` prefix). Online closed-loop token-level early-exit introduces two unmeasured costs: (1) multi-step autoregressive error compounding whenever a single skipped-pass token diverges, and (2) **un-updated KV/recurrent cache slots** for skipped deeper passes (`pass 2/3` caches become stale when a token exits at `pass 1`), which requires explicit cache-state propagation or projection to deploy online.

---

## 5. Contextualizing Within Recent Looped-MoE Literature (`LoopTitan` & HF Inference Hillclimb)

Recent pre-training analysis (**LoopTitan**, arXiv:2611.34821) found that under matched pre-training parameter budgets and uniform-step optimizers (Muon), *parameter revisit itself* provides limited intrinsic accuracy gain over an unshared deeper MoE. **LoopBench does not include an unshared non-revisit control model and therefore cannot independently adjudicate LoopTitan's pre-training claim.** Instead, LoopBench supplies a complementary **deployment-time systems characterization** of released 26.4B looped VLMs (`TML-d24-varloop-research` and `Inkling-2-Mini`):
1. **Inference-Time Cost–Benefit Trade-offs**: On real-world visual math (`ChartQA-80`, `MathVista-Hard-83`), increasing `loop=1 → loop=3` produces modest, statistically non-significant accuracy gains (`-1.25 pp` to `+2.41 pp`, exact McNemar `p=0.727–1.000`) alongside `+48 KB/tok/pass` KV growth (within the `<=1024`-token sliding window) and `+23.1 ms/tok/pass` decode cost in HF eager BF16, while reliably compressing explicit CoT (`-14%` to `-25%`) and reducing `2048`-token truncations.
2. **Inference-Time Levers Enabled by Weight-Tied Recurrence**: Even in eager PyTorch, native O(1) hybrid cache cropping (`activate_past_recording()` + `crop(-n)`) unlocks net-positive zero-training `loop=1→3` self-speculative speedup (`1.077×` at `K=4`, peak `1.28×`), and true-Coda per-pass entropy exposes a strong offline early-exit signal (`27.2%` total layer-visit saving vs. `loop=3` at `99.84%` single-step fidelity)—pointing toward Prelude/Pass-1 KV sharing, `mtp.safetensors` drafting, and closed-loop cache-aware early-exit as the concrete engineering agenda beyond stock BF16 eager serving (cf. the official HuggingFace Blackwell inference hillclimb achieving `157–190.6 tok/s` via quantized vLLM/DSpark).
