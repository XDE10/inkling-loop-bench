# Independent Multimodal Dataset Audit & External Frontier Reference (`Gemini 4 Argon`)

Date: 2026-11-24  
Model: `google/gemini-4-argon` (`thinking_level="MEDIUM"`, accessed via 5 OpenCode Go shards with persistent SQLite WAL caching)  
Role: **Blind visual solvability QA and external frontier reference ONLY** (never mixed into the `TML-d24-varloop-research` pass-depth curve per `docs/FINAL_PLAN.md`).

---

## 1. Two-Layer Caching Architecture

- **Layer 1 (Local Content-Addressed SQLite WAL Cache)**: `~/.cache/loopbench/argon_cache.sqlite3` (`src/loopbench/argon_client.py`) hashes `(model_id, generationConfig, image_sha256, prompt)` so resumed or repeated audit queries incur `0` network requests and `$0.00` quota consumption (`40` batch payloads persisted across `synth-240` and `ChartQA-80`).
- **Layer 2 (Multi-Key Sticky Session + Prefix Alignment)**: Batches `4` images (`8` paired questions) per request across `5` isolated API keys (`key_0..key_4`) with sticky `x-opencode-session` headers and a canonical `contents[0]` specification prefix (`>4,096` token cache boundary empirically verified on `google/v1beta:generateContent` to trigger `cachedContentTokenCount = 4,340` at `$0.10/M` vs `$2.00/M`).

---

## 2. Phase 4 `synth-240` Blind Solvability Audit (`120` Images / `240` Questions)

`gemini-4-argon` solved all `120` images × `2` tasks (`direct`, `multi_step`) **blindly from pixels + prompt text alone** (zero ground-truth hints):

| Subset (`n` questions) | Blind Agreement vs Ground Truth | Interpretation |
| :--- | :---: | :--- |
| **`chart / standard (v1) / direct`** (`30`) | **`100.0%` (`30/30`)** | Exact 5-unit grid & bar/series label legibility verified |
| **`chart / standard (v1) / multi_step`** (`30`) | **`100.0%` (`30/30`)** | Zero ambiguity across all 6 `v1` chart sub-templates |
| **`chart / hard (v2) / direct`** (`30`) | **`100.0%` (`30/30`)** | Exact pixel extraction across 3×4 grouped, dual-line, 3-tier stack, diverging, band, and matrix charts |
| **`chart / hard (v2) / multi_step`** (`30`) | **`100.0%` (`30/30`)** | 100% multi-hop arithmetic agreement |
| **`geometry / standard (v1) / direct`** (`30`) | **`100.0%` (`30/30`)** | Zero label occlusion across all 6 `v1` Euclidean sub-templates |
| **`geometry / standard (v1) / multi_step`** (`30`) | **`100.0%` (`30/30`)** | Exact Euclidean angle consistency verified |
| **`geometry / hard (v2) / direct`** (`30`) | **`100.0%` (`30/30`)** | **120/120 (`100.0%`) direct visual extraction across entire `synth-240` dataset** |
| **`geometry / hard (v2) / multi_step`** (`30`) | **`80.0%` (`24/30`)** | 5 single-pass JSON-order slips on `geometry-v2-4` (self-corrected to exact ground truth inside `note`) + 1 supplementary-ray slip on `geometry-v2-5` (solved `101.0°` by `TML-d24-varloop-research`) |
| **Overall `synth-240` (`240`)** | **`97.5%` (`234/240`), 95% CI `[95.4%, 99.2%]`** | **`120 / 120` (`100%`) images rated `"clear"`**; `240 / 240` programmatic ground-truth answers verified exact |

---

## 3. Phase 5 `ChartQA-80` Real-World Holdout Blind Reference (`40` Real Charts / `80` Human Questions)

Curated deterministically (`seed = 20261125`) from `HuggingFaceM4/ChartQA` (`split="test"`, pinned SHA `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`, `283` eligible human-authored numeric paired chart clusters):

| Metric | `Gemini 4 Argon` (`MEDIUM`) Score | `image_id` Cluster 95% CI (`40` clusters) |
| :--- | :---: | :---: |
| **Strict Numeric Match (`±0.01`)** | **`87.50%` (`70/80`)** | `[80.00%, 95.00%]` |
| **Standard ChartQA Relaxed (`±5%`)** | **`91.25%` (`73/80`)** | `[83.75%, 97.50%]` |
| **Percentage-Normalized Relaxed (`±5%`)** | **`93.75%` (`75/80`)** | `[87.50%, 98.75%]` |

*Audit note*: `chartqa-008` (`q1`, `q2`) stores percentage bars `30%` and `7%` as decimal fractions `0.3` and `0.07` in the upstream ChartQA ground truth; reporting both standard `±5%` and percentage-normalized `±5%` in `build_phase5_reports()` accounts for upstream convention differences without altering frozen holdout labels.
