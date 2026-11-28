# Phase 5B: One-Source Real-World Holdout (chartqa80)

- Source: `HuggingFaceM4/ChartQA` (`test` split, `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`), human-authored numeric paired clusters.

| Passes | Layers | Relaxed 5% Acc | Relaxed 95% CI | Strict (±0.01) Acc | Pct-Norm 5% Acc | Truncated | TTFT P50 (ms) | TPOT P50 (ms) | E2E P50 (s) | Peak VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 24 | 90.0% (72/80) | [82.5%, 96.2%] | 86.2% | 92.5% | 0 | 110.7 | 34.41 | 5.09 | 50661.7 |
| 2 | 36 | 88.8% (71/80) | [81.2%, 95.0%] | 86.2% | 91.2% | 1 | 169.3 | 59.00 | 8.88 | 50727.3 |
| 3 | 48 | 88.8% (71/80) | [81.2%, 95.0%] | 86.2% | 91.2% | 0 | 222.4 | 82.59 | 12.54 | 50799.3 |
| 5 | 72 | 90.0% (72/80) | [82.5%, 96.2%] | 86.2% | 92.5% | 0 | 339.2 | 132.74 | 19.79 | 50869.9 |
