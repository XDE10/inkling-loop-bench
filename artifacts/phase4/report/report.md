# Phase 4 Main Synthetic Study (synth240)

## Causal Pass-Depth Curve (`image_mode=correct`, `effort=medium`)

| Passes | Eff. Layers | Acc (`n`) | 95% Cluster CI | Standard (`v1`) | Hard (`v2`) | Chart | Geom | TTFT P50 (ms) | TPOT P50 (ms) | E2E P50 (ms) | Peak VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 24 | 93.8% (225/240) | [90.4%, 96.7%] | 94.2% | 93.3% | 99.2% | 88.3% | 115.9 | 35.10 | 9356.7 | 50697.2 |
| 2 | 36 | 92.9% (223/240) | [88.8%, 96.7%] | 95.8% | 90.0% | 100.0% | 85.8% | 172.6 | 58.43 | 15916.4 | 50759.7 |
| 3 | 48 | 93.8% (225/240) | [90.0%, 97.1%] | 97.5% | 90.0% | 98.3% | 89.2% | 229.5 | 83.03 | 21053.6 | 50820.2 |
| 5 | 72 | 95.4% (229/240) | [92.1%, 97.9%] | 98.3% | 92.5% | 99.2% | 91.7% | 345.0 | 132.68 | 33884.8 | 50940.4 |

## Paired `image_id` Cluster Bootstrap Differences

| Comparison | Point Δ Acc | 95% Cluster Bootstrap CI |
| :--- | :---: | :---: |
| `loop2_minus_loop1` | -0.83% | [-5.00%, +2.92%] |
| `loop3_minus_loop1` | +0.00% | [-3.75%, +3.75%] |
| `loop5_minus_loop1` | +1.67% | [-1.67%, +5.42%] |
| `loop3_minus_loop2` | +0.83% | [-2.50%, +4.58%] |
| `loop5_minus_loop3` | +1.67% | [-1.67%, +5.00%] |

## Predeclared `synth-320` Uncertainty Rule Decision

- `trigger_synth320`: `False`
- `reasons`: `[]`
