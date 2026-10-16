# Quality Inspector — Model Comparison (Step 9)

_Generated: 2026-10-01 04:23:48 UTC_

| Model | AUROC | F1 | Pixel AUROC | Latency (ms) | FPS | VRAM (MB) | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| patchcore | 1.0000 | 1.0000 | 0.9974 | 143.43 | 7.0 | 0.00 | `patchcore_metrics.json` |
| efficientad | 0.7900 | 0.8333 | 0.9787 | 53.39 | 18.7 | 0.00 | `efficientad_metrics.json` |
| padim | 1.0000 | 1.0000 | 0.9957 | 80.54 | 12.4 | 0.00 | `padim_metrics.json` |

## Quick read (experimental prototype only)

- **Highest image AUROC:** `patchcore` (1.0000)
- **Highest pixel AUROC:** `patchcore` (0.9974)
- **Lowest mean latency:** `efficientad` (53.39 ms)

> Benchmarks reflect controlled AutoVI test-split experiments on development hardware; not real production-line validation.
