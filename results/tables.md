### Table 1 — paper scale: 100 samples (60/20/20), noise σ = 0.02, mean ± std over 3 seeds

| Model | Paper | Accuracy (synthetic) | Acc. Δx | Acc. Δy | Within ±1 nm | MAE (nm) |
|---|---:|---:|---:|---:|---:|---:|
| Majority class (chance) | – | 4.2 ± 0.6% | 4.5 ± 0.7% | 4.0 ± 0.5% | 7.9 ± 0.7% | 14.9 ± 0.3 |
| `Linear Regression` | 10.5% | 3.2 ± 0.3% | 3.1 ± 0.3% | 3.3 ± 0.2% | 9.3 ± 0.3% | 8.0 ± 0.0 |
| `Linear Classification` | 18.3% | 3.6 ± 0.2% | 3.7 ± 0.2% | 3.4 ± 0.3% | 8.9 ± 0.4% | 12.8 ± 0.3 |
| `CNN Regression` | 22.7% | 3.0 ± 0.0% | 3.1 ± 0.2% | 3.0 ± 0.2% | 9.1 ± 0.1% | 8.3 ± 0.1 |
| SegNet + regression head (ablation) | – | 10.4 ± 0.5% | 10.6 ± 0.4% | 10.1 ± 0.9% | 27.7 ± 1.0% | 4.1 ± 0.2 |
| **`SegMet` (ours)** | **67.8%** | 11.6 ± 1.3% | 11.8 ± 1.3% | 11.3 ± 1.3% | 28.4 ± 2.0% | 4.2 ± 0.3 |

### Table 2 — noise sweep (100 samples, accuracy, mean ± std over seeds)

| Model | σ = 0.0 | σ = 0.02 | σ = 0.05 | σ = 0.1 |
|---|---:|---:|---:|---:|
| Majority class (chance) | 4.2 ± 0.8% | 4.2 ± 0.6% | 4.2 ± 0.6% | 4.2 ± 0.6% |
| `Linear Regression` | 2.9 ± 0.1% | 3.2 ± 0.3% | 3.2 ± 0.3% | 3.3 ± 0.2% |
| `Linear Classification` | 3.6 ± 0.0% | 3.6 ± 0.2% | 3.6 ± 0.2% | 3.6 ± 0.2% |
| `CNN Regression` | 3.0 ± 0.1% | 3.0 ± 0.0% | 3.4 ± 0.2% | 3.3 ± 0.4% |
| SegNet + regression head (ablation) | 10.7 ± 0.3% | 10.4 ± 0.5% | 9.4 ± 0.3% | 8.0 ± 0.5% |
| **`SegMet` (ours)** | 12.0 ± 0.8% | 11.6 ± 1.3% | 10.6 ± 0.9% | 9.3 ± 1.1% |

### Table 3 — dataset size (noise σ = 0.02, accuracy, mean ± std over seeds)

| Model | 100 samples (60 train) | 1000 samples (600 train) |
|---|---:|---:|
| Majority class (chance) | 4.2 ± 0.6% | 3.8% |
| `Linear Regression` | 3.2 ± 0.3% | 3.0% |
| `Linear Classification` | 3.6 ± 0.2% | 3.9% |
| `CNN Regression` | 3.0 ± 0.0% | 3.0% |
| SegNet + regression head (ablation) | 10.4 ± 0.5% | 19.5% |
| **`SegMet` (ours)** | 11.6 ± 1.3% | 21.7% |
