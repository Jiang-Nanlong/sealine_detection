# Experiment 4: Cross-Dataset Generalization on Buoy

Total samples: 996
Number of videos: 10

## Overall Performance Comparison

| Dataset | N | ρ Mean (px) | ρ Median | ρ P95 | ρ≤10px | ρ≤20px | θ Mean (°) | θ Median | θ≤1° | θ≤2° |
|---------|---|-------------|----------|-------|--------|--------|------------|----------|------|------|
| Buoy (Zero-shot) | 996 | 5.99 | 3.21 | 19.39 | 87.0 | 95.1 | 0.844 | 0.480 | 75.4 | 90.3 |

**Note**: Buoy evaluation uses MU-SID trained weights without any fine-tuning (zero-shot transfer).

## Per-Video Performance on Buoy

| Video | N | ρ Mean (px) | ρ Median | θ Mean (°) | θ Median | θ≤2° (%) | ρ≤10px (%) |
|-------|---|-------------|----------|------------|----------|----------|------------|
| buoyGT_2_5_3_0 | 100 | 3.45 | 2.07 | 0.532 | 0.317 | 99.0 | 94.0 |
| buoyGT_2_5_3_1 | 100 | 8.50 | 4.25 | 0.969 | 0.405 | 85.0 | 80.0 |
| buoyGT_2_5_3_2 | 100 | 10.15 | 3.92 | 1.368 | 0.733 | 82.0 | 75.0 |
| buoyGT_2_5_3_3 | 100 | 5.66 | 3.89 | 0.771 | 0.445 | 90.0 | 83.0 |
| buoyGT_2_5_3_4 | 100 | 4.80 | 3.43 | 0.857 | 0.590 | 90.0 | 88.0 |
| buoyGT_2_5_3_5 | 98 | 7.06 | 3.62 | 0.919 | 0.589 | 90.8 | 86.7 |
| buoyGT_2_6_3_0 | 100 | 5.10 | 2.85 | 0.997 | 0.579 | 85.0 | 90.0 |
| buoyGT_2_6_3_1 | 98 | 6.94 | 3.28 | 0.697 | 0.421 | 92.9 | 86.7 |
| buoyGT_2_6_3_2 | 100 | 3.51 | 2.61 | 0.578 | 0.443 | 97.0 | 94.0 |
| buoyGT_2_6_3_3 | 100 | 4.72 | 2.99 | 0.752 | 0.464 | 91.0 | 93.0 |

## LaTeX Tables

```latex
\begin{table}[htbp]
\centering
\caption{Cross-Dataset Generalization Performance on Buoy Dataset}
\label{tab:buoy_generalization}
\begin{tabular}{lcccccc}
\toprule
Dataset & N & $\rho$ Mean (px) & $\rho$ Median & $\rho\leq$10px & $\theta$ Mean (°) & $\theta\leq$2° \\
\midrule
Buoy (Zero-shot) & 996 & 5.99 & 3.21 & 87.0\% & 0.844° & 90.3\% \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Per-Video Performance on Buoy Dataset}
\label{tab:buoy_per_video}
\begin{tabular}{lcccccc}
\toprule
Video & N & $\rho$ Mean & $\rho$ Med & $\theta$ Mean & $\theta\leq$2° \\
\midrule
2-5-3-0 & 100 & 3.45 & 2.07 & 0.532 & 99.0\% \\
2-5-3-1 & 100 & 8.50 & 4.25 & 0.969 & 85.0\% \\
2-5-3-2 & 100 & 10.15 & 3.92 & 1.368 & 82.0\% \\
2-5-3-3 & 100 & 5.66 & 3.89 & 0.771 & 90.0\% \\
2-5-3-4 & 100 & 4.80 & 3.43 & 0.857 & 90.0\% \\
2-5-3-5 & 98 & 7.06 & 3.62 & 0.919 & 90.8\% \\
2-6-3-0 & 100 & 5.10 & 2.85 & 0.997 & 85.0\% \\
2-6-3-1 & 98 & 6.94 & 3.28 & 0.697 & 92.9\% \\
2-6-3-2 & 100 & 3.51 & 2.61 & 0.578 & 97.0\% \\
2-6-3-3 & 100 & 4.72 & 2.99 & 0.752 & 91.0\% \\
\bottomrule
\end{tabular}
\end{table}
```