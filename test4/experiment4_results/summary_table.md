# Experiment 4: Cross-Dataset Generalization Results

## Overall Performance Comparison

| Dataset | N | ρ Mean (px) | ρ Median | ρ P95 | ρ≤10px (%) | ρ≤20px (%) | θ Mean (°) | θ Median | θ≤1° (%) | θ≤2° (%) |
|---------|---|-------------|----------|-------|------------|------------|------------|----------|----------|----------|
| SMD (Zero-shot) | 2998 | 21.31 | 5.72 | 118.50 | 68.5 | 81.5 | 0.662 | 0.285 | 88.9 | 94.5 |
| MU-SID (In-domain) | 268 | 2.62 | 1.99 | 7.58 | 98.9 | 99.6 | 0.214 | 0.115 | 97.0 | 99.3 |

**Note**: SMD evaluation uses MU-SID trained weights without any fine-tuning (zero-shot transfer).

## Per-Domain Performance on SMD

| Domain | N | ρ Mean (px) | ρ Median | ρ≤10px (%) | θ Mean (°) | θ≤2° (%) | Line Dist Mean |
|--------|---|-------------|----------|------------|------------|----------|----------------|
| NIR | 1006 | 22.86 | 6.32 | 62.9 | 1.026 | 87.8 | 13.36 |
| VIS_Onboard | 998 | 30.35 | 5.93 | 66.6 | 0.451 | 99.5 | 16.74 |
| VIS_Onshore | 994 | 10.66 | 5.22 | 76.2 | 0.505 | 96.2 | 6.06 |

**Domain descriptions**:
- **NIR**: Near-infrared camera footage
- **VIS_Onboard**: Visible spectrum camera mounted on vessel
- **VIS_Onshore**: Visible spectrum camera from shore-based station
