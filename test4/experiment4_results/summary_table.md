# Experiment 4: Cross-Dataset Generalization Results

## Overall Performance Comparison

| Dataset | N | ρ Mean (px) | ρ Median | ρ P95 | ρ≤10px (%) | ρ≤20px (%) | θ Mean (°) | θ Median | θ≤1° (%) | θ≤2° (%) |
|---------|---|-------------|----------|-------|------------|------------|------------|----------|----------|----------|
| SMD (Zero-shot) | 2996 | 20.65 | 5.51 | 117.80 | 69.9 | 82.3 | 0.685 | 0.277 | 88.7 | 94.0 |
| MU-SID (In-domain) | 268 | 2.62 | 1.99 | 7.58 | 98.9 | 99.6 | 0.214 | 0.115 | 97.0 | 99.3 |

**Note**: SMD evaluation uses MU-SID trained weights without any fine-tuning (zero-shot transfer).

## Per-Domain Performance on SMD

| Domain | N | ρ Mean (px) | ρ Median | ρ≤10px (%) | θ Mean (°) | θ≤2° (%) | Line Dist Mean |
|--------|---|-------------|----------|------------|------------|----------|----------------|
| NIR | 1006 | 23.01 | 6.34 | 63.0 | 1.032 | 87.8 | 13.43 |
| VIS_Onboard | 996 | 28.39 | 5.41 | 70.8 | 0.511 | 98.2 | 15.56 |
| VIS_Onshore | 994 | 10.52 | 5.18 | 76.0 | 0.508 | 96.2 | 5.97 |

**Domain descriptions**:
- **NIR**: Near-infrared camera footage
- **VIS_Onboard**: Visible spectrum camera mounted on vessel
- **VIS_Onshore**: Visible spectrum camera from shore-based station
