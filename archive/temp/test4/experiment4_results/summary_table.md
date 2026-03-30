# Experiment 4: Cross-Dataset Generalization Results

## Overall Performance Comparison

| Dataset | N | ρ Mean (px) | ρ Median | ρ P95 | ρ≤10px (%) | ρ≤20px (%) | θ Mean (°) | θ Median | θ≤1° (%) | θ≤2° (%) |
|---------|---|-------------|----------|-------|------------|------------|------------|----------|----------|----------|
| SMD (Zero-shot) | 2627 | 12.48 | 5.10 | 50.56 | 75.4 | 88.5 | 0.458 | 0.267 | 92.4 | 97.1 |
| MU-SID (In-domain) | 268 | 2.62 | 1.99 | 7.58 | 98.9 | 99.6 | 0.214 | 0.115 | 97.0 | 99.3 |

**Note**: SMD evaluation uses MU-SID trained weights without any fine-tuning (zero-shot transfer).

## Per-Domain Performance on SMD

| Domain | N | ρ Mean (px) | ρ Median | ρ≤10px (%) | θ Mean (°) | θ≤2° (%) | Line Dist Mean |
|--------|---|-------------|----------|------------|------------|----------|----------------|
| NIR | 888 | 17.95 | 5.50 | 69.4 | 0.628 | 92.6 | 10.11 |
| VIS_Onboard | 808 | 11.81 | 4.99 | 77.8 | 0.414 | 100.0 | 6.81 |
| VIS_Onshore | 931 | 7.85 | 4.93 | 79.1 | 0.334 | 98.9 | 4.48 |

**Domain descriptions**:
- **NIR**: Near-infrared camera footage
- **VIS_Onboard**: Visible spectrum camera mounted on vessel
- **VIS_Onshore**: Visible spectrum camera from shore-based station
