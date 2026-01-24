# Experiment 4: Cross-Dataset Generalization on SMD

This directory contains all code for **Experiment 4** of the thesis: evaluating the cross-dataset generalization capability of the Fusion-CNN horizon detection method on the Singapore Maritime Dataset (SMD).

## 🎯 Experiment Goal

Verify that the model trained on MU-SID can generalize to a completely different maritime dataset (SMD) **without any fine-tuning** (zero-shot transfer).

## 📁 Directory Structure

```
test4/
├── run_experiment4.py              # 🚀 Master script (run this)
├── prepare_smd_testset.py          # Step 1: Extract frames from SMD videos
├── make_fusion_cache_smd.py        # Step 2: Generate fusion features
├── evaluate_fusion_cnn_smd.py      # Step 3: Run evaluation
├── summarize_smd_results.py        # Step 4: Generate thesis tables
├── visualize_smd_predictions.py    # Step 5: Generate visualizations
├── README.md                       # This file
│
├── SMD_GroundTruth.csv             # Generated: GT annotations
├── smd_frames/                     # Generated: Extracted video frames
├── splits/                         # Generated: Train/val/test indices
├── FusionCache_SMD_1024x576/       # Generated: Fusion features cache
├── eval_smd_test_per_sample.csv    # Generated: Per-sample results
├── experiment4_results/            # Generated: Summary tables
└── visualization/                  # Generated: Visualization images
```

## 🚀 Quick Start

### Option 1: Run Everything (Recommended)

```bash
# From project root
python test4/run_experiment4.py
```

### Option 2: Run Step by Step

```bash
# Step 1: Prepare SMD test set (extract frames from videos)
python test4/prepare_smd_testset.py

# Step 2: Generate fusion cache (requires GPU, ~30-60 min)
python test4/make_fusion_cache_smd.py

# Step 3: Evaluate Fusion-CNN
python test4/evaluate_fusion_cnn_smd.py

# Step 4: Generate summary tables
python test4/summarize_smd_results.py

# Step 5: Generate visualizations
python test4/visualize_smd_predictions.py --mode random --n_samples 20
python test4/visualize_smd_predictions.py --mode worst --n_samples 10
```

## 📊 Expected Output

### Console Output (Example)
```
========== SMD Evaluation (Fusion-CNN) ==========
Split: test | N=2997
Weights: weights/best_fusion_cnn_1024x576.pth

[Overall]
Rho abs error (px, original ~1920x1080): mean=X.XX, median=X.XX, p95=XX.XX
Theta error (deg, wrap-aware): mean=X.XXX, median=X.XXX, p95=X.XXX

[Per-domain breakdown]
--- NIR | N=XXX ---
--- VIS_Onboard | N=XXX ---
--- VIS_Onshore | N=XXX ---
```

### Generated Files

1. **experiment4_results/summary_table.md** - Markdown table for quick viewing
2. **experiment4_results/summary_table.tex** - LaTeX table for thesis
3. **visualization/random/** - Random sample visualizations
4. **visualization/worst/** - Worst-case visualizations (for failure analysis)

## 📋 Prerequisites

1. **SMD Dataset** must be available at:
   ```
   Singapore Maritime Dataset/
   ├── NIR/NIR/
   │   ├── Videos/*.avi
   │   └── HorizonGT/*_HorizonGT.mat
   ├── VIS_Onboard/VIS_Onboard/
   │   ├── Videos/*.avi
   │   └── HorizonGT/*_HorizonGT.mat
   └── VIS_Onshore/VIS_Onshore/
       ├── Videos/*.avi
       └── HorizonGT/*_HorizonGT.mat
   ```

2. **Pre-trained weights** in `weights/` directory:
   - `best_fusion_cnn_1024x576.pth` (Fusion CNN)
   - `rghnet_best_c2.pth` (UNet for feature extraction)
   - `Epoch99.pth` (Zero-DCE++ for image enhancement)

## 🔬 SMD Dataset Domains

| Domain | Description | Camera Type |
|--------|-------------|-------------|
| NIR | Near-infrared footage | NIR camera |
| VIS_Onboard | Visible spectrum from vessel | Onboard camera |
| VIS_Onshore | Visible spectrum from shore | Shore-based camera |

## 📝 For Thesis

Use the generated tables in **Section 4.5: Cross-Dataset Generalization**:

```latex
% Copy from experiment4_results/summary_table.tex
\input{experiment4_results/summary_table.tex}
```

Key points to discuss:
1. Zero-shot generalization performance
2. Per-domain performance differences (NIR vs VIS)
3. Comparison with in-domain (MU-SID) performance
4. Failure case analysis from worst visualizations

## ⚠️ Notes

- Cache generation (`make_fusion_cache_smd.py`) requires GPU and takes ~30-60 minutes
- If cache already exists, use `--skip_cache` to save time
- Visualization requires the `smd_frames/` directory to exist
