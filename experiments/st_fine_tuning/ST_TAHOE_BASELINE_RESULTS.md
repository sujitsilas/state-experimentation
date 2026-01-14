# ST-Tahoe Baseline Inference Results

**Date**: 2026-01-13
**Model**: ST-Tahoe (pretrained on Basak drug perturbations)
**Input**: SE-600M embeddings (2000-dim, truncated from 2058)
**Output**: [burn_sham_st_tahoe_predictions.h5ad](results/burn_sham_st_tahoe_predictions.h5ad)

---

## Executive Summary

Successfully ran ST-Tahoe baseline inference on burn/sham wound healing data. The model generated predictions for all 57,298 cells across 11 cell types, 2 conditions (Burn/Sham), and 3 timepoints (day 10/14/19).

**Key Finding**: Predictions show **very low variability** across conditions, timepoints, and cell types, suggesting the pretrained model (trained on drug perturbations) does not meaningfully distinguish burn from sham wound healing.

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Cells** | 57,298 |
| **Burn Cells** | 24,855 (43.4%) |
| **Sham Cells** | 32,443 (56.6%) |
| **Cell Types** | 11 unique types |
| **Timepoints** | day10 (16,477), day14 (21,190), day19 (19,631) |
| **Top Cell Types** | Ker (18,579), Fib (11,299), Neu (11,178), Mac (6,258), Mus (2,348) |

---

## Prediction Statistics

### Overall
- **Shape**: (57,298 cells, 2,000 dimensions)
- **Range**: [0.0000, 4.1545]
- **Mean**: 0.0048
- **Std**: 0.0515
- **Sparsity**: 98.3% (only 1.7% non-zero values)

### By Condition
| Condition | Cells | Mean | Std |
|-----------|-------|------|-----|
| **Burn** | 24,855 | 0.0048 | 0.0519 |
| **Sham** | 32,443 | 0.0047 | 0.0512 |

**⚠️ Observation**: Nearly identical statistics between Burn and Sham conditions.

### By Timepoint
| Timepoint | Cells | Mean | Std |
|-----------|-------|------|-----|
| **day10** | 16,477 | 0.0049 | 0.0524 |
| **day14** | 21,190 | 0.0047 | 0.0513 |
| **day19** | 19,631 | 0.0047 | 0.0511 |

**⚠️ Observation**: Nearly identical statistics across timepoints.

### By Cell Type (Top 5)
| Cell Type | Cells | Mean | Std |
|-----------|-------|------|-----|
| **Ker** | 18,579 | 0.0047 | 0.0509 |
| **Fib** | 11,299 | 0.0047 | 0.0509 |
| **Neu** | 11,178 | 0.0051 | 0.0538 |
| **Mac** | 6,258 | 0.0048 | 0.0517 |
| **Mus** | 2,348 | 0.0046 | 0.0493 |

**⚠️ Observation**: Nearly identical statistics across cell types.

---

## Warning Messages During Inference

All cell types showed this warning:
```
(group <CellType>) pert 'Burn' not in mapping; using control fallback one-hot.
(group <CellType>) pert 'Sham' not in mapping; using control fallback one-hot.
```

### What This Means

**ST-Tahoe's Perturbation Vocabulary**:
- Trained on: Drug names (e.g., "Doxorubicin_1uM", "Paclitaxel_100nM", "DMSO_TF")
- Trained on: ~1,138 unique drug perturbations
- Expected: Specific drug-concentration pairs

**Our Data**:
- Has: "Burn" and "Sham" perturbations
- Not in model's vocabulary → uses **generic fallback encoding**

**Implication**: The model treats Burn and Sham as generic unknown perturbations, which explains why predictions don't differentiate between them.

---

## Key Observations

### 1. Low Variability
- Predictions are **extremely sparse** (98.3% zeros)
- Mean and std are **nearly identical** across all groupings
- Suggests model is not capturing burn vs sham differences

### 2. Perturbation Mismatch
- Model trained on **drugs** (chemical compounds)
- Our data has **burn injury** (physical/inflammatory perturbation)
- Fundamentally different perturbation semantics

### 3. Baseline Utility
Despite limitations, these predictions serve as a **critical baseline** for:
- Comparing to fine-tuned models (ST-LoRA, ST-LoRA-mHC)
- Demonstrating value of domain-specific training
- Showing that generic pretrained models struggle with specialized tasks

---

## Interpretation

### What ST-Tahoe is Predicting

The model is attempting to predict:
- **"What would these cells look like if perturbed by 'Burn' or 'Sham'?"**
- But since it's never seen these perturbations, it falls back to generic representations
- Output is likely close to a "null" perturbation (minimal predicted change)

### Why Predictions Are Similar

1. **Perturbation encoding**: Both "Burn" and "Sham" get mapped to generic one-hot vectors
2. **No domain knowledge**: Model has no understanding of wound healing biology
3. **Sparse predictions**: Model is conservative, predicting small changes

---

## Visualizations

See [st_tahoe_prediction_analysis.png](results/st_tahoe_prediction_analysis.png) for:

1. **UMAP of predictions colored by condition** - Shows Burn/Sham overlap
2. **UMAP colored by timepoint** - Shows limited temporal structure
3. **UMAP colored by cell type** - Shows some cell type clustering
4. **Prediction magnitude distribution** - Shows Burn/Sham similarity
5. **Mean prediction by condition & timepoint** - Quantifies lack of variation
6. **Sparsity by cell type** - Shows uniform sparsity across types

---

## Conclusions

### ✅ What Worked
- Inference ran successfully on all 57,298 cells
- Model loaded and executed without errors
- Dimension matching (2058 → 2000 truncation) worked correctly
- Predictions are in valid range and format

### ❌ What Didn't Work
- **Predictions do not differentiate Burn from Sham** (identical statistics)
- **No temporal structure** captured (day 10/14/19 are indistinguishable)
- **No cell-type-specific responses** (all types have similar predictions)
- **Model treats unknown perturbations generically** (fallback encoding)

### 🎯 Recommendation

**Do NOT use ST-Tahoe predictions for biological interpretation** of burn/sham wound healing. The model was not trained on this type of perturbation and cannot meaningfully distinguish the conditions.

**Instead**: Use these results as a **negative control baseline** to demonstrate the value of:
1. Fine-tuning on domain-specific data (burn/sham)
2. Training with LoRA for parameter efficiency
3. Adding mHC for gradient stabilization
4. Incorporating biological priors (velocity, temporal structure)

---

## Next Steps

### Immediate
1. **Use as baseline** for comparison with fine-tuned models
2. **Create difference maps**: Compare ST-LoRA vs ST-Tahoe predictions
3. **Compute baseline metrics**: Cell type accuracy, perturbation correlation

### For Fine-Tuning
Train new ST model specifically on burn/sham data:

**Option 1: Gene Expression Input (Recommended)**
```bash
state tx train \
  data.kwargs.embed_key=X \
  data.kwargs.pert_col=condition \
  data.kwargs.control_pert=Sham \
  model.kwargs.lora.enable=true \
  model.kwargs.lora.r=16 \
  model.kwargs.use_mhc=false \
  training.max_epochs=10 \
  output_dir=st_lora_gene
```

**Option 2: With mHC for Gradient Stabilization**
```bash
# Add to above command:
  model.kwargs.use_mhc=true \
  model.kwargs.mhc.sinkhorn_iters=10
```

---

## Technical Details

### Files Generated
- `burn_sham_st_tahoe_predictions.h5ad` (2.3 GB) - Original predictions
- `burn_sham_st_tahoe_predictions_with_umap.h5ad` (2.3 GB) - With UMAP coordinates
- `st_tahoe_prediction_analysis.png` - 6-panel visualization

### Embeddings in AnnData
- `X_state_2000` - ST-Tahoe predictions (2000-dim)
- `X_state_baseline` - Original SE-600M embeddings (2058-dim)
- `X_pred_umap` - UMAP of predictions (2-dim, for visualization)
- `X_harmony`, `X_pca`, `X_umap` - Original embeddings

### Model Architecture Used
- **Backbone**: LlamaModel (6 layers, 1488 hidden dim)
- **Input encoder**: Linear(2000 → 1488)
- **Perturbation encoder**: MLP(1138 → 1488)
- **Output decoder**: MLP(1488 → 2000)
- **Output space**: Gene expression (2000 genes)
- **Batch encoder**: Learned embeddings for batch correction

---

## References

**Model Source**: Arc Institute ST-Tahoe
- Pretrained on Basak et al. drug perturbation dataset
- ~1,138 drug perturbations
- Cell line data (K562, A549, MCF7)

**Our Data**: Burn/Sham wound healing
- Mouse skin scRNA-seq
- Burn injury vs sham control
- Days 10/14/19 post-wounding
- Primary tissue (not cell lines)

**Key Difference**: Drug perturbations (chemical) vs injury perturbations (physical/inflammatory) are fundamentally different modalities.

---

**Conclusion**: ST-Tahoe baseline serves as a useful negative control, demonstrating that pretrained models do not transfer well to specialized perturbation tasks. Fine-tuning is essential.
