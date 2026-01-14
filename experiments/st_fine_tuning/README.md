# ST Fine-Tuning Experiments

This directory contains all ST (State Transition) model fine-tuning experiments for burn/sham wound healing.

## Directory Structure

```
experiments/st_fine_tuning/
├── README.md                          # This file
├── train_all_st_variants.ipynb       # Notebook to train all 4 variants
├── compare_st_results.ipynb          # Notebook to compare results
├── configs/
│   ├── burn_sham_st_training.toml    # Data configuration
│   └── burn_sham_gene.toml           # Gene expression config (optional)
├── train_scripts/                     # Auto-generated training scripts
│   ├── train_st_lora.sh
│   ├── train_st_lora_mhc.sh
│   ├── train_st_lora_velocity.sh
│   └── train_st_lora_mhc_velocity.sh
├── logs/                              # Training logs
│   └── *.log
└── results/                           # Results and visualizations
    ├── burn_sham_st_tahoe_predictions.h5ad
    ├── st_tahoe_prediction_analysis.png
    ├── loss_curves_comparison.png
    ├── umap_comparison_all_models.png
    └── *.csv                          # Metrics tables
```

## Model Variants

We train and compare 5 ST model variants:

| Model | LoRA | mHC | Velocity | Training Time | Status |
|-------|------|-----|----------|---------------|--------|
| **ST-Tahoe (Baseline)** | ❌ | ❌ | ❌ | - | ✅ Pretrained |
| **ST-LoRA** | ✅ | ❌ | ❌ | 2-3h | 🔄 Training |
| **ST-LoRA-mHC** | ✅ | ✅ | ❌ | 3-4h | ⏳ Pending |
| **ST-LoRA-Velocity** | ✅ | ❌ | ✅ | 3-4h | ⏳ Pending |
| **ST-LoRA-mHC-Velocity** | ✅ | ✅ | ✅ | 4-5h | ⏳ Pending |

### Model Features

- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning (~1.7% trainable params)
- **mHC (Manifold-Constrained Hyper-Connections)**: Gradient stabilization via Sinkhorn-Knopp
- **Velocity Alignment**: Physics-informed loss using RNA velocity data

## Quick Start

### 1. Train All Models

```bash
# Open training notebook
jupyter notebook train_all_st_variants.ipynb

# Or run training scripts directly
bash train_scripts/train_st_lora.sh 2>&1 | tee logs/st_lora_training.log
bash train_scripts/train_st_lora_mhc.sh 2>&1 | tee logs/st_lora_mhc_training.log
bash train_scripts/train_st_lora_velocity.sh 2>&1 | tee logs/st_lora_velocity_training.log
bash train_scripts/train_st_lora_mhc_velocity.sh 2>&1 | tee logs/st_lora_mhc_velocity_training.log
```

### 2. Monitor Training

```bash
# Check training logs
tail -f logs/st_lora_training.log

# TensorBoard (for each model)
tensorboard --logdir=/home/scumpia-mrl/state_models/st_lora
tensorboard --logdir=/home/scumpia-mrl/state_models/st_lora_mhc
tensorboard --logdir=/home/scumpia-mrl/state_models/st_lora_velocity
tensorboard --logdir=/home/scumpia-mrl/state_models/st_lora_mhc_velocity
```

### 3. Compare Results

```bash
# Open comparison notebook (after training completes)
jupyter notebook compare_st_results.ipynb
```

## Training Configuration

### Common Parameters

```yaml
data:
  embed_key: X_state_2000              # SE-600M embeddings (2000-dim)
  pert_col: condition                  # Perturbation column (Burn/Sham)
  control_pert: Sham                   # Control condition
  cell_type_key: cell_types_simple_short
  batch_col: mouse_id

model:
  lora:
    enable: true
    r: 16                              # LoRA rank
    alpha: 32                          # LoRA alpha
  use_mhc: false/true                  # Enable mHC
  use_velocity_alignment: false/true   # Enable velocity loss

training:
  max_steps: 10000
  lr: 5e-5
  batch_size: 8
  devices: 1                           # Single GPU
  strategy: auto
```

### Variant-Specific Settings

**ST-LoRA**:
- LoRA only, no additional features
- Baseline for comparison

**ST-LoRA-mHC**:
- LoRA + mHC
- `mhc.sinkhorn_iters: 10`
- Expect: smoother loss curves, more stable gradients

**ST-LoRA-Velocity**:
- LoRA + Velocity alignment
- `velocity_lambda: 0.1`, `velocity_beta: 1.0`, `velocity_warmup_steps: 1000`
- Requires: RNA velocity data in AnnData

**ST-LoRA-mHC-Velocity**:
- All features combined
- Best expected performance
- Longest training time

## Expected Results

### Training Dynamics

| Metric | ST-LoRA | ST-LoRA-mHC | ST-LoRA-Velocity | ST-LoRA-mHC-Velocity |
|--------|---------|-------------|------------------|----------------------|
| Loss Stability | Moderate | High ⭐ | Moderate | High ⭐ |
| Gradient Norms | Variable | Stable ⭐ | Variable | Stable ⭐ |
| Convergence | Good | Better ⭐ | Good | Better ⭐ |
| Training Time | 2-3h | 3-4h | 3-4h | 4-5h |

### Prediction Accuracy

Target metrics (to be evaluated after training):

- **Gene correlation**: > 0.6 (predicted vs actual)
- **Cell-wise distance**: < 0.3 (burn pred vs burn actual)
- **Perturbation direction**: cos_sim > 0.7 (sham → burn)
- **Temporal coherence**: Smooth day 10 → 14 → 19 trajectories

## Baseline Results (ST-Tahoe)

ST-Tahoe baseline predictions already computed:

```
✅ ST-Tahoe baseline predictions
   File: results/burn_sham_st_tahoe_predictions.h5ad
   Cells: 57,298

⚠️  IMPORTANT: ST-Tahoe was trained on DRUG perturbations, not burn injury
   - Predictions DO NOT meaningfully differentiate Burn from Sham
   - Burn/Sham statistics are nearly identical (mean ~0.005, std ~0.05)
   - Use as NEGATIVE CONTROL only for comparison

✅ Fine-tuned models (ST-LoRA variants) should significantly outperform baseline
```

See [ST_TAHOE_BASELINE_RESULTS.md](ST_TAHOE_BASELINE_RESULTS.md) for detailed analysis.

## File Descriptions

### Notebooks

**train_all_st_variants.ipynb**
- Comprehensive training notebook
- Generates training scripts for all 4 variants
- Can train sequentially or provide commands for parallel training
- Monitors training status

**compare_st_results.ipynb**
- Loads predictions from all models
- Compares training dynamics (loss curves, stability)
- Evaluates prediction accuracy (correlations, directions)
- Generates visualizations (UMAPs, heatmaps)
- Exports metrics to CSV

### Configuration Files

**burn_sham_st_training.toml**
```toml
[datasets]
burn_sham = "/path/to/burn_sham_baseline_embedded_2000.h5ad"

[training]
burn_sham = "train"
```

**burn_sham_gene.toml** (optional)
- Alternative config using gene expression (X) instead of embeddings
- More interpretable but higher dimensional (3000 genes vs 2000 dims)

### Training Scripts

Auto-generated bash scripts in `train_scripts/`:
- One script per model variant
- Copy-paste ready for execution
- Include all necessary parameters
- Log output to `logs/`

## Troubleshooting

### Training Fails to Start

**Problem**: Hydra config errors
**Solution**: Ensure all parameters use correct names and `+` prefix for new keys

**Problem**: NCCL timeout (multi-GPU)
**Solution**: Use single GPU (`devices=1`) with `strategy=auto`

**Problem**: Out of memory
**Solution**: Reduce `batch_size` (try 4 or 2)

### Training Instability

**Problem**: Loss spikes, gradient explosions
**Solution**: Use mHC variants (ST-LoRA-mHC or ST-LoRA-mHC-Velocity)

**Problem**: Slow convergence
**Solution**:
- Check learning rate (try 1e-4 instead of 5e-5)
- Increase warmup steps
- Monitor TensorBoard for issues

### Poor Predictions

**Problem**: Low correlation with ground truth
**Solution**:
- Verify data preprocessing (correct embeddings used)
- Check if model trained long enough (full 10k steps)
- Try different model variant (mHC or velocity)

## Next Steps

After training and evaluation:

1. **Document findings** in paper/presentation
2. **Biological validation**:
   - Macrophage polarization analysis
   - Temporal trajectory quality
   - Known wound healing markers
3. **Method comparison**: Compare with scVI, CPA, etc.
4. **Extend to other datasets**: Apply to different perturbation experiments

## References

### Documentation

- [IMPLEMENTATION_SUMMARY.md](../../IMPLEMENTATION_SUMMARY.md) - mHC and velocity loss implementation
- [ST_TRAINING_PLAN.md](../../ST_TRAINING_PLAN.md) - Training strategy overview
- [VELOCITY_INTEGRATION_GUIDE.md](../../VELOCITY_INTEGRATION_GUIDE.md) - Velocity alignment details
- [CLAUDE.md](../../CLAUDE.md) - 3-month project roadmap

### Papers

- **State (Arc Institute)**: [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2)
- **mHC**: [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)
- **LoRA**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **RNA Velocity**: [scVelo paper](https://www.nature.com/articles/s41587-020-0591-3)

---

**Last Updated**: 2026-01-13
**Status**: ST-LoRA training in progress, other variants pending
**Author**: Sujit (with Claude assistance)
