# ST Model Training Plan for Burn/Sham Wound Healing

**Date**: 2026-01-13
**Status**: Ready to train
**Author**: Claude (with Sujit)

---

## Summary

Successfully implemented **mHC** and **velocity loss** modules. Discovered that **ST-Tahoe cannot be used directly** due to incompatibility with burn/sham data.

**Solution**: Train ST model from scratch with LoRA on burn/sham wound healing data.

---

## Key Findings

### ❌ ST-Tahoe Incompatibility

**ST-Tahoe Model** (pretrained):
- Dataset: Basak drug perturbation (cell lines)
- Input: 2000 highly variable genes
- Perturbations: Drug treatments (DMSO control)
- Task: Drug response prediction

**Our Burn/Sham Data**:
- Dataset: Wound healing tissue scRNA-seq
- Input: 3000 genes (or 2058-dim SE embeddings)
- Perturbations: Burn vs Sham injury
- Task: Wound healing trajectory prediction

**Why incompatible**:
1. Different input dimensions (2000 vs 3000/2058)
2. Different perturbation semantics (drugs vs burn injury)
3. Different biological context (cell lines vs tissue)

**Error encountered**:
```
RuntimeError: shape '[1, -1, 2000]' is invalid for input of size 183162
```

This occurs because ST-Tahoe expects 2000-dim inputs but our data has 3000 genes or 2058-dim embeddings.

---

## ✅ Solution: Train from Scratch with LoRA

Instead of using pretrained ST-Tahoe, we'll train a new ST model specifically for burn/sham data using LoRA for efficiency.

### Two Training Options

#### **Option 1: Gene Expression Input (RECOMMENDED)**

**Pros**:
- More interpretable (gene-level predictions)
- Standard ST approach
- Can identify specific genes affected by burn

**Cons**:
- Higher dimensional (3000 genes)
- Slightly more parameters to train

**Command**:
```bash
state tx train \
  data.kwargs.toml_config_path=experiments/st_fine_tuning/configs/burn_sham_gene.toml \
  data.kwargs.embed_key=X \
  data.kwargs.pert_col=condition \
  data.kwargs.control_pert=Sham \
  data.kwargs.cell_type_key=cell_types_simple_short \
  data.kwargs.batch_col=mouse_id \
  model.kwargs.lora.enable=true \
  model.kwargs.lora.r=16 \
  model.kwargs.lora.alpha=32 \
  model.kwargs.use_mhc=false \
  training.max_epochs=10 \
  training.learning_rate=1e-4 \
  training.batch_size=16 \
  model.kwargs.hidden_dim=512 \
  model.kwargs.cell_set_len=128 \
  output_dir=/home/scumpia-mrl/state_models/st_lora_gene \
  name=st_lora_burn_sham_gene
```

#### **Option 2: SE-600M Embeddings Input**

**Pros**:
- Lower dimensional (2058 vs 3000)
- Leverages pretrained SE-600M knowledge
- Faster training

**Cons**:
- Less interpretable (embedding space)
- Can't directly identify affected genes

**Command**:
```bash
state tx train \
  data.kwargs.toml_config_path=experiments/st_fine_tuning/configs/burn_sham_emb.toml \
  data.kwargs.embed_key=X_state_baseline \
  data.kwargs.pert_col=condition \
  data.kwargs.control_pert=Sham \
  data.kwargs.cell_type_key=cell_types_simple_short \
  data.kwargs.batch_col=mouse_id \
  model.kwargs.lora.enable=true \
  model.kwargs.lora.r=16 \
  model.kwargs.lora.alpha=32 \
  model.kwargs.use_mhc=false \
  training.max_epochs=10 \
  training.learning_rate=1e-4 \
  training.batch_size=16 \
  model.kwargs.hidden_dim=512 \
  model.kwargs.cell_set_len=128 \
  output_dir=/home/scumpia-mrl/state_models/st_lora_emb \
  name=st_lora_burn_sham_emb
```

---

## Training Strategy

### Phase 1: ST-LoRA (LoRA only)

Train baseline model with LoRA adapters but no mHC.

**Expected time**: 2-4 hours on 2x RTX 5000 Ada
**Trainable parameters**: ~1-2% of total model

**Goals**:
- Establish baseline performance
- Verify training stability without mHC
- Identify loss curve characteristics

### Phase 2: ST-LoRA-mHC (LoRA + mHC)

Add mHC to stabilize optimal transport loss gradients.

**Expected time**: 3-5 hours on 2x RTX 5000 Ada
**Trainable parameters**: ~1-2% + mHC mixing matrices

**Command (add to Phase 1)**:
```bash
model.kwargs.use_mhc=true \
model.kwargs.mhc.sinkhorn_iters=10
```

**Goals**:
- Compare training stability vs LoRA-only
- Measure impact on convergence speed
- Evaluate final performance improvement

### Phase 3: Comparison and Analysis

Compare three variants:
1. **Random initialization** (no pretraining)
2. **ST-LoRA** (LoRA adapters)
3. **ST-LoRA-mHC** (LoRA + mHC)

**Metrics**:
- Training loss curves (smoothness)
- Gradient norm stability
- Perturbation prediction accuracy
- Cell-type-specific performance
- Biological interpretability

---

## Data Configuration

Created config file: `experiments/st_fine_tuning/configs/burn_sham_gene.toml`

**Key settings**:
- Input: Gene expression (3000 genes from adata.X)
- Perturbations: Sham (control) vs Burn (treatment)
- Cell types: 11 types (Mac, Fib, EC, Ker, etc.)
- Timepoints: Day 10, 14, 19 post-wounding
- Batch: Mouse ID
- Split: 80/20 train/val, stratified by cell type

---

## Expected Outcomes

### Training Dynamics

| Aspect | ST-LoRA | ST-LoRA-mHC |
|--------|---------|-------------|
| **Loss Stability** | Moderate (some OT spikes) | High (smooth curves) |
| **Convergence** | Good | Better |
| **Gradient Norms** | Variable | Stable |
| **Training Time** | 2-4 hours | 3-5 hours |

### Performance Metrics

Target metrics for successful training:

**Perturbation Prediction**:
- Gene correlation (predicted vs actual): > 0.6
- Cell-wise distance (burn pred vs burn actual): < 0.3
- Cell-type-specific accuracy: > 0.7

**Training Stability**:
- Loss variance (last 10% of training): < 0.1
- Gradient norm spikes: < 5 per epoch
- Validation loss improvement: > 20% vs initial

**Biological Validation**:
- Macrophage polarization trajectory recovered
- Temporal coherence (day 10 → 14 → 19) maintained
- Known wound healing markers upregulated in burn predictions

---

## Implemented Modules

### ✅ mHC Module (`src/state/tx/models/mhc.py`)

- **SinkhornKnopp**: Projects matrices to doubly stochastic
- **mHCTransformerLayer**: Wraps transformer layers with mixing matrices
- **apply_mhc_to_transformer**: Utility for GPT2/LLaMA/BERT

**Usage in config**:
```yaml
model:
  kwargs:
    use_mhc: true
    mhc:
      sinkhorn_iters: 10
      layer_indices: null  # Apply to all layers
```

### ✅ Velocity Loss Module (`src/state/tx/models/velocity_loss.py`)

- **VelocityAlignmentLoss**: Cosine similarity + magnitude matching
- **VelocityRegularizationLoss**: L2-based alternative

**Usage in config** (optional, for future experiments):
```yaml
model:
  kwargs:
    use_velocity_alignment: true
    velocity_lambda: 0.1
    velocity_beta: 1.0
    velocity_warmup_steps: 1000
```

---

## Next Steps

### Immediate Actions

1. **Choose input type**: Gene expression (Option 1) recommended for interpretability
2. **Create output directories**:
   ```bash
   mkdir -p /home/scumpia-mrl/state_models/st_lora_gene
   mkdir -p /home/scumpia-mrl/state_models/st_lora_gene_mhc
   ```
3. **Start Phase 1 training** (ST-LoRA without mHC)
4. **Monitor training** with TensorBoard
5. **Start Phase 2 training** (ST-LoRA-mHC with mHC)
6. **Compare results** and document findings

### Future Directions (Phase 3+)

If Phase 1-2 succeed:
- Add timepoint embeddings for temporal modeling
- Integrate RNA velocity features
- Validate against experimental perturbation data
- Apply to other wound healing datasets

---

## Monitoring Training

### TensorBoard

```bash
# Monitor ST-LoRA training
tensorboard --logdir=/home/scumpia-mrl/state_models/st_lora_gene

# Monitor ST-LoRA-mHC training
tensorboard --logdir=/home/scumpia-mrl/state_models/st_lora_gene_mhc
```

### Key Metrics to Watch

**During training**:
- `train/loss`: Should decrease smoothly
- `train/ot_loss`: Optimal transport loss component
- `train/grad_norm`: Gradient norms (mHC should stabilize)
- `val/loss`: Validation loss (check overfitting)

**Post-training**:
- Perturbation prediction accuracy
- Gene correlation scores
- Cell-type-specific performance
- Temporal trajectory quality

---

## Troubleshooting

### If training fails to converge:

1. **Reduce learning rate**: Try 5e-5 instead of 1e-4
2. **Increase warmup**: Add `training.warmup_steps=500`
3. **Reduce batch size**: Try 8 instead of 16
4. **Check data**: Verify perturbation labels and cell counts

### If memory issues occur:

1. **Reduce cell_set_len**: Try 64 instead of 128
2. **Reduce batch_size**: Try 8 instead of 16
3. **Use single GPU**: Set `training.devices=1`

### If mHC causes issues:

1. **Reduce Sinkhorn iterations**: Try 5 instead of 10
2. **Apply to fewer layers**: Set `layer_indices=[0, 1, 2]`
3. **Disable mHC temporarily**: Set `use_mhc=false`

---

## Success Criteria

### Minimum Viable Result (Pass)

- ✅ Training converges without crashes
- ✅ Validation loss improves > 20%
- ✅ Gene correlation > 0.5
- ✅ Qualitative UMAP shows burn/sham separation

### Strong Result (High Pass)

- ⭐ All of above +
- ⭐ Gene correlation > 0.65
- ⭐ mHC shows smoother training than LoRA-only
- ⭐ Cell-type-specific performance > 0.7
- ⭐ Temporal trajectories biologically plausible

### Excellent Result (Outstanding)

- 🏆 All of above +
- 🏆 Gene correlation > 0.75
- 🏆 Macrophage polarization dynamics recovered
- 🏆 Novel biological insights from predictions
- 🏆 Validation against known wound healing biology

---

## References

**Documentation**:
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Module implementation details
- [ST_LORA_MHC.md](ST_LORA_MHC.md) - Original project specification
- [CLAUDE.md](CLAUDE.md) - 3-month project roadmap

**Papers**:
- mHC: https://arxiv.org/abs/2512.24880
- LoRA: https://arxiv.org/abs/2106.09685
- State (Arc Institute): https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2

---

**Ready to train!** 🚀

All modules implemented, data configured, commands ready. Proceed with Phase 1: ST-LoRA training.
