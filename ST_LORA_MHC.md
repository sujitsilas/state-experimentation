# State Transition Model Fine-Tuning with LoRA and mHC

**Date**: 2026-01-12
**Purpose**: Fine-tune State Transition (ST) model for burn/sham wound healing perturbation prediction
**Innovation**: Combine LoRA parameter efficiency with mHC gradient stabilization

---

## Overview

This project fine-tunes the pretrained **ST-Tahoe** model from Arc Institute using:

1. **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning (~1-5% trainable parameters)
2. **mHC (Manifold-Constrained Hyper-Connections)**: Gradient stabilization for optimal transport loss

The goal is to predict cellular state transitions from sham (control) to burn (perturbation) conditions in wound healing across multiple timepoints.

---

## Motivation

### Why Fine-Tune ST-Tahoe?

The pretrained ST-Tahoe model is trained on diverse perturbation datasets but not specifically on wound healing biology. Fine-tuning allows us to:

- Adapt the model to wound healing-specific perturbation signatures
- Learn temporal dynamics unique to burn injury response
- Capture cell-type-specific responses to burn vs sham conditions

### Why LoRA?

**Problem**: Full model fine-tuning is computationally expensive and risks catastrophic forgetting.

**Solution**: LoRA adapts only attention layers via low-rank matrices:
- Original: `W_new = W_pretrained` (frozen)
- LoRA: `W_new = W_pretrained + ΔW`, where `ΔW = BA` (B and A are low-rank)
- Only `B` and `A` are trainable (~1-5% of total parameters)

**Benefits**:
- **Memory efficient**: Only store adapter weights
- **Fast training**: Fewer parameters to update
- **Modular**: Can swap adapters for different tasks

### Why mHC?

**Problem**: Optimal transport (OT) loss can have unstable gradients, especially with distributional distances like energy or Sinkhorn loss.

**Solution**: mHC constrains residual connections to doubly stochastic matrices via Sinkhorn-Knopp algorithm:
- Standard residual: `x_out = F(x) + x`
- mHC residual: `x_out = H_res @ x + H_post^T @ F(H_pre @ x)`
- `H_res` is projected to Birkhoff polytope (rows and columns sum to 1)

**Benefits**:
- **Gradient stability**: Prevents loss spikes during training
- **Identity preservation**: Maintains information flow through residuals
- **Theoretical guarantee**: Doubly stochastic constraint ensures proper gradient scaling

---

## Architecture

### Model Stack

```
Input: SE-600M Embeddings (2048-dim)
         ↓
    Perturbation Encoder (MLP: 2048 → 512)
         +
    Basal Cell Encoder (MLP: 2048 → 512)
         ↓
    Transformer Backbone (GPT2, 8 layers)
    [LoRA adapters on attention: r=16, α=32]
    [mHC layers wrapping each transformer layer]
         ↓
    Projection Head (MLP: 512 → 2048)
         ↓
Output: Predicted Perturbed State (2048-dim)
```

### LoRA Configuration

```yaml
lora:
  enable: true
  r: 16                    # Rank of low-rank matrices
  alpha: 32                # Scaling factor (α/r = 2.0)
  dropout: 0.05
  target_modules:
    - c_attn               # Query, Key, Value projections
    - c_proj               # Output projection
```

**Trainable Parameters**:
- Without LoRA: ~50M parameters
- With LoRA: ~1-2M parameters (96-98% reduction)

### mHC Configuration

```yaml
mhc:
  enable: true
  sinkhorn_iters: 10       # Sinkhorn-Knopp iterations
  layer_indices: null      # Apply to all transformer layers
```

**Components**:
1. **SinkhornKnopp Module**: Projects logits to doubly stochastic matrix
2. **mHCTransformerLayer**: Wraps each transformer layer with learned mixing matrices
   - `H_res`: Residual mixing (doubly stochastic)
   - `H_pre`: Input mixing (non-negative)
   - `H_post`: Output mixing (non-negative)

---

## Training Strategy

### Data

- **Input**: SE-600M baseline embeddings (`X_state`) from [baseline_analysis](experiments/baseline_analysis/)
- **Perturbation**: Condition (burn vs sham)
- **Control**: Sham (healthy wound)
- **Cell Types**: 10+ types (macrophages, fibroblasts, endothelial, etc.)
- **Timepoints**: Day 10, 14, 19 post-wounding

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 5 | Fine-tuning requires fewer epochs |
| Learning Rate | 5e-5 | Lower than pretraining (1e-4) |
| Batch Size | 16 | Fits in 2× RTX 5000 Ada memory |
| Cell Set Length | 256 | Number of cells per batch |
| Optimizer | AdamW | Standard for transformers |
| Weight Decay | 0.01 | Regularization |
| Warmup Steps | 100 | Gradual LR ramp-up |
| Gradient Clip | 1.0 | Prevent exploding gradients |

### Loss Function

**Optimal Transport Loss** (Energy distance from `geomloss`):
```python
loss = SamplesLoss(loss="energy", blur=0.05)
```

This loss measures distributional distance between predicted and actual perturbed cell sets, capturing both:
- Individual cell-level predictions
- Population-level distributional shifts

---

## Experimental Comparison

We compare **3 model variants**:

### 1. ST-Tahoe (Baseline)

- **Pretrained** model from Arc Institute
- **No fine-tuning** (zero-shot)
- Serves as baseline for comparison

### 2. ST-LoRA

- Fine-tuned with **LoRA adapters only**
- No mHC (standard residual connections)
- Tests parameter efficiency alone

### 3. ST-LoRA-mHC

- Fine-tuned with **LoRA + mHC**
- mHC applied to all transformer layers
- Tests combined approach

---

## Evaluation Metrics

### 1. Perturbation Prediction Accuracy

- **Nearest Neighbor Distance**: Distance between predicted and actual perturbed cells
  - Target: < 0.3 (minimum), < 0.2 (strong)
- **Gene Correlation**: Correlation between predicted and actual gene expression changes
  - Target: > 0.6 (good), > 0.7 (strong)
- **Cell-Type-Specific Accuracy**: Performance broken down by cell type
  - Important: Macrophages and fibroblasts are key wound healing cells

### 2. Training Stability

- **Loss Curves**: Smoothness and convergence speed
  - Expectation: mHC shows smoother curves
- **Gradient Norms**: Stability throughout training
  - Expectation: mHC prevents gradient spikes
- **Validation Loss**: Generalization quality
  - Expectation: Both LoRA variants generalize well

### 3. Efficiency

- **Number of Trainable Parameters**: % of total model
  - Expectation: LoRA ~1-5%
- **Training Time**: Hours per epoch
  - Expectation: LoRA-mHC slightly slower (Sinkhorn overhead)
- **Memory Usage**: GPU memory during training
  - Expectation: Both LoRA variants similar

### 4. Biological Interpretability

- **Temporal Coherence**: Smooth transitions across day 10 → 14 → 19
- **Wound Healing Trajectory**: Matches known biology
  - Early: Inflammation (macrophages, neutrophils)
  - Mid: Proliferation (fibroblasts, endothelial)
  - Late: Remodeling (maturation)
- **Cell-Type-Specific Signatures**: Captures known perturbation responses
  - Burn: Prolonged inflammation, delayed resolution
  - Sham: Normal wound healing progression

---

## Implementation Details

### File Structure

```
experiments/st_fine_tuning/
├── st_lora_mhc_experiment.ipynb   # Main experiment notebook
├── configs/
│   ├── lora_config.yaml           # LoRA-only config
│   └── lora_mhc_config.yaml       # LoRA + mHC config
├── data/                          # (gitignored)
│   └── burn_sham_baseline_embedded.h5ad
└── results/
    ├── st_lora/                   # LoRA model outputs
    └── st_lora_mhc/              # LoRA-mHC model outputs
```

### Code Modules

| File | Purpose |
|------|---------|
| `src/state/tx/models/state_transition.py` | Main ST model class |
| `src/state/tx/models/mhc.py` | mHC implementation (NEW) |
| `src/state/tx/models/utils.py` | LoRA utilities (existing) |

### Training Commands

**ST-LoRA**:
```bash
state tx train \
  data.kwargs.embed_key=X_state \
  data.kwargs.pert_col=condition \
  data.kwargs.control_pert=sham \
  model.kwargs.lora.enable=true \
  model.kwargs.lora.r=16 \
  model.kwargs.use_mhc=false \
  training.max_epochs=5 \
  output_dir=/home/scumpia-mrl/state_models/st_lora
```

**ST-LoRA-mHC**:
```bash
state tx train \
  data.kwargs.embed_key=X_state \
  data.kwargs.pert_col=condition \
  data.kwargs.control_pert=sham \
  model.kwargs.lora.enable=true \
  model.kwargs.lora.r=16 \
  model.kwargs.use_mhc=true \
  model.kwargs.mhc.sinkhorn_iters=10 \
  training.max_epochs=5 \
  output_dir=/home/scumpia-mrl/state_models/st_lora_mhc
```

---

## Expected Results

### Training Dynamics

| Aspect | ST-LoRA | ST-LoRA-mHC |
|--------|---------|-------------|
| **Loss Stability** | Moderate (some spikes) | High (smooth curves) |
| **Convergence Speed** | Fast | Fast |
| **Gradient Norms** | Variable | Stable |
| **Training Time** | 2-3 hours | 3-4 hours |

### Performance

| Metric | ST-Tahoe (Baseline) | ST-LoRA | ST-LoRA-mHC |
|--------|-------------------|---------|-------------|
| **NN Distance** | ~0.35 | < 0.25 | < 0.20 |
| **Gene Correlation** | ~0.55 | > 0.65 | > 0.70 |
| **Trainable Params** | 0% | ~2% | ~2% |
| **Cell-Type Accuracy** | Moderate | Good | Strong |

### Key Hypotheses

1. **LoRA improves over baseline**: Fine-tuning adapts to wound healing biology
2. **mHC stabilizes training**: Smoother loss curves, more consistent gradients
3. **mHC improves final performance**: Stable training leads to better convergence
4. **Both are parameter-efficient**: 95%+ of model remains frozen

---

## mHC Algorithm Details

### Sinkhorn-Knopp Algorithm

**Goal**: Project a matrix `M` onto the Birkhoff polytope (doubly stochastic matrices).

**Algorithm**:
```python
def sinkhorn_knopp(logits, n_iters=10, eps=1e-8):
    # Convert logits to non-negative
    mat = exp(logits - logits.max())

    # Iterative normalization
    for i in range(n_iters):
        # Normalize rows
        mat = mat / (mat.sum(dim=1, keepdim=True) + eps)
        # Normalize columns
        mat = mat / (mat.sum(dim=0, keepdim=True) + eps)

    return mat  # Doubly stochastic
```

**Properties**:
- Rows sum to 1 (proper distribution)
- Columns sum to 1 (preserves total mass)
- Differentiable (gradients flow through iterations)
- Converges in ~10 iterations

### mHC Layer Forward Pass

```python
def mhc_forward(x, base_layer, H_res, H_pre, H_post):
    # Residual path: H_res @ x
    residual = x @ H_res.T

    # Function path: H_post.T @ F(H_pre @ x)
    x_pre = x @ H_pre.T
    transformed = base_layer(x_pre)
    function_path = transformed @ H_post

    # Combine
    output = residual + function_path
    return output
```

**Why it works**:
- `H_res` is doubly stochastic → preserves identity mapping
- `H_pre` and `H_post` are non-negative → stable transformations
- Residual + function path → gradient flow is well-behaved

---

## References

### Papers

1. **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. **mHC**: [Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
3. **State (Arc Institute)**: [Predicting cellular responses to perturbation](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2)
4. **Optimal Transport**: [Computational Optimal Transport](https://arxiv.org/abs/1803.00567)

### Code Repositories

- **Arc State**: https://github.com/ArcInstitute/state
- **mHC Reference**: https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections
- **PEFT (LoRA)**: https://github.com/huggingface/peft

### Models

- **ST-Tahoe**: https://huggingface.co/arcinstitute/ST-Tahoe
- **SE-600M**: Arc Institute's State Embedding model

---

## Next Steps

### Immediate (Phase 2)

1. ✅ Implement mHC module
2. ✅ Integrate mHC into ST model
3. ✅ Create training configurations
4. ✅ Set up experiment notebook
5. 🔄 **Train ST-LoRA** (2-3 hours)
6. 🔄 **Train ST-LoRA-mHC** (3-4 hours)
7. 🔄 Evaluate and compare results
8. 🔄 Document findings

### Future Directions

If Phase 2 is successful:
- **Phase 3**: Add timepoint embeddings to ST model
- **Phase 4**: Integrate RNA velocity features
- **Extension**: Apply to other perturbation datasets
- **Biological Validation**: Compare predictions to experimental data

---

## Contact

**Project Owner**: Sujit
**Affiliation**: Philip Scumpia Lab
**Date**: January 2026
**Purpose**: PhD Qualifying Exam

---

**For implementation details, see**:
- Experiment notebook: [experiments/st_fine_tuning/st_lora_mhc_experiment.ipynb](experiments/st_fine_tuning/st_lora_mhc_experiment.ipynb)
- mHC implementation: [src/state/tx/models/mhc.py](src/state/tx/models/mhc.py)
- ST model integration: [src/state/tx/models/state_transition.py](src/state/tx/models/state_transition.py)
