# LoRA Multi-Covariate Fine-Tuning for SE-600M

## Overview

This implementation provides **parameter-efficient fine-tuning** of the STATE SE-600M embedding model using **LoRA (Low-Rank Adaptation)** adapters with **multi-covariate conditioning**.

### Key Concept

Instead of training a downstream task model (like CPA for perturbation prediction), we directly fine-tune the foundation embedding model itself to be aware of experimental covariates (timepoint, condition, etc.).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SE-600M Base Model (FROZEN)                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Token Encoder: Linear(5120→2048) + LayerNorm + SiLU     │  │
│  │ Transformer: 16 layers, 16 heads, 2048 hidden dim       │  │
│  │ Decoder: SkipBlock + Linear(2048→2048)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         ↓ (frozen weights)                       │
│                   Base Embedding (2048-dim)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                ┌─────────────────────────────┐
                │   LoRA Adapters (TRAINABLE) │
                │   - Applied to Q, V         │
                │   - Rank: 16, Alpha: 32     │
                │   - Dropout: 0.1            │
                └─────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               Covariate Encoder (TRAINABLE)                      │
│  ┌────────────────┐  ┌────────────────┐                        │
│  │  timepoint     │  │  condition     │                        │
│  │ Embedding(3,256)│  │ Embedding(2,256)│                       │
│  └────────┬───────┘  └────────┬───────┘                        │
│           └──────────┬──────────┘                               │
│                      ↓                                           │
│          Concat [z_time | z_cond] → 512-dim                    │
│                      ↓                                           │
│         MLP(512 → 1024 → 512 → 2048) with LayerNorm            │
│                      ↓                                           │
│            Covariate Embedding (2048-dim)                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│          Conditioning Projection (TRAINABLE)                     │
│  Concat(base_emb, cov_emb) → 4096-dim                          │
│                      ↓                                           │
│  Linear(4096→2048) + LayerNorm + SiLU                          │
│                      ↓                                           │
│  L2 Normalize → Conditioned Embedding (2048-dim)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Advantages

### 1. Parameter Efficiency
- **Base Model (Frozen)**: ~600M parameters
- **Trainable Components**: ~1-5M parameters (1-5% of total)
  - LoRA adapters: ~500K parameters
  - Covariate encoder: ~2M parameters
  - Conditioning projection: ~8M parameters

### 2. Generic Multi-Covariate Support
```yaml
covariates:
  - name: timepoint
    type: categorical
    num_categories: 3  # day10, day14, day19
    embed_dim: 256

  - name: condition
    type: categorical
    num_categories: 2  # burn, sham
    embed_dim: 256

  # Optional: continuous covariates
  - name: time_days
    type: continuous
    continuous_encoder:
      hidden_dim: 128
      output_dim: 256
      activation: silu
```

### 3. YAML Configuration
Users can add/remove covariates without code changes:
```yaml
# Easy to extend to new covariates
- name: dosage
  type: continuous

- name: treatment
  type: categorical
  num_categories: 5
```

### 4. Embedding Space Conditioning
- Produces general-purpose embeddings conditioned on covariates
- Can be used for any downstream task (clustering, classification, trajectory analysis)
- Not limited to perturbation prediction

---

## Implementation Files

### Core Module
- **[src/state/emb/nn/lora_covariate_model.py](src/state/emb/nn/lora_covariate_model.py)**
  - `LoRACovariateStateModel`: Main Lightning module
  - `MultiCovariateEncoder`: Generic covariate encoding
  - `ContinuousEncoder`: MLP for continuous covariates
  - `CombinationMLP`: Fuses multiple covariate embeddings

### Configuration
- **[configs/lora_multicov_config.yaml](configs/lora_multicov_config.yaml)**
  - Base checkpoint path
  - LoRA hyperparameters (rank, alpha, target modules)
  - Covariate specifications
  - Training settings (batch size, learning rate, devices)

### Training Script
- **[train_lora_multicov.py](train_lora_multicov.py)**
  - CLI interface for training
  - Data loading and preprocessing
  - Lightning Trainer setup
  - Checkpoint management

### Notebook
- **[phase2_lora_multicov_training.ipynb](phase2_lora_multicov_training.ipynb)**
  - Environment setup and validation
  - Model initialization
  - Training instructions
  - Evaluation and comparison

---

## Usage

### 1. Data Preparation

Ensure your H5AD file has the required metadata columns:
```python
import anndata as ad

adata = ad.read_h5ad("burn_sham_processed.h5ad")

# Required columns in adata.obs:
# - condition: categorical (e.g., 'burn', 'sham')
# - timepoint: categorical (e.g., 'day10', 'day14', 'day19')
# - cell_types_simple_short: cell type labels
# - mouse_id: batch/mouse identifiers
```

### 2. Configure Covariates

Edit `configs/lora_multicov_config.yaml`:
```yaml
covariates:
  covariates:
    - name: condition
      type: categorical
      num_categories: 2
      embed_dim: 256

    - name: timepoint
      type: categorical
      num_categories: 3
      embed_dim: 256
```

### 3. Train

```bash
# Run training
python train_lora_multicov.py --config configs/lora_multicov_config.yaml

# Monitor with TensorBoard
tensorboard --logdir=/home/scumpia-mrl/state_models/burn_sham_lora_multicov
```

**Expected Training Time**: 4-6 hours on 2x RTX 5000 Ada (33GB each)

### 4. Extract Embeddings

After training, extract covariate-conditioned embeddings:
```python
from src.state.emb.nn.lora_covariate_model import LoRACovariateStateModel

# Load trained model
model = LoRACovariateStateModel.load_from_checkpoint("checkpoints/best.ckpt")
model.eval()

# Process cells with covariates
for cell in dataset:
    embedding = model.forward(
        src=cell["sentence"],
        mask=cell["mask"],
        covariate_dict={
            "condition": torch.tensor([0]),  # burn
            "timepoint": torch.tensor([1]),  # day14
        }
    )
```

---

## Comparison: CPA vs. LoRA Approach

| Aspect | CPA (❌ Wrong Approach) | LoRA (✅ Correct Approach) |
|--------|------------------------|---------------------------|
| **Target Model** | Downstream CPA task model | SE-600M foundation model |
| **Training Space** | Perturbation prediction | Embedding space |
| **Parameters Trained** | All CPA params (~20M) | LoRA + covariates (~1-5M) |
| **Training Time** | 4-6 hours | 4-6 hours |
| **Output** | Perturbed gene expression predictions | Covariate-conditioned embeddings |
| **Flexibility** | Specific to perturbation tasks | General-purpose embeddings |
| **Extensibility** | Requires retraining full model | Only retrain LoRA adapters |
| **Use Cases** | Perturbation response prediction | Any downstream task (clustering, DE, trajectories) |

---

## LoRA Technical Details

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:
1. Freezes pretrained model weights
2. Adds small trainable low-rank matrices to specific layers
3. Significantly reduces trainable parameters while maintaining performance

### Mathematical Formulation

For a pretrained weight matrix W₀ ∈ ℝᵈˣᵏ:

**Standard Fine-Tuning**:
```
W = W₀ + ΔW
```
- Requires training all d×k parameters

**LoRA Fine-Tuning**:
```
W = W₀ + BA
where:
  B ∈ ℝᵈˣʳ
  A ∈ ℝʳˣᵏ
  r << min(d, k)
```
- Only trains 2×d×r parameters (much smaller!)
- Example: d=2048, k=2048, r=16
  - Standard: 4.2M parameters
  - LoRA: 65K parameters (98.5% reduction!)

### Applied to Attention

For attention projections (Q, K, V):
```python
# Standard
Q = x @ W_q

# With LoRA
Q = x @ (W_q_frozen + B_q @ A_q)
  = x @ W_q_frozen + (x @ B_q) @ A_q
```

### Hyperparameters

- **Rank (r)**: Low-rank dimension
  - Smaller r → fewer parameters, less expressive
  - Larger r → more parameters, more expressive
  - Typical: r ∈ [4, 16, 32]

- **Alpha (α)**: Scaling factor
  - Controls magnitude of LoRA contribution
  - Typical: α = 2r

- **Target Modules**: Which layers to apply LoRA
  - Attention: q_proj, k_proj, v_proj, out_proj
  - MLP: fc1, fc2

---

## Covariate Conditioning Details

### Categorical Covariates

Encoded using learned embeddings:
```python
timepoint_emb = nn.Embedding(num_timepoints, embed_dim)
z_time = timepoint_emb(timepoint_idx)  # (batch, embed_dim)
```

### Continuous Covariates

Encoded using MLP:
```python
time_encoder = nn.Sequential(
    nn.Linear(1, hidden_dim),
    nn.SiLU(),
    nn.Linear(hidden_dim, output_dim)
)
z_time = time_encoder(time_days)  # (batch, output_dim)
```

### Combination Methods

**1. Concatenation + MLP** (Recommended):
```python
z_combined = torch.cat([z_time, z_cond], dim=1)  # (batch, 512)
z_cov = MLP(z_combined)  # (batch, 2048)
```
- Learns non-linear interactions between covariates
- Most expressive

**2. Simple Addition**:
```python
z_cov = z_time + z_cond  # (batch, 256)
```
- Assumes independence between covariates
- Less expressive but faster

---

## Evaluation Metrics

After training, evaluate with:

### 1. Cell Type Classification (kNN)
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train_celltype)
accuracy = knn.score(X_test, y_test_celltype)
```
**Target**: ≥96.4% (baseline SE-600M performance)

### 2. Batch Correction (Silhouette)
```python
from scib.metrics import silhouette

sil_batch = silhouette(adata, batch_key='mouse_id', group_key='cell_type', embed='X_lora')
```
**Target**: Higher = better batch mixing

### 3. Temporal Coherence
```python
# Distance between consecutive timepoints
from scipy.spatial.distance import pdist

dists_10_14 = compute_distance(embeddings_day10, embeddings_day14)
dists_14_19 = compute_distance(embeddings_day14, embeddings_day19)
```
**Target**: Smooth transitions across time

### 4. Covariate Disentanglement
```python
# Can we predict covariates from embeddings?
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(embeddings, covariate_labels)
```
**Target**: High accuracy = covariate information retained

---

## Future Extensions

### 1. Continuous Time Interpolation
```yaml
- name: time_days
  type: continuous
  continuous_encoder:
    hidden_dim: 128
    output_dim: 256
```
Enable prediction for unseen timepoints (e.g., day 12, day 16)

### 2. Additional Covariates
```yaml
- name: dosage
  type: continuous

- name: treatment
  type: categorical
  num_categories: 5

- name: age
  type: continuous
```

### 3. Contrastive Learning
Implement proper contrastive loss (NT-Xent, InfoNCE) to better separate conditions:
```python
def contrastive_loss(embeddings, labels):
    # Positive pairs: same condition
    # Negative pairs: different condition
    ...
```

### 4. Zero-Shot Prediction
Use LoRA adapters to predict unseen covariate combinations:
```python
# Train on: day10 burn, day14 burn, day19 burn, day10 sham
# Predict: day14 sham, day19 sham
```

---

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size or use gradient accumulation
```yaml
training:
  batch_size: 8  # down from 16
  accumulate_grad_batches: 8  # effective batch size = 64
```

### Issue: LoRA not learning
**Solution**: Increase rank or alpha
```yaml
lora:
  r: 32  # up from 16
  lora_alpha: 64  # up from 32
```

### Issue: Covariate signal weak
**Solution**: Increase covariate embedding dimensions
```yaml
- name: timepoint
  embed_dim: 512  # up from 256
```

---

## References

- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **PEFT Library**: [Hugging Face PEFT](https://github.com/huggingface/peft)
- **STATE Model**: [Arc Institute STATE](https://github.com/ArcInstitute/state)
- **CPA**: [Compositional Perturbation Autoencoder](https://www.biorxiv.org/content/10.1101/2021.04.14.439903v1)

---

## Contact & Support

For questions or issues:
1. Check this README
2. Review notebook examples
3. Inspect configuration files
4. Check training logs and TensorBoard

**Created**: 2025-12-16
**Author**: Sujit (with Claude assistance)
**Version**: 1.0
