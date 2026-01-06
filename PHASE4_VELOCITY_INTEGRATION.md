# Phase 4: RNA Velocity Integration - Implementation Summary

**Date**: 2026-01-05
**Status**: ✅ Code Complete - Ready for Data Processing
**Strategy**: Velocity-Derived Scalar Features as ST Model Covariates

---

## 📋 Overview

This phase integrates RNA velocity data into the Arc Institute State Transition model to improve temporal perturbation predictions for burn/sham wound healing. We extract scalar velocity features and encode them as continuous covariates, following the same pattern used for timepoint embeddings in Phase 3.

**Quick Validation Approach** (2-3 hours hands-on):
1. Load velocyto .loom files and merge with AnnData
2. Compute scVelo dynamics and extract 3 scalar features
3. Add velocity encoders to ST model (continuous → embedding)
4. Train and compare to baseline (with/without velocity)

---

## ✅ Completed Work

### 1. Project Organization

Created structured directory for velocity experiments:

```
state-experimentation/
├── experiments/velocity_integration/
│   ├── notebooks/
│   │   ├── step0_load_velocyto_data.ipynb          ✅
│   │   ├── step1_compute_velocity_features.ipynb   ✅
│   │   └── step2_velocity_st_training.ipynb        (to be created)
│   ├── data/
│   │   ├── burn_sham_with_velocyto.h5ad           (created by step0)
│   │   └── burn_sham_with_velocity.h5ad           (created by step1)
│   ├── configs/
│   │   └── st_with_velocity.yaml                  (to be created)
│   └── results/
│       ├── figures/
│       └── metrics/
├── velocyto_loom/  (16 .loom files)                ✅
```

### 2. Data Processing Notebooks

#### **[experiments/velocity_integration/notebooks/step0_load_velocyto_data.ipynb](experiments/velocity_integration/notebooks/step0_load_velocyto_data.ipynb)**

**Purpose**: Load velocyto .loom files and merge with existing AnnData

**Key Functions**:
```python
def load_and_rename_loom(path, sample_id):
    """Load loom file and rename barcodes to match existing AnnData format.

    Velocyto barcodes: "BARCODE:1x"
    Target format: "sample_id_BARCODE-1"
    """
    ldata = sc.read(str(path), cache=True)
    barcodes = [bc.split(':')[1] for bc in ldata.obs.index.tolist()]
    barcodes = [bc[0:len(bc)-1] + '-1' for bc in barcodes]
    barcodes = [f"{sample_id}_" + bc for bc in barcodes]
    ldata.obs.index = barcodes
    ldata.var_names_make_unique()
    return ldata
```

**Workflow**:
1. Copy 16 .loom files from external drive to `velocyto_loom/`
2. Load and rename barcodes for each sample
3. Merge by timepoint/condition:
   - Burn: D10 (3 mice) + D14 (3 mice) + D19 (2 mice)
   - Sham: D10 (3 mice) + D14 (3 mice) + D19 (2 mice)
4. Merge velocyto layers into existing AnnData using `scv.utils.merge()`
5. Save to `burn_sham_with_velocyto.h5ad`

**Output Layers**:
- `spliced`: Spliced mRNA counts
- `unspliced`: Unspliced mRNA counts
- `ambiguous`: Ambiguous reads

---

#### **[experiments/velocity_integration/notebooks/step1_compute_velocity_features.ipynb](experiments/velocity_integration/notebooks/step1_compute_velocity_features.ipynb)**

**Purpose**: Compute scVelo dynamics and extract velocity-derived scalar features

**Workflow**:
1. Load `burn_sham_with_velocyto.h5ad`
2. scVelo preprocessing:
   ```python
   scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
   scv.pp.moments(adata, n_pcs=50, n_neighbors=30)
   ```
3. Compute velocity:
   ```python
   scv.tl.recover_dynamics(adata, n_jobs=12)  # ~10-15 minutes
   scv.tl.velocity(adata, mode="stochastic")
   scv.tl.velocity_graph(adata)
   scv.tl.latent_time(adata)
   ```
4. Extract 3 scalar features:
   - **velocity_magnitude**: `np.linalg.norm(adata.layers['velocity'], axis=1)`
     *Interpretation*: Transcriptional activity (higher = more dynamic)
   - **velocity_pseudotime**: `adata.obs['latent_time']`
     *Interpretation*: Trajectory position from velocity
   - **velocity_confidence**: `adata.obs['velocity_length']`
     *Interpretation*: Reliability of velocity estimate
5. Normalize all features to [0, 1] using MinMaxScaler
6. Validate (no NaNs, correct range)
7. Visualize velocity field and feature distributions
8. Save to `burn_sham_with_velocity.h5ad`

**Output Features** (in `.obs`):
- `velocity_magnitude` (0-1)
- `velocity_pseudotime` (0-1)
- `velocity_confidence` (0-1)

**Visualizations**:
- Velocity stream plots (condition, timepoint, cell type)
- Feature UMAPs
- Distribution plots (violin plots by condition/timepoint)

---

### 3. Code Modifications

#### **File: [src/state/tx/data/dataset/scgpt_perturbation_dataset.py](src/state/tx/data/dataset/scgpt_perturbation_dataset.py)**

**Modification 1**: Validation in `__init__` (lines 111-119)

```python
# Validate velocity features exist (Phase 4 - Velocity Integration)
required_velocity_features = ['velocity_magnitude', 'velocity_pseudotime', 'velocity_confidence']
self.has_velocity = all(f"obs/{feature}" in self.h5_file for feature in required_velocity_features)
if self.has_velocity:
    logger.info(f"scGPTPerturbationDataset ([{self.name}]) Found velocity features: {required_velocity_features}")
else:
    missing = [f for f in required_velocity_features if f"obs/{f}" not in self.h5_file]
    if missing:
        logger.warning(f"scGPTPerturbationDataset ([{self.name}]) Missing velocity features: {missing}")
```

**Modification 2**: Feature extraction in `__getitem__` (lines 224-235)

```python
# Add velocity features if available (Phase 4 - Velocity Integration)
if self.has_velocity:
    try:
        velocity_mag = float(self.h5_file["obs/velocity_magnitude"][underlying_idx])
        velocity_pseudo = float(self.h5_file["obs/velocity_pseudotime"][underlying_idx])
        velocity_conf = float(self.h5_file["obs/velocity_confidence"][underlying_idx])

        sample["velocity_magnitude"] = velocity_mag
        sample["velocity_pseudotime"] = velocity_pseudo
        sample["velocity_confidence"] = velocity_conf
    except Exception as e:
        logger.warning(f"Could not read velocity features for idx {underlying_idx}: {e}")
```

---

#### **File: [src/state/tx/models/state_transition.py](src/state/tx/models/state_transition.py)**

**Modification 1**: Velocity encoders in `__init__` (lines 212-237)

```python
# Velocity feature encoders (Phase 4 - Velocity Integration)
# Continuous scalar → hidden_dim embedding
self.use_velocity = kwargs.get("use_velocity_features", False)

if self.use_velocity:
    self.velocity_mag_encoder = nn.Sequential(
        nn.Linear(1, 128),
        nn.SiLU(),
        nn.Linear(128, hidden_dim)
    )

    self.velocity_pseudotime_encoder = nn.Sequential(
        nn.Linear(1, 128),
        nn.SiLU(),
        nn.Linear(128, hidden_dim)
    )

    self.velocity_confidence_encoder = nn.Sequential(
        nn.Linear(1, 128),
        nn.SiLU(),
        nn.Linear(128, hidden_dim)
    )

    # Learnable weight for velocity importance
    self.velocity_weight = nn.Parameter(torch.tensor(0.3))
    logger.info(f"Added velocity encoders: 3 features × {hidden_dim} dim (initial weight: 0.3)")
```

**Modification 2**: Velocity embeddings in forward pass (lines 477-499)

```python
# Add velocity embeddings if available (Phase 4 - Velocity Integration)
if self.use_velocity and "velocity_magnitude" in batch:
    # Extract velocity scalars (batch_size,) → (batch_size, 1)
    velocity_mag = batch["velocity_magnitude"].unsqueeze(-1).float()
    velocity_pseudo = batch["velocity_pseudotime"].unsqueeze(-1).float()
    velocity_conf = batch["velocity_confidence"].unsqueeze(-1).float()

    # Encode to hidden_dim
    velocity_mag_emb = self.velocity_mag_encoder(velocity_mag)  # Shape: [B, hidden_dim]
    velocity_pseudo_emb = self.velocity_pseudotime_encoder(velocity_pseudo)
    velocity_conf_emb = self.velocity_confidence_encoder(velocity_conf)

    # Combine (average)
    velocity_emb = (velocity_mag_emb + velocity_pseudo_emb + velocity_conf_emb) / 3  # Shape: [B, hidden_dim]

    # Reshape to match sequence structure
    if padded:
        velocity_emb = velocity_emb.reshape(-1, self.cell_sentence_len, velocity_emb.size(-1))
    else:
        velocity_emb = velocity_emb.reshape(1, -1, velocity_emb.size(-1))

    # Add to latent with learnable weight
    seq_input = seq_input + self.velocity_weight * velocity_emb
```

---

## 🎯 Next Steps: Data Processing & Training

### Step 1: Run Data Processing Notebooks (~30 minutes)

```bash
# Start Jupyter
jupyter notebook

# Run in order:
# 1. experiments/velocity_integration/notebooks/step0_load_velocyto_data.ipynb
# 2. experiments/velocity_integration/notebooks/step1_compute_velocity_features.ipynb
```

**Expected Outputs**:
- `experiments/velocity_integration/data/burn_sham_with_velocyto.h5ad` (~1-2 GB)
- `experiments/velocity_integration/data/burn_sham_with_velocity.h5ad` (~1-2 GB)
- Velocity field visualizations in `results/figures/`

---

### Step 2: Create Training Configuration (~5 minutes)

Copy Phase 3 config and add velocity flags:

**File**: `experiments/velocity_integration/configs/st_with_velocity.yaml`

```yaml
data:
  toml_config_path: examples/burn_sham.toml
  embed_key: X_state  # Baseline SE-600M embeddings
  pert_col: condition
  control_pert: sham
  cell_type_key: cell_types_simple_short
  batch_col: mouse_id
  batch_size: 16
  num_workers: 8

model:
  model_class: state_transition
  input_dim: 2048
  output_dim: 2048
  hidden_dim: 512
  cell_set_len: 256
  use_timepoint_embedding: true  # Keep Phase 3 modification
  num_timepoints: 3
  use_velocity_features: true    # NEW - Enable velocity
  num_timepoints: 3

training:
  max_steps: 20000
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_steps: 1000
  gradient_clip_val: 1.0
  devices: 2
  strategy: ddp
  log_every_n_steps: 50
  val_check_interval: 500

output:
  output_dir: /home/scumpia-mrl/state_models/st_burn_sham_velocity
  experiment_name: st_burn_sham_velocity_v1
  save_top_k: 3
  monitor: val_loss
```

---

### Step 3: Train Model with Velocity (~4-6 hours)

```bash
state tx train \
  data.kwargs.toml_config_path=examples/burn_sham.toml \
  data.kwargs.embed_key=X_state \
  data.kwargs.pert_col=condition \
  data.kwargs.control_pert=sham \
  data.kwargs.cell_type_key=cell_types_simple_short \
  data.kwargs.batch_col=mouse_id \
  model=state \
  model.kwargs.input_dim=2048 \
  model.kwargs.output_dim=2048 \
  model.kwargs.hidden_dim=512 \
  model.kwargs.cell_set_len=256 \
  model.kwargs.use_timepoint_embedding=true \
  model.kwargs.num_timepoints=3 \
  model.kwargs.use_velocity_features=true \
  training.max_steps=20000 \
  training.batch_size=16 \
  training.learning_rate=0.0001 \
  training.weight_decay=0.01 \
  training.warmup_steps=1000 \
  training.gradient_clip_val=1.0 \
  training.devices=2 \
  training.strategy=ddp \
  training.log_every_n_steps=50 \
  training.val_check_interval=500 \
  output_dir=/home/scumpia-mrl/state_models/st_burn_sham_velocity \
  name=st_with_velocity_v1
```

**Monitor Training**:
```bash
tensorboard --logdir=/home/scumpia-mrl/state_models/st_burn_sham_velocity
```

---

### Step 4: Evaluation & Comparison (~30 minutes)

Create evaluation notebook to compare velocity vs. baseline:

**Metrics to Compare**:

1. **Perturbation Prediction Accuracy**:
   - Per-gene Pearson correlation with actual burn cells
   - Compare: `mean_corr_with_velocity` vs `mean_corr_baseline`
   - **Success threshold**: Improvement > +0.02

2. **Temporal Coherence**:
   - Cosine distance between consecutive timepoints (day10→14, day14→19)
   - Compare to non-consecutive (day10→19)
   - **Success threshold**: Improvement > +0.01

3. **Burn-Specific Improvement**:
   - Separate burn vs sham predictions
   - Check if velocity especially helps burn trajectory
   - **Success threshold**: Improvement > +0.03 for burn cells

**Decision Framework**:
- **Strong Success** (all thresholds exceeded): Write up findings, proceed with analysis
- **Moderate Success** (some thresholds): Investigate which features help most
- **No Improvement** (no thresholds): Velocity doesn't help, focus on alternatives

---

## 🔬 Technical Details

### Architecture Pattern

Follows the same pattern as timepoint embeddings from Phase 3:

**Categorical Covariate** (timepoint):
```python
# In __init__:
self.timepoint_encoder = nn.Embedding(num_timepoints, hidden_dim)

# In forward:
timepoint_emb = self.timepoint_encoder(timepoint_ids)
seq_input = seq_input + timepoint_emb
```

**Continuous Covariate** (velocity):
```python
# In __init__:
self.velocity_encoder = nn.Sequential(
    nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, hidden_dim)
)

# In forward:
velocity_emb = self.velocity_encoder(velocity_feature.unsqueeze(-1))
seq_input = seq_input + self.velocity_weight * velocity_emb
```

---

### Why ST Model Instead of SE Model?

| Aspect | SE Model | ST Model | Winner |
|--------|----------|----------|--------|
| **Architecture** | Rigid (ESM2 embeddings → Transformer) | Modular (covariate composition) | ✅ ST |
| **Modification Effort** | Major rewrite required | Follow existing pattern | ✅ ST |
| **Velocity Integration** | Gene-level embedding changes | Continuous covariates | ✅ ST |
| **Pattern Precedent** | None | Batch encoder, timepoint encoder | ✅ ST |
| **Risk** | High (breaks pretrained model) | Low (additive modification) | ✅ ST |

**Decision**: Integrate velocity at ST level as continuous covariates

---

### Why Scalar Features Instead of Gene-Level Velocity?

| Approach | Pros | Cons | Timeline |
|----------|------|------|----------|
| **Scalar Features** (chosen) | - Quick validation (2-3 hours)<br>- Low complexity<br>- Captures global trends | - Less granular<br>- May miss gene-specific effects | 2-3 hours |
| **Gene-Level Velocity** | - Full velocity information<br>- Gene-specific dynamics | - High complexity<br>- Requires velocity matrix encoder<br>- 1-2 weeks implementation | 1-2 weeks |

**Decision**: Start with scalar features for quick validation. If successful, consider gene-level integration later.

---

## 📊 Expected Outcomes

### Best Case (Strong Success)
- All metrics exceed thresholds (+0.02, +0.01, +0.03)
- Velocity provides clear temporal signal
- Burn trajectory predictions significantly improved
- **Action**: Write up findings, prepare for qualifying exam

### Realistic Case (Moderate Success)
- Some metrics improve, others don't
- Velocity magnitude or pseudotime helps, but not confidence
- **Action**: Ablation study (which features help?), refine approach

### Worst Case (No Improvement)
- All metrics < +0.01
- Velocity redundant with timepoint embedding
- **Action**: Investigate alternatives (cell cycle, gene modules)

---

## 🚧 Potential Issues & Solutions

### Issue 1: Velocity Features Redundant with Timepoint

**Diagnostic**: Check correlation between `velocity_pseudotime` and `timepoint_id`

```python
import pandas as pd
from scipy.stats import spearmanr

# If correlation > 0.8, velocity is redundant
corr, p = spearmanr(adata.obs['velocity_pseudotime'], adata.obs['time_days'])
```

**Solution**: Use only `velocity_magnitude` and `velocity_confidence` (independent of time)

---

### Issue 2: Noisy Velocity Estimates

**Diagnostic**: Visualize velocity stream plots - inconsistent directions?

**Solutions**:
1. Increase `n_neighbors` in `scv.pp.moments()` (30 → 50)
2. Use dynamical mode: `scv.tl.velocity(adata, mode='dynamical')`
3. Weight by confidence: `seq_input += velocity_conf * velocity_emb`

---

### Issue 3: CUDA Out of Memory

**Solution**: Reduce batch size or cell set length

```bash
# Reduce batch_size from 16 to 8
training.batch_size=8

# Reduce cell_set_len from 256 to 128
model.kwargs.cell_set_len=128
```

---

### Issue 4: Training Not Converging

**Diagnostic**: TensorBoard shows flat loss

**Solutions**:
1. Increase warmup: `training.warmup_steps=2000`
2. Adjust learning rate: `training.learning_rate=0.00005`
3. Check velocity weight initialization: Try 0.1 instead of 0.3

---

## 📁 File Structure Summary

```
state-experimentation/
├── experiments/velocity_integration/          ✅ NEW
│   ├── notebooks/
│   │   ├── step0_load_velocyto_data.ipynb    ✅ Created
│   │   ├── step1_compute_velocity_features.ipynb ✅ Created
│   │   └── step2_velocity_st_training.ipynb  (to create)
│   ├── data/
│   │   ├── burn_sham_with_velocyto.h5ad     (step0 output)
│   │   └── burn_sham_with_velocity.h5ad     (step1 output)
│   ├── configs/
│   │   └── st_with_velocity.yaml            (to create)
│   └── results/
│       ├── figures/                          (velocity plots)
│       └── metrics/                          (comparison metrics)
├── velocyto_loom/                            ✅ NEW (16 .loom files)
├── src/state/tx/
│   ├── data/dataset/
│   │   └── scgpt_perturbation_dataset.py    ✅ Modified (lines 111-119, 224-235)
│   └── models/
│       └── state_transition.py              ✅ Modified (lines 212-237, 477-499)
├── PHASE4_VELOCITY_INTEGRATION.md           ✅ This file
└── .claude/plans/vast-herding-lamport.md    ✅ Detailed implementation plan
```

---

## 🎓 Qualifying Exam Implications

### If Velocity Improves Predictions

**Narrative**: *"We integrated RNA velocity to capture transcriptional dynamics, improving temporal perturbation prediction by X%. This demonstrates that velocity-derived features complement static embeddings for trajectory modeling."*

**Slides**:
- Velocity field visualizations
- Comparison metrics (with/without velocity)
- Biological interpretation (velocity in burn vs. sham)

### If Velocity Doesn't Help

**Narrative**: *"We systematically evaluated RNA velocity integration but found it redundant with timepoint information. This ablation study validates our modeling choices and shows velocity may not be necessary when explicit time covariates are available."*

**Slides**:
- Systematic evaluation approach
- Diagnostic analysis (correlation with timepoint)
- Alternative features tested (cell cycle, gene modules)

**Both outcomes are valuable contributions!**

---

## 📚 References

- **scVelo Paper**: Bergen et al., Nature Biotechnology 2020
- **velocyto**: La Manno et al., Nature 2018
- **Arc Institute State Model**: [GitHub](https://github.com/ArcInstitute/state)
- **Phase 3 Documentation**: [PHASE3_SUMMARY.md](PHASE3_SUMMARY.md)
- **Implementation Plan**: [.claude/plans/vast-herding-lamport.md](.claude/plans/vast-herding-lamport.md)

---

## ✨ Summary

Phase 4 velocity integration is **code complete**! All modifications follow proven patterns from Phase 3 (timepoint embeddings) and are ready for testing.

**Time Commitment**:
- Data processing (notebooks): 30 minutes
- Training: 4-6 hours (unattended)
- Evaluation: 30 minutes

**Risk Level**: Low (conservative approach, follows existing patterns)

**Expected Outcome**: Quick validation of whether velocity helps temporal predictions, with clear decision framework for next steps.

---

**Created**: 2026-01-05
**Last Updated**: 2026-01-05
**Status**: ✅ Code Complete - Ready for Data Processing

---

## Quick Start Commands

```bash
# 1. Run data processing notebooks
jupyter notebook experiments/velocity_integration/notebooks/step0_load_velocyto_data.ipynb
jupyter notebook experiments/velocity_integration/notebooks/step1_compute_velocity_features.ipynb

# 2. Train model with velocity
state tx train \
  model.kwargs.use_velocity_features=true \
  [... see full command in Step 3 above ...]

# 3. Monitor
tensorboard --logdir=/home/scumpia-mrl/state_models/st_burn_sham_velocity

# 4. Evaluate (create notebook based on phase3c_st_evaluation.ipynb)
```

**Next**: Run step0 notebook to load velocyto data!
