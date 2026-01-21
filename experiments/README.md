# Experiments Directory

This directory contains all experimental work for the burn/sham wound healing State model project, organized by experiment type.

**Created**: 2026-01-05
**Last Updated**: 2026-01-12

---

## Directory Structure

```
experiments/
├── baseline_analysis/          # Phase 1: Baseline SE-600M embeddings
├── st_fine_tuning/            # Phase 2: ST model fine-tuning (LoRA + mHC)
├── st_training/                # Phase 3: State Transition model training
└── velocity_integration/       # Phase 4: RNA velocity integration
```

---

## Experiment Descriptions

### 1. baseline_analysis/

**Purpose**: Establish baseline embeddings using pretrained SE-600M model

**Contents**:
- `baseline_analysis.ipynb`: Extract and evaluate baseline embeddings
- `data/`:
  - `filtered_final_sample_removed.h5ad`: Raw processed data
  - `burn_sham_processed.h5ad`: Cleaned and annotated
  - `burn_sham_baseline_embedded.h5ad`: With SE-600M embeddings

**Status**: ✅ Complete

**Key Findings**:
- Cell type accuracy: 96%
- Good batch correction
- Clear temporal structure
- **Decision**: Use baseline embeddings for ST training

---

### 2. st_fine_tuning/

**Purpose**: Fine-tune State Transition model with LoRA adapters and mHC for perturbation prediction

**Contents**:
- `st_lora_mhc_experiment.ipynb`: Comprehensive comparison notebook
- `configs/`:
  - `lora_config.yaml`: LoRA-only configuration
  - `lora_mhc_config.yaml`: LoRA + mHC configuration
- `results/`: Training outputs and comparisons

**Status**: 🔄 Ready for Training

**Approach**:
Compare 3 ST model variants:
1. **ST-Tahoe (Baseline)**: Pretrained model (no fine-tuning)
2. **ST-LoRA**: Fine-tuned with LoRA adapters (~1-5% parameters)
3. **ST-LoRA-mHC**: Fine-tuned with LoRA + mHC (manifold-constrained gradients)

**Key Innovation**:
- **mHC (Manifold-Constrained Hyper-Connections)**: Stabilizes optimal transport loss gradients via Sinkhorn-Knopp projection to doubly stochastic matrices
- **LoRA**: Parameter-efficient fine-tuning (only adapt attention layers)
- **Input**: Fixed SE-600M embeddings (from baseline_analysis)

**Expected Outcomes**:
- LoRA: Parameter efficiency (95%+ frozen)
- mHC: Training stability (smoother loss curves)
- Comparison: Best approach for burn/sham perturbation prediction

---

### 3. st_training/

**Purpose**: Train State Transition model for perturbation prediction (burn vs sham)

**Contents**:
- `phase3a_st_data_preparation.ipynb`: Data validation and config creation
- `phase3b_st_model_training.ipynb`: ST model training workflow
- `phase3c_st_evaluation.ipynb`: Evaluation and metrics
- `data/`: (Created during training)

**Status**: ✅ Code Complete - Ready for Training

**Configuration**:
- Baseline SE-600M embeddings (`X_state`)
- Timepoint embeddings added (discrete: day10/14/19)
- Hidden dim: 512
- Cell set length: 256
- Training steps: 20,000
- Hardware: 2× RTX 5000 Ada (DDP)

**Code Modifications**:
- `src/state/tx/models/state_transition.py`: Added timepoint encoder (lines 200-210, 431-448)
- `src/state/tx/data/dataset/scgpt_perturbation_dataset.py`: Extract timepoint_ids (lines 175-212)

**Success Criteria**:
- NN prediction distance < 0.3 (minimum)
- NN distance < 0.2 (strong)
- Temporal coherence maintained
- Cell-type-specific responses observed

---

### 4. velocity_integration/

**Purpose**: Integrate RNA velocity features to improve temporal predictions

**Contents**:
- `notebooks/`:
  - `step0_load_velocyto_data.ipynb`: Load .loom files, merge with AnnData
  - `step1_compute_velocity_features.ipynb`: Compute scVelo dynamics, extract features
  - `step2_velocity_st_training.ipynb`: (To be created) Training workflow
- `configs/`:
  - `st_with_velocity.yaml`: (To be created) Training configuration
- `data/`:
  - `burn_sham_with_velocyto.h5ad`: (Created by step0) With spliced/unspliced layers
  - `burn_sham_with_velocity.h5ad`: (Created by step1) With velocity features
- `results/`:
  - `figures/`: Velocity visualizations
  - `metrics/`: Comparison metrics

**Status**: ✅ Code Complete - Ready for Data Processing

**Approach**:
- Extract 3 scalar velocity features per cell:
  - `velocity_magnitude`: Transcriptional activity (0-1)
  - `velocity_pseudotime`: Trajectory position (0-1)
  - `velocity_confidence`: Velocity reliability (0-1)
- Encode as continuous covariates (nn.Sequential: Linear → SiLU → Linear)
- Add to ST model latent representation with learnable weight

**Code Modifications**:
- `src/state/tx/models/state_transition.py`: Added 3 velocity encoders (lines 212-237, 477-499)
- `src/state/tx/data/dataset/scgpt_perturbation_dataset.py`: Extract velocity features (lines 111-119, 224-235)

**Success Criteria**:
- Gene correlation improvement > +0.02
- Temporal coherence improvement > +0.01
- Burn-specific improvement > +0.03

**Quick Validation Timeline**:
1. Data processing (steps 0-1): 30 minutes
2. Training: 4-6 hours (unattended)
3. Evaluation: 30 minutes

---

## Workflow Summary

### Completed Phases

```
Phase 1 (baseline_analysis) → SE-600M embeddings (96% cell type accuracy)
                                      ↓
                                Phase 2 (st_fine_tuning)
                                      ├─→ ST-LoRA (LoRA adapters)
                                      └─→ ST-LoRA-mHC (LoRA + mHC)
                                      ↓
                                Phase 3 (st_training)
                                      ↓ (Add velocity)
                                Phase 4 (velocity_integration)
```

### Key Decisions

| Phase | Decision | Rationale |
|-------|----------|-----------|
| 1 → 2 | Fine-tune ST (not SE) | Perturbation prediction is the goal |
| 2 | Use LoRA + mHC | Parameter efficiency + gradient stability |
| 2 → 3 | Compare variants | Find best approach for wound healing |
| 3 → 4 | Add velocity as covariates | Improve temporal predictions without breaking embeddings |

---

## Data Files Summary

### Baseline Data (in `baseline_analysis/data/`)

| File | Size | Description | Used By |
|------|------|-------------|---------|
| `filtered_final_sample_removed.h5ad` | 1.5 GB | Raw processed data | Initial processing |
| `burn_sham_processed.h5ad` | 1.4 GB | Cleaned, annotated | Baseline embedding extraction |
| `burn_sham_baseline_embedded.h5ad` | 1.9 GB | With SE-600M embeddings | ST training (Phase 3) |

### ST Fine-Tuning Data (in `st_fine_tuning/results/`)

| File | Size | Description | Created By |
|------|------|-------------|------------|
| Model checkpoints | Varies | Trained LoRA/mHC models | Training scripts |
| Predictions | ~500 MB | Model predictions on test set | Evaluation |

### ST Training Data (in `st_training/data/`)

Currently empty - training generates checkpoints in `/home/scumpia-mrl/state_models/`

### Velocity Data (in `velocity_integration/data/`)

| File | Size | Description | Created By |
|------|------|-------------|------------|
| `burn_sham_with_velocyto.h5ad` | ~1-2 GB | With spliced/unspliced layers | step0_load_velocyto_data.ipynb |
| `burn_sham_with_velocity.h5ad` | ~1-2 GB | With velocity features | step1_compute_velocity_features.ipynb |

---

## External Data Sources

### Velocyto .loom Files

**Location**: `/home/scumpia-mrl/Desktop/Sujit/Projects/state-experimentation/velocyto_loom/`

**Copied From**: `/home/scumpia-mrl/Desktop/Sujit/Philip Scumpia Lab/burn_sham_wounds_scrnaseq_20250217/velocyto_output/`

**Files** (16 total):
- Burn samples (8): S1-S8 (D10×3, D14×3, D19×2)
- Sham samples (8): S9-S16 (D10×3, D14×3, D19×2)

**Note**: `.loom` files are in `.gitignore` (large binary files)

---


## Computational Resources

### Hardware

- **GPUs**: 2× NVIDIA RTX 5000 Ada (33 GB VRAM each)
- **Strategy**: DDP (Distributed Data Parallel)
- **CPUs**: 12 cores for scVelo dynamics



### File Paths

| What | Path |
|------|------|
| Raw data | `experiments/baseline_analysis/data/burn_sham_processed.h5ad` |
| Baseline embeddings | `experiments/baseline_analysis/data/burn_sham_baseline_embedded.h5ad` |
| LoRA embeddings | `experiments/lora_training/data/burn_sham_lora_embedded.h5ad` |
| Velocity data | `experiments/velocity_integration/data/burn_sham_with_velocity.h5ad` |
| Model checkpoints | `/home/scumpia-mrl/state_models/` |
| Velocyto files | `velocyto_loom/` (project root) |

---

## Contact & Notes

**Project Owner**: Sujit
**Created**: 2026-01-05

