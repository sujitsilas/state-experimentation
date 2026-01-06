# Experiments Directory

This directory contains all experimental work for the burn/sham wound healing State model project, organized by experiment type.

**Created**: 2026-01-05
**Last Updated**: 2026-01-05

---

## Directory Structure

```
experiments/
├── baseline_analysis/          # Phase 1: Baseline SE-600M embeddings
├── lora_training/              # Phase 2: LoRA adapter training
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

### 2. lora_training/

**Purpose**: Train LoRA adapter to incorporate temporal/condition information into embeddings

**Contents**:
- `phase2_lora_multicov_training.ipynb`: LoRA training with multiple covariates
- `data/`:
  - `burn_sham_lora_embedded.h5ad`: LoRA-adapted embeddings

**Status**: ✅ Complete

**Key Findings**:
- Cell type accuracy: ~75% (degraded from 96%)
- Strong temporal/condition separation
- Mixed cell type clusters
- **Decision**: Do NOT use LoRA embeddings for ST model (cell type structure critical)

**Lessons Learned**:
- LoRA optimized for wrong objective (temporal signal at cost of cell types)
- Better to add temporal/condition as ST model covariates
- Strong temporal signal validates that information is learnable

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
Phase 1 (baseline_analysis) → Phase 2 (lora_training)
                                      ↓ (Decision: Use baseline)
                                Phase 3 (st_training)
                                      ↓ (Add velocity)
                                Phase 4 (velocity_integration)
```

### Key Decisions

| Phase | Decision | Rationale |
|-------|----------|-----------|
| 1 → 2 | Try LoRA adaptation | Incorporate temporal/condition into embeddings |
| 2 → 3 | Use baseline (not LoRA) | Cell type structure critical for ST model |
| 3 → 4 | Add velocity as covariates | Improve temporal predictions without breaking embeddings |

---

## Data Files Summary

### Baseline Data (in `baseline_analysis/data/`)

| File | Size | Description | Used By |
|------|------|-------------|---------|
| `filtered_final_sample_removed.h5ad` | 1.5 GB | Raw processed data | Initial processing |
| `burn_sham_processed.h5ad` | 1.4 GB | Cleaned, annotated | Baseline embedding extraction |
| `burn_sham_baseline_embedded.h5ad` | 1.9 GB | With SE-600M embeddings | ST training (Phase 3) |

### LoRA Data (in `lora_training/data/`)

| File | Size | Description | Used By |
|------|------|-------------|---------|
| `burn_sham_lora_embedded.h5ad` | 1.9 GB | LoRA-adapted embeddings | Evaluation only (not used for ST) |

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

## .gitignore Rules

The following are excluded from version control:

```gitignore
velocyto_loom/          # Large binary .loom files
experiments/*/data/     # All data files (too large)
experiments/*/results/  # Generated outputs
*.h5ad                  # AnnData files (global rule)
```

**What IS tracked**:
- Jupyter notebooks (`.ipynb`)
- Configuration files (`.yaml`, `.toml`)
- Code modifications (`src/`)
- Documentation (`.md`)

---

## Computational Resources

### Hardware

- **GPUs**: 2× NVIDIA RTX 5000 Ada (33 GB VRAM each)
- **Strategy**: DDP (Distributed Data Parallel)
- **CPUs**: 12 cores for scVelo dynamics

### Training Times

| Experiment | Duration | Hardware |
|------------|----------|----------|
| Baseline embedding extraction | ~30 min | Single GPU |
| LoRA training | 4-6 hours | 2× GPU (DDP) |
| ST training (Phase 3) | 4-6 hours | 2× GPU (DDP) |
| ST training (Phase 4, with velocity) | 4-6 hours | 2× GPU (DDP) |
| scVelo dynamics computation | 10-15 min | 12 CPUs |

---

## Next Steps

### Immediate (Phase 4)

1. **Run data processing notebooks**:
   ```bash
   jupyter notebook experiments/velocity_integration/notebooks/step0_load_velocyto_data.ipynb
   jupyter notebook experiments/velocity_integration/notebooks/step1_compute_velocity_features.ipynb
   ```

2. **Create training configuration**:
   - Copy from `configs/state_transition_burn_sham.yaml`
   - Add `use_velocity_features: true`
   - Save to `experiments/velocity_integration/configs/st_with_velocity.yaml`

3. **Train model**:
   ```bash
   state tx train model.kwargs.use_velocity_features=true [...]
   ```

4. **Evaluate and compare** to Phase 3 baseline

### Future Directions

If Phase 4 velocity integration is successful:
- Gene-level velocity integration (1-2 weeks)
- Velocity matrix encoder (attention over genes)
- Cell-type-specific velocity patterns

If Phase 4 shows no improvement:
- Cell cycle scoring (S/G2M phases)
- Wound healing gene module signatures
- Trajectory pseudotime (Palantir)

---

## Documentation

### Phase Summaries

- **Phase 3**: [/PHASE3_SUMMARY.md](../PHASE3_SUMMARY.md)
- **Phase 4**: [/PHASE4_VELOCITY_INTEGRATION.md](../PHASE4_VELOCITY_INTEGRATION.md)

### Implementation Plans

- **Velocity Integration**: [/.claude/plans/vast-herding-lamport.md](../.claude/plans/vast-herding-lamport.md)

### Project Guide

- **Overall Plan**: [/CLAUDE.md](../CLAUDE.md) (in `.gitignore`)

---

## Quick Reference

### Common Commands

```bash
# Start Jupyter
jupyter notebook experiments/

# Train ST model (Phase 3)
state tx train data.kwargs.embed_key=X_state model.kwargs.use_timepoint_embedding=true [...]

# Train ST model (Phase 4, with velocity)
state tx train data.kwargs.embed_key=X_state model.kwargs.use_velocity_features=true [...]

# Monitor training
tensorboard --logdir=/home/scumpia-mrl/state_models/

# Check data files
ls -lh experiments/*/data/
```

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
**Purpose**: PhD Qualifying Exam (3-month timeline)

**Notes**:
- All phases build on each other sequentially
- Each phase has independent notebooks for reproducibility
- Data files are large (~1-2 GB each) and excluded from git
- Training requires 2× RTX 5000 Ada GPUs
- Total compute time: ~25-30 hours across all phases

---

**For detailed implementation information, see the phase-specific documentation linked above.**
