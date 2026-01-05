# Phase 3: State Transition Model - Complete Implementation Summary

**Date**: 2026-01-05
**Status**: ✅ Ready for Training
**Strategy**: Baseline SE-600M + Temporal Covariates (Strategy 2)

---

## 📋 Overview

This phase implements Arc Institute's State Transition (ST) model for perturbation prediction on burn/sham wound healing data. We use baseline SE-600M embeddings (96% cell type accuracy) with temporal covariates, rather than LoRA embeddings (75% cell type accuracy), to preserve cell type structure critical for perturbation prediction.

---

## ✅ Completed Work

### 1. Code Modifications

Modified Arc Institute's State Transition model to support timepoint embeddings:

#### **File: [src/state/tx/models/state_transition.py](src/state/tx/models/state_transition.py)**

**Lines 200-210**: Added timepoint encoder initialization
```python
# Add an optional encoder for timepoint variable
self.timepoint_encoder = None
self.timepoint_dim = None
if kwargs.get("use_timepoint_embedding", False):
    num_timepoints = kwargs.get("num_timepoints", 3)  # Default: day10, day14, day19
    self.timepoint_encoder = nn.Embedding(
        num_embeddings=num_timepoints,
        embedding_dim=hidden_dim,
    )
    self.timepoint_dim = num_timepoints
    logger.info(f"Added timepoint embedding: {num_timepoints} timepoints × {hidden_dim} dim")
```

**Lines 431-448**: Added timepoint embedding in forward pass
```python
if self.timepoint_encoder is not None:
    # Extract timepoint indices (assume they are integers: 0=day10, 1=day14, 2=day19)
    timepoint_indices = batch.get("timepoint_ids")

    if timepoint_indices is not None:
        # Handle one-hot encoded timepoint indices
        if timepoint_indices.dim() > 1 and timepoint_indices.size(-1) == self.timepoint_dim:
            timepoint_indices = timepoint_indices.argmax(-1)

        # Reshape timepoint indices to match sequence structure
        if padded:
            timepoint_indices = timepoint_indices.reshape(-1, self.cell_sentence_len)
        else:
            timepoint_indices = timepoint_indices.reshape(1, -1)

        # Get timepoint embeddings and add to sequence input
        timepoint_embeddings = self.timepoint_encoder(timepoint_indices.long())  # Shape: [B, S, hidden_dim]
        seq_input = seq_input + timepoint_embeddings
```

#### **File: [src/state/tx/data/dataset/scgpt_perturbation_dataset.py](src/state/tx/data/dataset/scgpt_perturbation_dataset.py)**

**Lines 175-212**: Added timepoint extraction and encoding
```python
# Get timepoint information (if available in the h5 file)
timepoint_id = None
if "obs/timepoint" in self.h5_file:
    try:
        timepoint_data = self.h5_file["obs/timepoint"]
        if hasattr(timepoint_data, 'categories'):  # Categorical data
            timepoint_code = timepoint_data['codes'][underlying_idx]
            timepoint_name = timepoint_data['categories'][timepoint_code].decode() if isinstance(timepoint_data['categories'][timepoint_code], bytes) else str(timepoint_data['categories'][timepoint_code])
        else:  # Direct string array
            timepoint_name = timepoint_data[underlying_idx]
            if isinstance(timepoint_name, bytes):
                timepoint_name = timepoint_name.decode()
            else:
                timepoint_name = str(timepoint_name)

        # Map timepoint to integer ID (day10=0, day14=1, day19=2)
        timepoint_map = {"day10": 0, "day14": 1, "day19": 2}
        timepoint_id = timepoint_map.get(timepoint_name, 0)
    except Exception as e:
        logger.warning(f"Could not read timepoint for idx {underlying_idx}: {e}")

# ... in sample dict:
if timepoint_id is not None:
    sample["timepoint_ids"] = timepoint_id
```

### 2. Configuration Files

Created complete training configuration:

#### **[examples/burn_sham.toml](examples/burn_sham.toml)**
```toml
[datasets]
burn_sham = "/home/scumpia-mrl/Desktop/Sujit/Projects/state-experimentation/"

[training]
burn_sham = "train"
```

#### **[configs/state_transition_burn_sham.yaml](configs/state_transition_burn_sham.yaml)**
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
  use_timepoint_embedding: true  # Our modification
  num_timepoints: 3  # day10, day14, day19

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
  output_dir: /home/scumpia-mrl/state_models/st_burn_sham
  experiment_name: st_burn_sham_v1
  save_top_k: 3
  monitor: val_loss
```

### 3. Jupyter Notebooks

Created three comprehensive notebooks for the complete workflow:

#### **[phase3a_st_data_preparation.ipynb](phase3a_st_data_preparation.ipynb)**
- Loads baseline SE-600M embeddings
- Validates data format and required columns
- Creates TOML and YAML configuration files
- Visualizes data distributions
- Documents code modifications

#### **[phase3b_st_model_training.ipynb](phase3b_st_model_training.ipynb)**
- Validates prerequisites (data, configs)
- Constructs training command with Hydra overrides
- Provides multiple training options (terminal, tmux/screen)
- Sets up monitoring tools (TensorBoard)
- Tracks checkpoints and training progress
- Parses training logs and plots loss curves

#### **[phase3c_st_evaluation.ipynb](phase3c_st_evaluation.ipynb)**
- Loads best trained checkpoint
- Generates perturbation predictions (sham → burn)
- Computes comprehensive evaluation metrics:
  - Perturbation prediction accuracy (cosine distance)
  - Temporal coherence
  - Cell-type-specific responses
- Creates UMAP visualizations comparing predictions to actual
- Generates comprehensive evaluation report
- Saves all results (metrics, predictions, plots)

### 4. Documentation

Updated all existing notebooks:

#### **[phase2_lora_multicov_training.ipynb](phase2_lora_multicov_training.ipynb)**
- Added comprehensive summary cell
- Documented LoRA findings (strong temporal/condition signal, mixed cell types)
- Explained decision to use baseline SE-600M instead
- Links to Phase 3 notebooks

---

## 🎯 Training Command

To start training, run in terminal:

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
  training.max_steps=20000 \
  training.batch_size=16 \
  training.learning_rate=0.0001 \
  training.devices=2 \
  training.strategy=ddp \
  training.gradient_clip_val=1.0 \
  training.val_check_interval=500 \
  training.log_every_n_steps=50 \
  output_dir=/home/scumpia-mrl/state_models/st_burn_sham \
  name=st_burn_sham_v1
```

**Expected Duration**: 4-6 hours on 2× RTX 5000 Ada (33GB VRAM each)

---

## 📊 Evaluation Metrics

After training, the evaluation notebook computes:

### 1. Perturbation Prediction Accuracy
- **Metric**: Cosine distance between predicted and actual burn embeddings
- **Success Criterion**: NN distance < 0.3
- **Interpretation**: Lower is better (closer predictions to actual)

### 2. Temporal Coherence
- **Metric**: Average distance between consecutive timepoint predictions
- **Interpretation**: Lower indicates smoother temporal progression

### 3. Cell-Type-Specific Responses
- **Metric**: Perturbation effect magnitude per cell type
- **Interpretation**: Validates biological interpretability (different cell types should show different responses)

---

## 📁 File Structure

```
state-experimentation/
├── phase3a_st_data_preparation.ipynb      ✅ Data prep & config
├── phase3b_st_model_training.ipynb        ✅ Training setup
├── phase3c_st_evaluation.ipynb            ✅ Evaluation & analysis
├── configs/
│   └── state_transition_burn_sham.yaml    ✅ Training config
├── examples/
│   └── burn_sham.toml                     ✅ Data split config
├── src/state/tx/models/
│   └── state_transition.py                ✅ Modified (lines 200-210, 431-448)
├── src/state/tx/data/dataset/
│   └── scgpt_perturbation_dataset.py      ✅ Modified (lines 175-212)
└── results/                                (Created after evaluation)
    ├── st_prediction_metrics.csv
    ├── st_temporal_coherence.csv
    ├── st_predictions_day10.h5ad
    ├── st_predictions_day14.h5ad
    ├── st_predictions_day19.h5ad
    └── st_evaluation_report.txt
```

---

## 🔄 Workflow

### Step 1: Data Preparation (5 minutes)
```bash
jupyter notebook phase3a_st_data_preparation.ipynb
```
- Run all cells to validate data and create configs
- Verify all required columns exist
- Check data distributions

### Step 2: Training (4-6 hours)
```bash
# Option A: Direct terminal
state tx train [... see training command above ...]

# Option B: Background (tmux)
tmux new -s st_training
state tx train [...]
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t st_training
```

Monitor with TensorBoard:
```bash
tensorboard --logdir=/home/scumpia-mrl/state_models/st_burn_sham
```

### Step 3: Evaluation (10-15 minutes)
```bash
jupyter notebook phase3c_st_evaluation.ipynb
```
- Run all cells to generate predictions and metrics
- Review UMAPs and evaluation report
- Check if success criteria met

---

## 🎓 Success Criteria

### Minimum Success (Pass Qualifying Exam)
- ✅ NN prediction distance < 0.3
- ✅ Temporal progression maintained
- ✅ Cell-type-specific responses observed

### Strong Success
- ✅ All of above +
- ✅ NN distance < 0.2
- ✅ Smooth temporal coherence (consistent distances)
- ✅ Biologically interpretable perturbation effects

### Outstanding Success
- ✅ All of above +
- ✅ Novel biological insights discovered
- ✅ Validation against known wound healing markers
- ✅ Generalizes to unseen timepoints/cell types

---

## 🔧 Troubleshooting

### If Training Fails

**CUDA Out of Memory**:
- Reduce `batch_size` to 8 or 12
- Reduce `cell_set_len` to 128

**Loss Not Decreasing**:
- Increase `warmup_steps` to 2000
- Try different `learning_rate` (5e-5, 2e-4)
- Check data loading (verify timepoint_ids are present)

**Validation Loss Increases**:
- Add regularization: `weight_decay=0.05`
- Reduce learning rate after plateau
- Check for overfitting (compare train/val curves)

### If Evaluation Shows Poor Performance (NN distance > 0.3)

**Short-term fixes**:
1. Train longer (`max_steps=30000`)
2. Adjust architecture (`hidden_dim=768`, `cell_set_len=384`)
3. Try different loss function (`loss=se` for combined sinkhorn+energy)

**Alternative strategy**:
- **Strategy 3**: Fine-tune ST model with LoRA on burn/sham data
  - See plan document for implementation details
  - More direct adaptation to wound healing biology

---

## 📚 Key Decisions & Rationale

### Why Baseline SE-600M Instead of LoRA?

| Aspect | LoRA Embeddings | Baseline SE-600M | Winner |
|--------|-----------------|------------------|---------|
| **Cell Type Accuracy** | ~75% | 96% | ✅ Baseline |
| **Temporal Separation** | ✅ Strong | Moderate | LoRA |
| **Condition Separation** | ✅ Strong | Moderate | LoRA |
| **ST Model Requirement** | Cell type structure critical | Cell type structure critical | ✅ Baseline |
| **Perturbation Prediction** | Mixed cell types → poor cell-specific responses | Preserved cell types → good cell-specific responses | ✅ Baseline |

**Decision**: Use baseline SE-600M + add temporal/condition as metadata covariates in ST training.

**Rationale**:
- ST model learns cell-type-specific perturbation responses
- Requires strong cell type clustering (baseline: 96% vs LoRA: 75%)
- Temporal/condition information can be added as embeddings during ST training
- LoRA optimized for wrong objective (temporal signal at cost of cell types)

### Why Arc Institute State Transition (not CPA or scGPT)?

- User explicitly specified: "we only use Arc Institutes State Transition and State Embedding model"
- State Transition is purpose-built for perturbation prediction
- Handles distributional matching between cell populations
- Supports transformer backbones (GPT2/Llama) for learning compositional effects
- Already has infrastructure for batch/covariate embeddings

---

## 🚀 Next Steps After Evaluation

### If Successful
1. **Biological Analysis**:
   - Identify genes driving predicted burn responses
   - Compare to known wound healing markers
   - Analyze macrophage polarization (M1/M2)
   - Validate temporal progression

2. **Presentation Preparation**:
   - Create slides for qualifying exam
   - Highlight key metrics and visualizations
   - Prepare biological interpretation
   - Document limitations and future work

3. **Publication**:
   - Write methods section
   - Create figure panels
   - Compare to existing methods
   - Discuss novel insights

### If Needs Improvement
1. **Extended Training**: More steps, adjusted hyperparameters
2. **Architecture Tuning**: Different hidden_dim, cell_set_len
3. **Loss Function**: Try sinkhorn+energy combination
4. **Strategy 3**: Fine-tune ST model with LoRA (see plan document)

---

## 📖 References

- **Arc Institute State Model**: [GitHub](https://github.com/ArcInstitute/state)
- **State Embedding Paper**: [HuggingFace](https://huggingface.co/ArcInstitute/state-embedding)
- **LoRA Paper**: Low-Rank Adaptation of Large Language Models
- **Compositional Perturbation Autoencoder (CPA)**: Lotfollahi et al.

---

## ✨ Summary

Phase 3 is **ready for training**! All code modifications are complete, configuration files are created, and comprehensive notebooks guide the entire workflow from data preparation through evaluation.

**Time Commitment**:
- Data prep: 5 minutes
- Training: 4-6 hours (unattended)
- Evaluation: 10-15 minutes

**Expected Outcome**: Successful perturbation prediction model for burn/sham wound healing with quantitative metrics and biological interpretability.

---

**Created**: 2026-01-05
**Last Updated**: 2026-01-05
**Status**: ✅ Complete - Ready for Training
