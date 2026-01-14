#!/bin/bash

# ST-LoRA-mHC-Velocity Training Script for Burn/Sham Wound Healing
# Full approach: LoRA + mHC + Velocity Alignment
# NOTE: Requires RNA velocity data computed with scVelo in adata.obsm['velocity_latent']

cd /home/scumpia-mrl/Desktop/Sujit/Projects/state-experimentation

state tx train \
  data.kwargs.toml_config_path=experiments/st_fine_tuning/configs/burn_sham_st_training.toml \
  data.kwargs.embed_key=X_state_2000 \
  data.kwargs.pert_col=condition \
  data.kwargs.control_pert=Sham \
  data.kwargs.cell_type_key=cell_types_simple_short \
  data.kwargs.batch_col=mouse_id \
  data.kwargs.num_workers=0 \
  +model.kwargs.lora.enable=true \
  +model.kwargs.lora.r=16 \
  +model.kwargs.lora.alpha=32 \
  +model.kwargs.use_mhc=true \
  +model.kwargs.mhc.sinkhorn_iters=10 \
  +model.kwargs.use_velocity_alignment=true \
  +model.kwargs.velocity_lambda=0.1 \
  +model.kwargs.velocity_beta=1.0 \
  +model.kwargs.velocity_warmup_steps=1000 \
  +model.kwargs.velocity_min_confidence=0.0 \
  training.max_steps=50000 \
  training.val_freq=2000 \
  training.lr=5e-5 \
  training.batch_size=8 \
  training.devices=1 \
  training.strategy=auto \
  output_dir=/home/scumpia-mrl/state_models/st_lora_mhc_velocity \
  name=st_lora_mhc_velocity_burn_sham

# Training details:
# - 50,000 steps = ~7 epochs
# - Validation every 2,000 steps
# - Single GPU training
# - LoRA: r=16, alpha=32 (~1.7% trainable params)
# - mHC: Sinkhorn-Knopp with 10 iterations (gradient stabilization)
# - Velocity alignment: lambda=0.1, beta=1.0, warmup=1000 steps
# - Expected: Best performance (combines all benefits)
# - Expected time: ~5-6 hours on RTX 5000 Ada
#
# ⚠️  PREREQUISITE: RNA velocity data must be computed first!
# Run scVelo on data and store velocity in adata.obsm['velocity_latent']
