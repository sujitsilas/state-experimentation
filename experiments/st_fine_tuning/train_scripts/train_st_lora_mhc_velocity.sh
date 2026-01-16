#!/bin/bash

# ST-LoRA-mHC-Velocity Training Script for Burn/Sham Wound Healing
# Full approach: LoRA + LoR2C-mHC + Velocity Alignment
#
# Combines three techniques:
# - LoRA: Parameter-efficient fine-tuning of attention weights
# - LoR2C-mHC: Low-rank residual connections with doubly stochastic constraint
# - Velocity Alignment: Physics-informed loss using RNA velocity
#
# References:
# - LoR2C paper: https://arxiv.org/abs/2503.00572
# - mHC paper: https://arxiv.org/abs/2512.24880 (DeepSeek)

cd /home/scumpia-mrl/Desktop/Sujit/Projects/state-experimentation

# Use single-line format to avoid Hydra parsing issues with multiline commands
# NOTE: Uses burn_sham_st_training_velocity.toml which points to burn_sham_with_velocity.h5ad
state tx train data.kwargs.toml_config_path=experiments/st_fine_tuning/configs/burn_sham_st_training_velocity.toml data.kwargs.embed_key=X_state_2000 data.kwargs.pert_col=condition data.kwargs.control_pert=Sham data.kwargs.cell_type_key=cell_types_simple_short data.kwargs.batch_col=mouse_id data.kwargs.num_workers=0 +model.kwargs.lora.enable=true +model.kwargs.lora.r=16 +model.kwargs.lora.alpha=32 +model.kwargs.use_mhc=false +model.kwargs.use_lor2c_mhc=true +model.kwargs.lor2c_mhc.rank=16 +model.kwargs.lor2c_mhc.sinkhorn_iters=20 +model.kwargs.lor2c_mhc.alpha=32.0 +model.kwargs.lor2c_mhc.share_A=false +model.kwargs.use_velocity_alignment=true +model.kwargs.velocity_lambda=0.1 +model.kwargs.velocity_beta=1.0 +model.kwargs.velocity_warmup_steps=1000 +model.kwargs.velocity_min_confidence=0.0 training.max_steps=5000 training.val_freq=500 training.lr=5e-5 training.batch_size=8 training.devices=1 training.strategy=auto output_dir=/home/scumpia-mrl/state_models/st_lora_mhc_velocity name=st_lora_mhc_velocity_burn_sham

# Training details:
# - 5,000 steps = ~313 epochs (16 batches/epoch)
# - Validation every 500 steps (~31 epochs)
# - Single GPU training
# - LoRA: r=16, alpha=32 (~1.7% trainable params)
# - LoR2C-mHC: rank=16, alpha=32, 20 Sinkhorn iterations
# - Velocity alignment: lambda=0.1, beta=1.0, warmup=1000 steps
#   - Uses velocity_latent from scVelo projected to State embedding space
#   - Aligns predicted state transitions with RNA velocity directions
# - Expected: Best performance (combines all benefits)
# - Expected time: ~2.5-3 hours on RTX 5000 Ada
#
# PREREQUISITE: Run prepare_velocity_data.py first to compute velocity_latent!
