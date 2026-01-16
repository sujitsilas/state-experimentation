#!/bin/bash

# ST-LoRA Training Script for Burn/Sham Wound Healing
# LoRA only, no mHC, no velocity alignment (baseline fine-tuning)

cd /home/scumpia-mrl/Desktop/Sujit/Projects/state-experimentation

# Use single-line format to avoid Hydra parsing issues with multiline commands
state tx train data.kwargs.toml_config_path=experiments/st_fine_tuning/configs/burn_sham_st_training.toml data.kwargs.embed_key=X_state_2000 data.kwargs.pert_col=condition data.kwargs.control_pert=Sham data.kwargs.cell_type_key=cell_types_simple_short data.kwargs.batch_col=mouse_id data.kwargs.num_workers=0 +model.kwargs.lora.enable=true +model.kwargs.lora.r=16 +model.kwargs.lora.alpha=32 +model.kwargs.use_mhc=false +model.kwargs.use_lor2c_mhc=false +model.kwargs.use_velocity_alignment=false training.max_steps=5000 training.val_freq=500 training.lr=5e-5 training.batch_size=8 training.devices=1 training.strategy=auto output_dir=/home/scumpia-mrl/state_models/st_lora name=st_lora_burn_sham

# Training details:
# - 5,000 steps = ~313 epochs (16 batches/epoch)
# - Validation every 500 steps (~31 epochs)
# - Single GPU training
# - LoRA: r=16, alpha=32 (~1.7% trainable params)
# - Expected time: ~1.5-2 hours on RTX 5000 Ada
