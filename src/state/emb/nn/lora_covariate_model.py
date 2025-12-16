"""
LoRA-based Covariate-Conditioned State Embedding Model

This module implements a parameter-efficient fine-tuning approach for the SE-600M model
using LoRA (Low-Rank Adaptation) adapters with multi-covariate conditioning.

Key Features:
- Freezes base SE-600M model weights
- Adds trainable LoRA adapters to attention layers
- Implements generic multi-covariate conditioning (categorical + continuous)
- YAML-configurable covariate specifications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from peft import LoraConfig, get_peft_model
import lightning as L
from omegaconf import OmegaConf

from .model import StateEmbeddingModel


class ContinuousEncoder(nn.Module):
    """Encodes continuous covariates (e.g., time_days) using an MLP."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, output_dim: int = 128, activation: str = "silu"):
        super().__init__()

        act_fn = {
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
        }[activation.lower()]

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size,) continuous values
        Returns:
            (batch_size, output_dim) embeddings
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.encoder(x)


class CombinationMLP(nn.Module):
    """Combines multiple covariate embeddings using an MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.1,
        use_norm: str = "layer"
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))

            if use_norm == "layer":
                layers.append(nn.LayerNorm(hidden_dim))
            elif use_norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) concatenated covariate embeddings
        Returns:
            (batch_size, output_dim) combined embedding
        """
        return self.mlp(x)


class MultiCovariateEncoder(nn.Module):
    """
    Generic multi-covariate encoder supporting arbitrary categorical and continuous covariates.

    Example config:
    {
        "covariates": [
            {"name": "timepoint", "type": "categorical", "num_categories": 3, "embed_dim": 128},
            {"name": "condition", "type": "categorical", "num_categories": 2, "embed_dim": 128},
            {"name": "time_days", "type": "continuous", "continuous_encoder": {"hidden_dim": 64, "output_dim": 128}}
        ],
        "combination": {
            "method": "concat_mlp",
            "mlp_hidden_dims": [512, 256],
            "mlp_output_dim": 128,
            "dropout_rate": 0.1,
            "use_norm": "layer"
        }
    }
    """

    def __init__(
        self,
        covariate_configs: List[Dict],
        combination_config: Dict,
        output_dim: int = 128
    ):
        super().__init__()

        self.covariate_names = [cov["name"] for cov in covariate_configs]
        self.covariate_types = {cov["name"]: cov["type"] for cov in covariate_configs}

        # Create encoders for each covariate
        self.encoders = nn.ModuleDict()
        total_embed_dim = 0

        for cov in covariate_configs:
            name = cov["name"]
            cov_type = cov["type"]

            if cov_type == "categorical":
                num_categories = cov["num_categories"]
                embed_dim = cov.get("embed_dim", 128)
                self.encoders[name] = nn.Embedding(num_categories, embed_dim)
                total_embed_dim += embed_dim

            elif cov_type == "continuous":
                cont_config = cov.get("continuous_encoder", {})
                hidden_dim = cont_config.get("hidden_dim", 64)
                embed_dim = cont_config.get("output_dim", 128)
                activation = cont_config.get("activation", "silu")

                self.encoders[name] = ContinuousEncoder(
                    input_dim=1,
                    hidden_dim=hidden_dim,
                    output_dim=embed_dim,
                    activation=activation
                )
                total_embed_dim += embed_dim
            else:
                raise ValueError(f"Unsupported covariate type: {cov_type}")

        # Create combination MLP
        combination_method = combination_config.get("method", "concat_mlp")

        if combination_method == "concat_mlp":
            hidden_dims = combination_config.get("mlp_hidden_dims", [512, 256])
            dropout_rate = combination_config.get("dropout_rate", 0.1)
            use_norm = combination_config.get("use_norm", "layer")
            mlp_output_dim = combination_config.get("mlp_output_dim", output_dim)

            self.combination_mlp = CombinationMLP(
                input_dim=total_embed_dim,
                hidden_dims=hidden_dims,
                output_dim=mlp_output_dim,
                dropout_rate=dropout_rate,
                use_norm=use_norm
            )
        elif combination_method == "sum":
            # Simple summation (all embeddings must have same dim)
            self.combination_mlp = None
        else:
            raise ValueError(f"Unsupported combination method: {combination_method}")

    def forward(self, covariate_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            covariate_dict: Dictionary mapping covariate names to tensors
                - Categorical: (batch_size,) long tensors with category indices
                - Continuous: (batch_size,) float tensors with values
        Returns:
            (batch_size, output_dim) combined covariate embedding
        """
        embeddings = []

        for name in self.covariate_names:
            if name not in covariate_dict:
                raise ValueError(f"Covariate '{name}' not found in input dict")

            cov_value = covariate_dict[name]

            if self.covariate_types[name] == "categorical":
                # Ensure long type for embedding lookup
                emb = self.encoders[name](cov_value.long())
            else:  # continuous
                emb = self.encoders[name](cov_value.float())

            embeddings.append(emb)

        # Concatenate all embeddings
        combined = torch.cat(embeddings, dim=1)

        # Apply combination MLP if configured
        if self.combination_mlp is not None:
            combined = self.combination_mlp(combined)

        return combined


class LoRACovariateStateModel(L.LightningModule):
    """
    LoRA-based covariate-conditioned State Embedding Model.

    This model:
    1. Loads a pretrained SE-600M checkpoint
    2. Freezes all base model parameters
    3. Adds LoRA adapters to transformer attention layers
    4. Adds covariate embedding layers
    5. Conditions embeddings on user-specified covariates
    """

    def __init__(
        self,
        base_checkpoint_path: str,
        base_config_path: str,
        covariate_config: Dict,
        lora_config: Optional[Dict] = None,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

        # Load config from YAML file
        print(f"Loading config from {base_config_path}")
        base_cfg = OmegaConf.load(base_config_path)

        # Instantiate base model
        print(f"Creating SE-600M model architecture...")
        self.base_model = StateEmbeddingModel(
            token_dim=base_cfg.tokenizer.token_dim,
            d_model=base_cfg.model.emsize,
            nhead=base_cfg.model.nhead,
            d_hid=base_cfg.model.d_hid,
            nlayers=base_cfg.model.nlayers,
            output_dim=base_cfg.model.output_dim,
            dropout=base_cfg.model.dropout,
            cfg=base_cfg,
        )

        # Load pretrained weights
        print(f"Loading pretrained weights from {base_checkpoint_path}")
        checkpoint = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)

        if "state_dict" in checkpoint:
            self.base_model.load_state_dict(checkpoint["state_dict"], strict=False)
            print("✓ Loaded pretrained weights from checkpoint")
        else:
            print("WARNING: No 'state_dict' found in checkpoint, model will use random initialization")

        # Load protein embeddings if available
        if "protein_embeds_dict" in checkpoint:
            self.base_model.protein_embeds = checkpoint["protein_embeds_dict"]
            print("✓ Loaded protein embeddings from checkpoint")
        else:
            print("WARNING: No protein embeddings in checkpoint, will need to load separately")

        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        print("Base SE-600M model loaded and frozen")

        # Configure LoRA
        if lora_config is None:
            lora_config = {
                "r": 16,  # LoRA rank
                "lora_alpha": 32,  # LoRA scaling factor
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],  # Apply LoRA to Q, V projections
                "bias": "none",
            }

        # Apply LoRA to transformer layers
        peft_config = LoraConfig(**lora_config)
        self.base_model.transformer_encoder = get_peft_model(
            self.base_model.transformer_encoder, peft_config
        )

        print(f"LoRA adapters added: r={lora_config['r']}, alpha={lora_config['lora_alpha']}")

        # Create multi-covariate encoder
        self.covariate_encoder = MultiCovariateEncoder(
            covariate_configs=covariate_config["covariates"],
            combination_config=covariate_config["combination"],
            output_dim=base_cfg.model.output_dim  # Match base model output dim
        )

        print(f"Covariate encoder created with {len(covariate_config['covariates'])} covariates")

        # Conditioning projection: combines base embedding + covariate embedding
        self.conditioning_projection = nn.Sequential(
            nn.Linear(base_cfg.model.output_dim * 2, base_cfg.model.output_dim),
            nn.LayerNorm(base_cfg.model.output_dim),
            nn.SiLU(),
        )

        self.output_dim = base_cfg.model.output_dim
        self.base_cfg = base_cfg

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor,
        covariate_dict: Dict[str, torch.Tensor],
        counts: Optional[torch.Tensor] = None,
        dataset_nums: Optional[torch.Tensor] = None
    ):
        """
        Forward pass with covariate conditioning.

        Args:
            src: (batch_size, seq_len, token_dim) cell sentence embeddings
            mask: (batch_size, seq_len) padding mask
            covariate_dict: Dictionary of covariate tensors
            counts: Optional gene counts
            dataset_nums: Optional dataset IDs

        Returns:
            gene_output: (batch_size, seq_len, output_dim) token-level outputs
            embedding: (batch_size, output_dim) cell-level embedding (CLS token)
            covariate_embedding: (batch_size, output_dim) covariate embedding
        """
        # Get base embedding with LoRA adapters active (gradients enabled!)
        gene_output_base, embedding_base, dataset_emb = self.base_model(
            src, mask, counts=counts, dataset_nums=dataset_nums
        )

        # Get covariate embedding
        covariate_embedding = self.covariate_encoder(covariate_dict)

        # Combine base embedding + covariate embedding
        combined = torch.cat([embedding_base, covariate_embedding], dim=1)
        conditioned_embedding = self.conditioning_projection(combined)

        # L2 normalize
        conditioned_embedding = nn.functional.normalize(conditioned_embedding, dim=1)

        return gene_output_base, conditioned_embedding, covariate_embedding

    def training_step(self, batch, batch_idx):
        """
        Training step using reconstruction loss.
        Predicts gene presence/absence using SE-600M's binary decoder.
        """
        # Forward pass through SE-600M + LoRA adapters
        gene_output, conditioned_embedding, cov_emb = self.forward(
            batch["cell_sentences"],
            batch["mask"],
            batch["covariates"],
            batch.get("counts"),
            batch.get("dataset_nums")
        )

        # Prepare task: predict gene presence/absence from embedding
        X = self.base_model.gene_embedding_layer(batch["cell_sentences"])  # Gene embeddings
        Y = (batch["counts"] > 0).float()  # Binary: 1 if gene expressed, 0 otherwise

        # Expand conditioned embedding to match sequence length
        z = conditioned_embedding.unsqueeze(1).repeat(1, X.shape[1], 1)

        # Optionally add RDA (relative dataset abundance) if using counts
        if self.base_cfg.model.rda and batch.get("counts") is not None:
            mu = torch.nanmean(
                batch["counts"].float().masked_fill(batch["counts"] == 0, float("nan")),
                dim=1, keepdim=True
            )
            mu = torch.nan_to_num(mu, nan=0.0)
            mu_expanded = mu.unsqueeze(2).repeat(1, X.shape[1], 1)
            combine = torch.cat([X, z, mu_expanded], dim=2)
        else:
            combine = torch.cat([X, z], dim=2)

        # Forward through binary decoder
        predictions = self.base_model.binary_decoder(combine)  # (B, seq_len, 1)

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(
            predictions.squeeze(-1),
            Y,
            reduction='mean'
        )

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/accuracy", ((predictions.squeeze(-1) > 0) == Y).float().mean())

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with same reconstruction loss."""
        # Forward pass through SE-600M + LoRA adapters
        gene_output, conditioned_embedding, cov_emb = self.forward(
            batch["cell_sentences"],
            batch["mask"],
            batch["covariates"],
            batch.get("counts"),
            batch.get("dataset_nums")
        )

        # Prepare task: predict gene presence/absence from embedding
        X = self.base_model.gene_embedding_layer(batch["cell_sentences"])
        Y = (batch["counts"] > 0).float()

        # Expand conditioned embedding to match sequence length
        z = conditioned_embedding.unsqueeze(1).repeat(1, X.shape[1], 1)

        # Optionally add RDA if using counts
        if self.base_cfg.model.rda and batch.get("counts") is not None:
            mu = torch.nanmean(
                batch["counts"].float().masked_fill(batch["counts"] == 0, float("nan")),
                dim=1, keepdim=True
            )
            mu = torch.nan_to_num(mu, nan=0.0)
            mu_expanded = mu.unsqueeze(2).repeat(1, X.shape[1], 1)
            combine = torch.cat([X, z, mu_expanded], dim=2)
        else:
            combine = torch.cat([X, z], dim=2)

        # Forward through binary decoder
        predictions = self.base_model.binary_decoder(combine)

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(
            predictions.squeeze(-1),
            Y,
            reduction='mean'
        )

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/accuracy", ((predictions.squeeze(-1) > 0) == Y).float().mean())

        return loss

    def configure_optimizers(self):
        """Configure optimizer with warmup + cosine decay."""
        # Only optimize LoRA parameters + covariate encoder + conditioning projection
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # TODO: Add warmup + cosine annealing scheduler

        return optimizer

    def print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_params = 0

        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Trainable params: {trainable_params:,} | "
              f"All params: {all_params:,} | "
              f"Trainable%: {100 * trainable_params / all_params:.2f}%")
