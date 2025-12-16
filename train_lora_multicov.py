"""
Training script for LoRA-based covariate-conditioned State Embedding Model

Usage:
    python train_lora_multicov.py --config configs/lora_multicov_config.yaml
"""

import argparse
import os
from pathlib import Path
import yaml
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.state.emb.nn.lora_covariate_model import LoRACovariateStateModel


class CovariateAnnDataset(Dataset):
    """
    Dataset for loading single-cell data with covariate information.
    Loads ESM2 gene embeddings for expressed genes.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        covariate_columns: list,
        protein_embeds: dict,
        max_seq_len: int = 512,
        cell_type_col: str = None,
    ):
        self.adata = adata
        self.covariate_columns = covariate_columns
        self.protein_embeds = protein_embeds
        self.max_seq_len = max_seq_len
        self.cell_type_col = cell_type_col

        # Build covariate mappings
        self.covariate_mappings = {}
        for col in covariate_columns:
            if col in adata.obs.columns:
                unique_values = adata.obs[col].unique()
                self.covariate_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
                print(f"Covariate '{col}': {len(unique_values)} categories")
                print(f"  Values: {unique_values}")
            else:
                print(f"WARNING: Covariate column '{col}' not found in adata.obs")

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing:
        - cell_sentences: (max_seq_len, 5120) ESM2 gene embeddings
        - mask: (max_seq_len,) boolean mask (True = valid gene, False = padding)
        - counts: (max_seq_len,) gene counts
        - covariates: dict of covariate category indices
        """
        cell = self.adata[idx]

        # Get gene counts
        counts = cell.X.toarray().flatten() if hasattr(cell.X, "toarray") else cell.X.flatten()

        # Get top N expressed genes (sorted by count)
        gene_indices = np.argsort(counts)[::-1][:self.max_seq_len]
        expressed_genes = cell.var_names[gene_indices].tolist()
        gene_counts = counts[gene_indices]

        # Lookup ESM2 embeddings for these genes
        gene_embeddings = []
        for gene in expressed_genes:
            if gene in self.protein_embeds:
                gene_embeddings.append(self.protein_embeds[gene])
            else:
                # Zero embedding for missing genes
                gene_embeddings.append(torch.zeros(5120))

        gene_embeddings = torch.stack(gene_embeddings)  # (seq_len, 5120)

        # Create mask (True = valid gene, False = padding)
        mask = torch.ones(len(expressed_genes), dtype=torch.bool)

        # Pad to max_seq_len
        if len(expressed_genes) < self.max_seq_len:
            padding = self.max_seq_len - len(expressed_genes)
            gene_embeddings = torch.cat([
                gene_embeddings,
                torch.zeros(padding, 5120)
            ])
            mask = torch.cat([mask, torch.zeros(padding, dtype=torch.bool)])
            gene_counts = np.concatenate([gene_counts, np.zeros(padding)])

        # Extract covariates
        covariates = {}
        for col in self.covariate_columns:
            if col in cell.obs.columns:
                val = cell.obs[col].values[0]
                covariates[col] = self.covariate_mappings[col][val]
            else:
                covariates[col] = 0  # Default

        return {
            "cell_sentences": gene_embeddings,  # (max_seq_len, 5120)
            "mask": mask,  # (max_seq_len,)
            "counts": torch.tensor(gene_counts, dtype=torch.float32),  # (max_seq_len,)
            "covariates": covariates,  # Dict[str, int]
        }


def collate_fn(batch):
    """Custom collate function to batch gene sequences with covariates."""
    return {
        "cell_sentences": torch.stack([item["cell_sentences"] for item in batch]),  # (B, max_seq_len, 5120)
        "mask": torch.stack([item["mask"] for item in batch]),  # (B, max_seq_len)
        "counts": torch.stack([item["counts"] for item in batch]),  # (B, max_seq_len)
        "covariates": {
            key: torch.tensor([item["covariates"][key] for item in batch])
            for key in batch[0]["covariates"].keys()
        },  # Dict[str, (B,)]
    }


class LoRADataModule(L.LightningDataModule):
    """Lightning DataModule for LoRA fine-tuning."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.adata = None

    def setup(self, stage=None):
        """Load and split dataset."""
        print(f"Loading data from {self.config['data']['h5ad_path']}")
        self.adata = ad.read_h5ad(self.config["data"]["h5ad_path"])

        print(f"Dataset: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")

        # Load ESM2 protein embeddings
        protein_embeds_path = self.config["base_config"].replace("config.yaml", "protein_embeddings.pt")
        print(f"Loading protein embeddings from {protein_embeds_path}")
        self.protein_embeds = torch.load(protein_embeds_path, weights_only=False)
        print(f"✓ Loaded {len(self.protein_embeds)} gene embeddings")

        # Create dataset with protein embeddings
        self.train_dataset = CovariateAnnDataset(
            self.adata,
            covariate_columns=self.config["data"]["covariate_columns"],
            protein_embeds=self.protein_embeds,
            max_seq_len=512,
            cell_type_col=self.config["data"].get("cell_type_col"),
        )

        # Use same dataset for validation (TODO: implement proper split)
        self.val_dataset = self.train_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Train LoRA multi-covariate State model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("LoRA Multi-Covariate Training Configuration")
    print("=" * 80)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 80)

    # Create output directory
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    print("\nInitializing LoRA model...")
    model = LoRACovariateStateModel(
        base_checkpoint_path=config["base_checkpoint"],
        base_config_path=config["base_config"],
        covariate_config=config["covariates"],
        lora_config=config["lora"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
    )

    # Print trainable parameters
    model.print_trainable_parameters()

    # Initialize data module
    print("\nInitializing data module...")
    data_module = LoRADataModule(config)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor=config["training"]["monitor"],
        mode=config["training"]["mode"],
        save_top_k=config["training"]["save_top_k"],
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=config["output"]["experiment_name"],
    )

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="gpu",
        devices=config["training"]["devices"],
        strategy=config["training"]["strategy"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        val_check_interval=config["training"]["val_check_interval"],
        log_every_n_steps=config["training"]["log_every_n_steps"],
        enable_progress_bar=config["training"]["enable_progress_bar"],
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, data_module)

    print(f"\nTraining complete! Checkpoints saved to {output_dir / 'checkpoints'}")

    # Extract embeddings if configured
    if config["output"].get("save_embeddings", False):
        print("\nExtracting covariate-conditioned embeddings...")
        # TODO: Implement embedding extraction
        print("Embedding extraction not yet implemented")


if __name__ == "__main__":
    main()
