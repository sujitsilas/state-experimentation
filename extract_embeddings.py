"""
Extract covariate-conditioned embeddings from trained LoRA model checkpoint.

Usage:
    python extract_embeddings.py --config configs/lora_multicov_config.yaml --checkpoint /path/to/checkpoint.ckpt
"""

import argparse
from pathlib import Path
import yaml
import torch
import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.state.emb.nn.lora_covariate_model import LoRACovariateStateModel


class CovariateAnnDataset(Dataset):
    """
    Dataset for loading single-cell data with covariate information.
    Loads ESM2 gene embeddings for expressed genes.
    """

    def __init__(
        self,
        adata,
        covariate_columns,
        protein_embeds,
        max_seq_len=513,
        cell_type_col=None,
    ):
        self.adata = adata
        self.covariate_columns = covariate_columns
        self.protein_embeds = protein_embeds
        self.max_seq_len = max_seq_len
        self.cell_type_col = cell_type_col

        # Encode covariates
        self.covariate_encoders = {}
        for col in covariate_columns:
            unique_vals = adata.obs[col].unique()
            self.covariate_encoders[col] = {val: idx for idx, val in enumerate(unique_vals)}

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        # Get cell data
        cell = self.adata[idx]

        # Get counts (sparse or dense)
        if hasattr(cell.X, "toarray"):
            counts = cell.X.toarray().flatten()
        else:
            counts = cell.X.flatten()

        # Get top N expressed genes (sorted by count)
        gene_indices = np.argsort(counts)[::-1][:self.max_seq_len]
        expressed_genes = cell.var_names[gene_indices].tolist()
        gene_counts = counts[gene_indices]

        # Lookup ESM2 embeddings for these genes
        gene_embeddings = []
        for gene in expressed_genes:
            if gene in self.protein_embeds:
                emb = self.protein_embeds[gene]
                # Convert to numpy if tensor
                if hasattr(emb, 'numpy'):
                    emb = emb.numpy()
                gene_embeddings.append(emb)
            else:
                # Zero embedding for missing genes
                gene_embeddings.append(np.zeros(5120))

        gene_embeddings = np.stack(gene_embeddings)  # (seq_len, 5120)

        # Encode covariates
        covariates = {}
        for col in self.covariate_columns:
            val = cell.obs[col].iloc[0]
            covariates[col] = self.covariate_encoders[col][val]

        return {
            "cell_sentences": gene_embeddings,
            "counts": gene_counts,
            "covariates": covariates,
            "cell_type": cell.obs[self.cell_type_col].iloc[0] if self.cell_type_col else None,
        }

    def collate_fn(self, batch):
        """Collate batch with padding for variable length sequences."""
        # Find max sequence length in batch
        max_len = max(item["cell_sentences"].shape[0] for item in batch)

        # Pad sequences
        padded_sentences = []
        padded_counts = []
        masks = []

        for item in batch:
            seq_len = item["cell_sentences"].shape[0]
            pad_len = max_len - seq_len

            # Pad gene embeddings
            padded_sent = np.pad(
                item["cell_sentences"],
                ((0, pad_len), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            padded_sentences.append(padded_sent)

            # Pad counts
            padded_count = np.pad(item["counts"], (0, pad_len), mode="constant", constant_values=0)
            padded_counts.append(padded_count)

            # Create mask (1 for real tokens, 0 for padding)
            mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)])
            masks.append(mask)

        # Stack into tensors
        batch_dict = {
            "cell_sentences": torch.FloatTensor(np.stack(padded_sentences)),
            "counts": torch.FloatTensor(np.stack(padded_counts)),
            "mask": torch.BoolTensor(np.stack(masks)),
            "covariates": {
                key: torch.LongTensor([item["covariates"][key] for item in batch])
                for key in batch[0]["covariates"].keys()
            },
        }

        if batch[0]["cell_type"] is not None:
            batch_dict["cell_type"] = [item["cell_type"] for item in batch]

        return batch_dict


def extract_embeddings(checkpoint_path, config_path, output_path):
    """Extract embeddings from trained checkpoint."""
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("LoRA Embedding Extraction")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print("=" * 80)

    # Load model from checkpoint
    print("\nLoading model from checkpoint...")
    model = LoRACovariateStateModel.load_from_checkpoint(
        checkpoint_path,
        base_checkpoint_path=config["base_checkpoint"],
        base_config_path=config["base_config"],
        covariate_config=config["covariates"],
        lora_config=config["lora"],
    )
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Load data
    print(f"\nLoading data from {config['data']['h5ad_path']}...")
    adata = ad.read_h5ad(config["data"]["h5ad_path"])
    print(f"Dataset: {adata.n_obs} cells x {adata.n_vars} genes")

    # Load protein embeddings
    protein_embeds_path = Path(config["base_checkpoint"]).parent / "protein_embeddings.pt"
    print(f"Loading protein embeddings from {protein_embeds_path}...")
    protein_embeds = torch.load(protein_embeds_path, map_location="cpu")
    print(f"✓ Loaded {len(protein_embeds)} gene embeddings")

    # Create dataset
    dataset = CovariateAnnDataset(
        adata,
        covariate_columns=config["data"]["covariate_columns"],
        protein_embeds=protein_embeds,
        max_seq_len=513,
        cell_type_col=config["data"].get("cell_type_col"),
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )

    # Extract embeddings
    embeddings_list = []

    print(f"\nExtracting embeddings for {len(dataset)} cells...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            # Move batch to device (including nested dict)
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                elif isinstance(v, dict):
                    # Handle covariates dict
                    batch_device[k] = {key: val.to(device) for key, val in v.items()}
                else:
                    batch_device[k] = v

            # Forward pass to get conditioned embeddings
            _, conditioned_embedding, _ = model(
                batch_device["cell_sentences"],
                batch_device["mask"],
                batch_device["covariates"],
                counts=batch_device.get("counts"),
                dataset_nums=None,
            )

            # Move to CPU and store
            embeddings_list.append(conditioned_embedding.cpu().numpy())

    # Concatenate all embeddings
    all_embeddings = np.concatenate(embeddings_list, axis=0)

    print(f"\n✓ Extracted embeddings shape: {all_embeddings.shape}")

    # Add embeddings to AnnData
    adata.obsm["X_lora_conditioned"] = all_embeddings

    # Save to output path
    print(f"\nSaving embeddings to {output_path}...")
    adata.write_h5ad(output_path)
    print(f"✓ Saved {len(adata)} cells with conditioned embeddings")
    print(f"\nEmbedding key in .obsm: 'X_lora_conditioned'")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from trained LoRA checkpoint")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training config YAML file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for h5ad with embeddings (default: from config)",
    )

    args = parser.parse_args()

    # Load config to get output path if not provided
    if args.output is None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        output_path = config["output"]["embeddings_output"]
    else:
        output_path = args.output

    extract_embeddings(
        checkpoint_path=args.checkpoint, config_path=args.config, output_path=output_path
    )


if __name__ == "__main__":
    main()
