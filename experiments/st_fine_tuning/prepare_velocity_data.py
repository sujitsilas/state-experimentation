#!/usr/bin/env python3
"""
Prepare Velocity Data for ST-LoRA-mHC-Velocity Training

This script takes the burn_sham_with_velocity.h5ad file (with scVelo velocity)
and adds the necessary fields for velocity-aligned training:

1. X_state_2000: State embeddings (need to be computed if not present)
2. velocity_latent: RNA velocity projected to State embedding space
3. velocity_confidence: Confidence score for velocity estimates

The velocity_latent is computed by projecting the gene-space velocity
to the State embedding space using the State model's encoder.

Usage:
    python prepare_velocity_data.py
"""

import sys
from pathlib import Path

import numpy as np
import anndata as ad
import scanpy as sc
from scipy import sparse

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def compute_velocity_latent_simple(adata: ad.AnnData) -> np.ndarray:
    """
    Compute velocity in latent space using a simple PCA-based projection.

    For a more accurate projection, you would use the State model encoder,
    but this provides a reasonable approximation for initial experiments.

    The idea: velocity in gene space can be approximated in latent space
    by projecting through the same transformation used for gene expression.
    """
    print("Computing velocity_latent using PCA projection...")

    # Get velocity from layers
    if 'velocity' not in adata.layers:
        raise ValueError("No 'velocity' layer found in AnnData. Run scVelo first.")

    velocity = adata.layers['velocity']
    if sparse.issparse(velocity):
        velocity = velocity.toarray()

    # Handle NaN values (velocity is undefined for some genes)
    velocity = np.nan_to_num(velocity, nan=0.0)

    # Get spliced counts (what State embeddings are based on)
    if 'spliced' in adata.layers:
        X = adata.layers['spliced']
    elif 'Ms' in adata.layers:  # scVelo imputed spliced
        X = adata.layers['Ms']
    else:
        X = adata.X

    if sparse.issparse(X):
        X = X.toarray()

    # Fit PCA on expression data
    from sklearn.decomposition import PCA
    n_components = min(2000, X.shape[1], X.shape[0])

    pca = PCA(n_components=n_components)
    pca.fit(X)

    # Project velocity through PCA
    # velocity_latent = velocity @ pca.components_.T
    velocity_latent = pca.transform(velocity + X) - pca.transform(X)

    # Normalize to similar scale as State embeddings
    if 'X_state_2000' in adata.obsm:
        state_scale = np.std(adata.obsm['X_state_2000'])
        velocity_scale = np.std(velocity_latent)
        if velocity_scale > 0:
            velocity_latent = velocity_latent * (state_scale / velocity_scale) * 0.1

    print(f"  velocity_latent shape: {velocity_latent.shape}")
    print(f"  velocity_latent stats: mean={np.mean(velocity_latent):.4f}, std={np.std(velocity_latent):.4f}")

    return velocity_latent.astype(np.float32)


def compute_velocity_confidence(adata: ad.AnnData) -> np.ndarray:
    """
    Compute confidence scores for velocity estimates.

    Uses velocity_self_transition as a proxy for confidence:
    - Higher self-transition = cell is stable (high confidence velocity)
    - Lower self-transition = cell is transitioning (velocity is informative)

    We invert this so that transitioning cells have higher weight.
    """
    print("Computing velocity_confidence...")

    if 'velocity_self_transition' in adata.obs:
        # Self-transition probability: low = transitioning, high = stable
        self_trans = adata.obs['velocity_self_transition'].values

        # Invert: transitioning cells (low self_trans) get high confidence
        confidence = 1.0 - self_trans

        # Clip to [0, 1]
        confidence = np.clip(confidence, 0.0, 1.0)
    else:
        # Fallback: uniform confidence
        print("  Warning: No velocity_self_transition found, using uniform confidence")
        confidence = np.ones(adata.n_obs)

    # Add small epsilon to avoid zero confidence
    confidence = confidence + 0.1
    confidence = confidence / confidence.max()

    print(f"  confidence stats: mean={np.mean(confidence):.4f}, std={np.std(confidence):.4f}")

    return confidence.astype(np.float32)


def main():
    # Paths
    input_path = project_root / "experiments/baseline_analysis/data/burn_sham_with_velocity.h5ad"
    output_path = project_root / "experiments/baseline_analysis/data/burn_sham_with_velocity.h5ad"

    print(f"Loading: {input_path}")
    adata = ad.read_h5ad(input_path)

    print(f"\nAnnData shape: {adata.shape}")
    print(f"obs keys: {list(adata.obs.columns)}")
    print(f"obsm keys: {list(adata.obsm.keys())}")
    print(f"layers keys: {list(adata.layers.keys())}")

    # Check for required fields
    needs_velocity_latent = 'velocity_latent' not in adata.obsm
    needs_velocity_confidence = 'velocity_confidence' not in adata.obs
    needs_state_embeddings = 'X_state_2000' not in adata.obsm

    if needs_state_embeddings:
        print("\n⚠️  X_state_2000 not found in obsm!")
        print("   Please run State embedding extraction first:")
        print("   state emb transform --input <file> --output <file> --embed-key X_state_2000")
        print("\n   Alternatively, copy embeddings from burn_sham_baseline_embedded_2000.h5ad")

        # Try to copy from baseline file
        baseline_path = project_root / "experiments/baseline_analysis/data/burn_sham_baseline_embedded_2000.h5ad"
        if baseline_path.exists():
            print(f"\n   Attempting to copy from: {baseline_path}")
            baseline = ad.read_h5ad(baseline_path)

            # Match by barcode if available
            if 'barcode' in adata.obs.columns and 'barcode' in baseline.obs.columns:
                # Create mapping
                baseline_barcodes = set(baseline.obs['barcode'])
                adata_barcodes = adata.obs['barcode']

                # Check overlap
                overlap = sum(1 for b in adata_barcodes if b in baseline_barcodes)
                print(f"   Barcode overlap: {overlap}/{len(adata_barcodes)}")

                if overlap > len(adata_barcodes) * 0.9:
                    # Good overlap, copy embeddings
                    barcode_to_idx = {b: i for i, b in enumerate(baseline.obs['barcode'])}
                    indices = [barcode_to_idx.get(b, -1) for b in adata_barcodes]

                    embeddings = np.zeros((len(adata_barcodes), baseline.obsm['X_state_2000'].shape[1]), dtype=np.float32)
                    for i, idx in enumerate(indices):
                        if idx >= 0:
                            embeddings[i] = baseline.obsm['X_state_2000'][idx]

                    adata.obsm['X_state_2000'] = embeddings
                    print(f"   ✓ Copied X_state_2000 from baseline ({embeddings.shape})")
                    needs_state_embeddings = False

    if needs_state_embeddings:
        print("\n❌ Cannot proceed without State embeddings.")
        return

    # Compute velocity_latent
    if needs_velocity_latent:
        print("\nComputing velocity_latent...")
        velocity_latent = compute_velocity_latent_simple(adata)
        adata.obsm['velocity_latent'] = velocity_latent
        print("✓ Added velocity_latent to obsm")
    else:
        print("\n✓ velocity_latent already present")

    # Compute velocity_confidence
    if needs_velocity_confidence:
        print("\nComputing velocity_confidence...")
        velocity_confidence = compute_velocity_confidence(adata)
        adata.obs['velocity_confidence'] = velocity_confidence
        print("✓ Added velocity_confidence to obs")
    else:
        print("\n✓ velocity_confidence already present")

    # Add additional useful velocity features to obs
    if 'velocity_pseudotime' in adata.obs:
        print("✓ velocity_pseudotime already present")

    if 'latent_time' in adata.obs:
        print("✓ latent_time already present")

    # Save
    print(f"\nSaving to: {output_path}")
    adata.write_h5ad(output_path)

    print("\n" + "="*60)
    print("Velocity data preparation complete!")
    print("="*60)
    print(f"\nNew obsm keys: {list(adata.obsm.keys())}")
    print(f"New obs keys with 'velocity': {[k for k in adata.obs.columns if 'velocity' in k.lower()]}")
    print("\nYou can now run:")
    print("  bash experiments/st_fine_tuning/train_scripts/train_st_lora_mhc_velocity.sh")


if __name__ == "__main__":
    main()
