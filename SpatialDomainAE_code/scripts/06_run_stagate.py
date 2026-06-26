#!/usr/bin/env python3
"""Run STAGATE as benchmark baseline on LOSO CV."""
import sys
import numpy as np
import pandas as pd
import h5py
import scanpy as sc
import STAGATE_pyG
import torch
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (accuracy_score, f1_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.neural_network import MLPClassifier

PROJ = Path(__file__).resolve().parent.parent
K_SPATIAL = 15


def build_spatial_net(adata, k=15):
    """Build spatial neighbor graph compatible with STAGATE (fixes pandas issue)."""
    coor = adata.obsm["spatial"]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coor)
    distances, indices = nbrs.kneighbors(coor)

    rows, cols, dists = [], [], []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]):  # skip self
            rows.append(adata.obs_names[i])
            cols.append(adata.obs_names[indices[i, j]])
            dists.append(distances[i, j])

    Spatial_Net = pd.DataFrame({
        "Cell1": rows, "Cell2": cols, "Distance": dists
    })
    adata.uns["Spatial_Net"] = Spatial_Net
    print(f"  Graph: {len(Spatial_Net)} edges, "
          f"{len(Spatial_Net)/adata.n_obs:.1f} neighbors/spot")


def _metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "ari": adjusted_rand_score(labels, preds),
        "nmi": normalized_mutual_info_score(labels, preds),
    }


def main():
    print("STAGATE Benchmark (LOSO CV)")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    with h5py.File(PROJ / "data/processed/spatial_dataset.h5", "r") as hf:
        labels = hf["label_indices"][:]
    with h5py.File(PROJ / "data/processed/pca_features.h5", "r") as hf:
        pca = hf["pca"][:]
    with h5py.File(PROJ / "data/processed/spatial_coords.h5", "r") as hf:
        coords = hf["spatial_coords"][:]
        sample_ids = np.array([s.decode() for s in hf["sample_ids"][:]])

    samples = sorted(set(sample_ids))
    print(f"Spots: {len(labels)}, Samples: {samples}")

    results = {}

    for test_sample in samples:
        print(f"\n=== Fold: test={test_sample} ===")

        test_mask = sample_ids == test_sample
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]

        # Run STAGATE on ALL data to get embeddings
        adata = sc.AnnData(X=pca.copy())
        adata.obsm["spatial"] = coords.copy()

        # Build spatial graph (our own function, avoids pandas bug)
        build_spatial_net(adata, k=K_SPATIAL)

        # Train STAGATE
        adata = STAGATE_pyG.train_STAGATE(
            adata, hidden_dims=[512, 30],
            n_epochs=500, lr=0.001, random_seed=42, device=device)

        emb = adata.obsm["STAGATE"]

        # Classify using STAGATE embeddings
        train_emb = emb[train_idx]
        test_emb = emb[test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                            random_state=42, early_stopping=True)
        clf.fit(train_emb, train_labels)
        preds = clf.predict(test_emb)

        r = _metrics(test_labels, preds)
        results[test_sample] = r
        print(f"  Acc={r['accuracy']:.3f}  F1w={r['f1_weighted']:.3f}  "
              f"ARI={r['ari']:.3f}  NMI={r['nmi']:.3f}")

        del adata
        torch.cuda.empty_cache()

    # Summary
    print("\n=== STAGATE Summary ===")
    for metric in ["accuracy", "f1_weighted", "ari", "nmi"]:
        vals = [r[metric] for r in results.values()]
        print(f"  {metric}: {np.mean(vals):.3f}±{np.std(vals):.3f}")


if __name__ == "__main__":
    main()
