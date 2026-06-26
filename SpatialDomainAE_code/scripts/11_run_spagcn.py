#!/usr/bin/env python3
"""
11_run_spagcn.py — Add SpaGCN to the unsupervised benchmark.

SpaGCN (Hu et al. 2021) integrates expression, spatial coordinates, and
optionally histology through graph convolutional networks. We use the
same per-sample protocol as the rest of the benchmark: train on each
sample independently, extract embeddings, cluster with the same Leiden
resolution grid, report best ARI/NMI.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import h5py
import scanpy as sc
import torch
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import SpaGCN as spg
from SpaGCN.calculate_adj import calculate_adj_matrix

PROJ = Path(__file__).resolve().parent.parent
SEED = 42


def cluster_and_score(emb, labels):
    """Apply same oracle Leiden grid as the rest of the benchmark."""
    adata = sc.AnnData(X=emb)
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=15)

    best_ari = -1
    best = None
    for res in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        sc.tl.leiden(adata, resolution=res, key_added="cluster")
        pred = adata.obs["cluster"].astype(int).values
        ari = adjusted_rand_score(labels, pred)
        if ari > best_ari:
            best_ari = ari
            nmi = normalized_mutual_info_score(labels, pred)
            n_found = len(set(pred))
            best = (ari, nmi, n_found, res)
    return best


def run_spagcn_sample(expr, coords, n_pcs=50):
    """Run SpaGCN on a single sample using its standard protocol."""
    adata = sc.AnnData(X=expr.copy())
    adata.obsm["spatial"] = coords.copy()

    # SpaGCN expects integer pixel coordinates as obs columns
    adata.obs["x_array"] = coords[:, 0].astype(int)
    adata.obs["y_array"] = coords[:, 1].astype(int)
    adata.obs["x_pixel"] = coords[:, 0].astype(int)
    adata.obs["y_pixel"] = coords[:, 1].astype(int)

    # PCA for SpaGCN
    sc.pp.pca(adata, n_comps=n_pcs)

    # Adjacency matrix from spatial coordinates (no histology)
    adj = calculate_adj_matrix(
        x=adata.obs["x_pixel"].tolist(),
        y=adata.obs["y_pixel"].tolist(),
        histology=False,
    )

    # SpaGCN main parameters
    p = 0.5
    l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    # SpaGCN's search_res requires the louvain Python package, which does
    # not build on this system. We use kmeans initialization with a fixed
    # resolution (0.7, the SpaGCN default starting point) instead — the
    # downstream Leiden clustering on embeddings still uses the same
    # oracle resolution grid as every other method in the benchmark.
    clf = spg.SpaGCN()
    clf.set_l(l)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    clf.train(adata, adj, init_spa=True, init="kmeans", n_clusters=20,
              tol=5e-3, lr=0.05, max_epochs=200)

    # Extract embedding (z) from the trained SpaGCN model
    z, q = clf.model.predict(clf.embed, clf.adj_exp)
    emb = z.detach().cpu().numpy()
    return emb


def main():
    print("=" * 60)
    print("SpaGCN benchmark on MCAO samples")
    print("=" * 60)

    # Load data
    with h5py.File(PROJ / "data/processed/spatial_dataset.h5", "r") as hf:
        expr_all = hf["expression"][:]
        labels_all = hf["label_indices"][:]
    with h5py.File(PROJ / "data/processed/spatial_coords.h5", "r") as hf:
        coords_all = hf["spatial_coords"][:]
        sample_ids = np.array([s.decode() for s in hf["sample_ids"][:]])

    samples = sorted(set(sample_ids))
    new_rows = []

    for sample in samples:
        print(f"\n--- {sample} ---")
        mask = sample_ids == sample
        expr = expr_all[mask]
        coords = coords_all[mask]
        labels = labels_all[mask]

        emb = run_spagcn_sample(expr, coords)
        if emb is None:
            print("  No embedding returned, skipping")
            continue

        ari, nmi, n_found, res = cluster_and_score(emb, labels)
        print(f"  ARI={ari:.3f}  NMI={nmi:.3f}  clusters={n_found} (res={res})")
        new_rows.append({
            "sample": sample, "method": "SpaGCN",
            "ari": ari, "nmi": nmi, "n_clusters": n_found,
        })

    if not new_rows:
        print("No SpaGCN results — exiting")
        return

    new_df = pd.DataFrame(new_rows)

    csv_path = PROJ / "results/unsupervised_benchmark.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["method"] != "SpaGCN"]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # Summary
    print("\n=== SpaGCN summary ===")
    print(f"  ARI: {new_df['ari'].mean():.3f} +/- {new_df['ari'].std(ddof=0):.3f}")
    print(f"  NMI: {new_df['nmi'].mean():.3f} +/- {new_df['nmi'].std(ddof=0):.3f}")


if __name__ == "__main__":
    main()
