#!/usr/bin/env python3
"""
12_save_all_predictions.py — Train all 6 methods per sample and save
the Leiden cluster labels (post-Hungarian-matched to ground truth) so
fig1 can show all methods side-by-side.

Methods (in fig1 row order):
  - Ground truth (just labels_all)
  - SpatialDomainAE (Ours)
  - SpatialGATAE
  - ExprOnlyAE
  - STAGATE
  - GraphST
  - SpaGCN
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import h5py
import scanpy as sc
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
from model import SpatialDomainAE, SpatialGATAE, ExprOnlyAE

K_SPATIAL = 15
K_FEATURE = 20
LATENT_DIM = 64
SEED = 42


def build_knn_edges(coords, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)
    src, dst = [], []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]):
            src.append(i)
            dst.append(indices[i, j])
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)


def build_feature_edges(expr, k):
    adj = kneighbors_graph(expr, k, mode="connectivity",
                           metric="correlation", include_self=False)
    rows, cols = adj.nonzero()
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


def train_ae(model_cls, expr, edge_sp, edge_ft, device,
             n_genes, n_epochs=500, lr=1e-3):
    """Train an autoencoder variant and return embeddings."""
    torch.manual_seed(SEED)
    model = model_cls(n_genes, latent_dim=LATENT_DIM, hidden_dim=256,
                      n_heads=4, dropout=0.3) if model_cls is not ExprOnlyAE \
        else ExprOnlyAE(n_genes, latent_dim=LATENT_DIM, hidden_dim=256, dropout=0.3)
    model.to(device)
    expr_t = torch.tensor(expr, dtype=torch.float32).to(device)
    edge_sp_t = torch.tensor(np.stack([edge_sp[0], edge_sp[1]]),
                             dtype=torch.long).to(device)
    edge_ft_t = torch.tensor(np.stack([edge_ft[0], edge_ft[1]]),
                             dtype=torch.long).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_loss = float("inf")
    best_state = None
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(expr_t, edge_sp_t, edge_ft_t)
        recon = out[0] if isinstance(out, tuple) else out
        loss = nn.functional.mse_loss(recon, expr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb = model.get_embeddings(expr_t, edge_sp_t, edge_ft_t)
    emb_np = emb.cpu().numpy()
    del expr_t, edge_sp_t, edge_ft_t, model
    torch.cuda.empty_cache()
    return emb_np


def cluster_with_oracle(emb, labels, sample_name, method_name):
    """Same oracle Leiden grid as benchmark; return best-ARI cluster labels."""
    adata = sc.AnnData(X=emb)
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=15)
    best_ari = -1
    best_pred = None
    best_res = None
    for res in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        sc.tl.leiden(adata, resolution=res, key_added="cluster")
        pred = adata.obs["cluster"].astype(int).values
        ari = adjusted_rand_score(labels, pred)
        if ari > best_ari:
            best_ari = ari
            best_pred = pred.copy()
            best_res = res
    print(f"  {method_name}: ARI={best_ari:.3f} (res={best_res})")
    return best_pred


def hungarian_match(pred, gt, n_gt_classes):
    """
    Map predicted cluster IDs to GT class IDs using Hungarian assignment
    on the contingency table (visualization only — does not affect ARI).
    """
    n_clusters = int(pred.max()) + 1
    cost = np.zeros((n_clusters, n_gt_classes))
    for c in range(n_clusters):
        for g in range(n_gt_classes):
            cost[c, g] = -np.sum((pred == c) & (gt == g))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}
    # Unmapped clusters (more clusters than GT classes): keep as new IDs
    next_id = n_gt_classes
    for c in range(n_clusters):
        if c not in mapping:
            mapping[c] = next_id
            next_id += 1
    mapped = np.array([mapping[int(c)] for c in pred])
    return mapped


def run_external(method, expr, coords, sample, n_pcs=50):
    """Run external baselines (STAGATE, GraphST, SpaGCN) and return embedding."""
    if method == "STAGATE":
        import STAGATE_pyG
        adata = sc.AnnData(X=expr.copy())
        adata.obsm["spatial"] = coords.copy()
        coor = pd.DataFrame(coords, columns=["x", "y"], index=adata.obs_names)
        nbrs = NearestNeighbors(n_neighbors=K_SPATIAL + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        rows_list, cols_list, dist_list = [], [], []
        for i in range(indices.shape[0]):
            for j in range(1, indices.shape[1]):
                rows_list.append(coor.index[i])
                cols_list.append(coor.index[indices[i, j]])
                dist_list.append(distances[i, j])
        adata.uns["Spatial_Net"] = pd.DataFrame({
            "Cell1": rows_list, "Cell2": cols_list, "Distance": dist_list})
        adata = STAGATE_pyG.train_STAGATE(
            adata, hidden_dims=[512, 30],
            n_epochs=500, lr=0.001, random_seed=SEED, device="cuda")
        return adata.obsm["STAGATE"]

    elif method == "GraphST":
        from GraphST import GraphST
        adata = sc.AnnData(X=expr.copy())
        adata.obsm["spatial"] = coords.copy()
        adata.var["highly_variable"] = True
        m = GraphST.GraphST(adata, device="cuda:0", epochs=600,
                             dim_output=64, datatype="10X")
        adata = m.train()
        return adata.obsm["emb"]

    elif method == "SpaGCN":
        import SpaGCN as spg
        from SpaGCN.calculate_adj import calculate_adj_matrix
        adata = sc.AnnData(X=expr.copy())
        adata.obsm["spatial"] = coords.copy()
        adata.obs["x_array"] = coords[:, 0].astype(int)
        adata.obs["y_array"] = coords[:, 1].astype(int)
        adata.obs["x_pixel"] = coords[:, 0].astype(int)
        adata.obs["y_pixel"] = coords[:, 1].astype(int)
        sc.pp.pca(adata, n_comps=n_pcs)
        adj = calculate_adj_matrix(
            x=adata.obs["x_pixel"].tolist(),
            y=adata.obs["y_pixel"].tolist(),
            histology=False)
        l = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01,
                          max_run=100)
        clf = spg.SpaGCN()
        clf.set_l(l)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        clf.train(adata, adj, init_spa=True, init="kmeans", n_clusters=20,
                  tol=5e-3, lr=0.05, max_epochs=200)
        z, q = clf.model.predict(clf.embed, clf.adj_exp)
        return z.detach().cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with h5py.File(PROJ / "data/processed/spatial_dataset.h5", "r") as hf:
        expr_all = hf["expression"][:]
        labels_all = hf["label_indices"][:]
        label_names = [s.decode() for s in hf["label_names"][:]]
    with h5py.File(PROJ / "data/processed/spatial_coords.h5", "r") as hf:
        coords_all = hf["spatial_coords"][:]
        sample_ids = np.array([s.decode() for s in hf["sample_ids"][:]])

    n_genes = expr_all.shape[1]
    n_classes = len(label_names)
    samples = sorted(set(sample_ids))

    methods_internal = {
        "SpatialDomainAE": SpatialDomainAE,
        "SpatialGATAE": SpatialGATAE,
        "ExprOnlyAE": ExprOnlyAE,
    }
    methods_external = ["STAGATE", "GraphST", "SpaGCN"]

    out_data = {}  # {(sample, method): mapped_cluster_labels}

    for sample in samples:
        print(f"\n=== {sample} ===")
        mask = sample_ids == sample
        expr = expr_all[mask]
        coords = coords_all[mask]
        labels = labels_all[mask]

        sp_src, sp_dst = build_knn_edges(coords, K_SPATIAL)
        ft_src, ft_dst = build_feature_edges(expr, K_FEATURE)

        # Internal AE methods
        for name, cls in methods_internal.items():
            print(f"  [training {name}]")
            emb = train_ae(cls, expr, (sp_src, sp_dst), (ft_src, ft_dst),
                            device, n_genes)
            pred = cluster_with_oracle(emb, labels, sample, name)
            mapped = hungarian_match(pred, labels, n_classes)
            out_data[(sample, name)] = mapped

        # External baselines
        for name in methods_external:
            print(f"  [training {name}]")
            try:
                emb = run_external(name, expr, coords, sample)
                pred = cluster_with_oracle(emb, labels, sample, name)
                mapped = hungarian_match(pred, labels, n_classes)
                out_data[(sample, name)] = mapped
            except Exception as e:
                print(f"    FAILED ({type(e).__name__}: {e})")
                out_data[(sample, name)] = None

    # Save
    save_dict = {}
    for (sample, method), arr in out_data.items():
        if arr is not None:
            save_dict[f"{sample}__{method}"] = arr
    np.savez(PROJ / "results/all_method_predictions.npz", **save_dict)
    print(f"\nSaved {len(save_dict)} (sample, method) prediction arrays")


if __name__ == "__main__":
    main()
