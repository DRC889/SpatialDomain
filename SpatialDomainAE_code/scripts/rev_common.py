"""rev_common.py — shared utilities for revision experiments.

Reuses the exact pipeline from 07_unsupervised_benchmark.py:
  - per-sample KNN spatial graph (Euclidean, k_s)
  - per-sample feature graph (correlation KNN, k_f)
  - dual-graph GAT autoencoder, MSE recon, AdamW + cosine, 500 epochs
  - Leiden over fixed oracle resolution grid, report best per-sample ARI/NMI
All model variants are pure-PyTorch (no torch_geometric dependency).
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import h5py
from pathlib import Path
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Clustering backend: prefer scanpy (matches the submitted paper's protocol).
try:
    import scanpy as sc
    _HAVE_SCANPY = True
except Exception:
    import igraph as ig
    import leidenalg as la
    _HAVE_SCANPY = False

WORK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK / "src"))
from model import SpatialDomainAE, SpatialGATAE, ExprOnlyAE  # noqa: E402

DATA = WORK / "data" / "processed"
K_SPATIAL = 15
K_FEATURE = 20
RES_GRID = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    with h5py.File(DATA / "spatial_dataset.h5", "r") as hf:
        expr_all = hf["expression"][:]
        labels_all = hf["label_indices"][:]
        gene_names = [s.decode() for s in hf["gene_names"][:]]
    with h5py.File(DATA / "spatial_coords.h5", "r") as hf:
        coords_all = hf["spatial_coords"][:]
        sample_ids = np.array([s.decode() for s in hf["sample_ids"][:]])
    return expr_all, labels_all, coords_all, sample_ids, gene_names


def build_knn_edges(coords, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(coords)
    _, idx = nbrs.kneighbors(coords)
    src, dst = [], []
    for i in range(idx.shape[0]):
        for j in range(1, idx.shape[1]):
            src.append(i); dst.append(idx[i, j])
    return np.array(src, np.int64), np.array(dst, np.int64)


def build_feature_edges(expr, k, metric="correlation"):
    adj = kneighbors_graph(expr, k, mode="connectivity",
                           metric=metric, include_self=False)
    rows, cols = adj.nonzero()
    return np.array(rows, np.int64), np.array(cols, np.int64)


def train_ae(model, expr, edge_sp, edge_ft, device, n_epochs=500, lr=1e-3):
    model.to(device)
    expr_t = torch.tensor(expr, dtype=torch.float32, device=device)
    edge_sp_t = torch.tensor(np.stack(edge_sp), dtype=torch.long, device=device)
    edge_ft_t = torch.tensor(np.stack(edge_ft), dtype=torch.long, device=device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=n_epochs)
    best_loss, best_state = float("inf"), None
    for ep in range(n_epochs):
        model.train(); opt.zero_grad()
        recon, z, alpha = model(expr_t, edge_sp_t, edge_ft_t)
        loss = nn.functional.mse_loss(recon, expr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model.get_embeddings(expr_t, edge_sp_t, edge_ft_t)
    emb = z.cpu().numpy()
    del expr_t, edge_sp_t, edge_ft_t
    torch.cuda.empty_cache()
    return emb


def _build_igraph(emb, n_neighbors=15):
    """KNN graph on embeddings -> undirected igraph (scanpy-free Leiden input)."""
    A = kneighbors_graph(emb.astype(np.float32), n_neighbors,
                         mode="connectivity", include_self=False)
    src, dst = A.nonzero()
    g = ig.Graph(n=emb.shape[0], edges=list(zip(src.tolist(), dst.tolist())),
                 directed=False)
    g.simplify(multiple=True, loops=True)
    return g


def cluster_and_eval(emb, labels, seed=0, res_grid=None):
    """Leiden over the oracle resolution grid; report best per-sample ARI.
    Uses scanpy (neighbors + leiden) when available to match the submitted
    paper's protocol; otherwise falls back to leidenalg directly.
    """
    grid = res_grid if res_grid is not None else RES_GRID
    best = (-1, -1, 0, None, None)
    if _HAVE_SCANPY:
        adata = sc.AnnData(X=emb.astype(np.float32))
        sc.pp.neighbors(adata, use_rep="X", n_neighbors=15)
        for res in grid:
            sc.tl.leiden(adata, resolution=res, key_added="cl")
            pred = adata.obs["cl"].astype(int).values
            ari = adjusted_rand_score(labels, pred)
            if ari > best[0]:
                nmi = normalized_mutual_info_score(labels, pred)
                best = (ari, nmi, len(set(pred)), res, pred.copy())
        return best
    # scanpy-free fallback
    g = _build_igraph(emb, n_neighbors=15)
    for res in grid:
        part = la.find_partition(g, la.RBConfigurationVertexPartition,
                                 resolution_parameter=res, seed=seed)
        pred = np.array(part.membership)
        ari = adjusted_rand_score(labels, pred)
        if ari > best[0]:
            nmi = normalized_mutual_info_score(labels, pred)
            best = (ari, nmi, len(set(pred.tolist())), res, pred.copy())
    return best


MODELS = {
    "SpatialDomainAE": lambda g: SpatialDomainAE(g, latent_dim=64, hidden_dim=256, n_heads=4, dropout=0.3),
    "SpatialGATAE":    lambda g: SpatialGATAE(g, latent_dim=64, hidden_dim=256, n_heads=4, dropout=0.3),
    "ExprOnlyAE":      lambda g: ExprOnlyAE(g, latent_dim=64, hidden_dim=256, dropout=0.3),
}
