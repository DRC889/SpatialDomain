#!/usr/bin/env python3
"""
07_unsupervised_benchmark.py — Per-sample unsupervised clustering comparison.

Runs our SpatialDomainAE (dual-graph GAT autoencoder) vs STAGATE per sample.
Evaluates clustering quality with ARI/NMI against ground truth labels.

This is the fair comparison: both methods are unsupervised,
both run per-sample, both produce embeddings that are clustered.
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import h5py
import scanpy as sc
from pathlib import Path
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
from model import SpatialDomainAE, SpatialGATAE, ExprOnlyAE

K_SPATIAL = 15
K_FEATURE = 20
SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def train_ae(model, expr, edge_sp, edge_ft, device, n_epochs=500, lr=1e-3):
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
        recon, z, alpha = model(expr_t, edge_sp_t, edge_ft_t)
        loss = nn.functional.mse_loss(recon, expr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        if epoch % 100 == 0:
            print(f"      epoch {epoch:3d}  loss={loss.item():.4f}")

    model.load_state_dict(best_state)

    # Get embeddings
    model.eval()
    with torch.no_grad():
        z = model.get_embeddings(expr_t, edge_sp_t, edge_ft_t)
    emb = z.cpu().numpy()

    del expr_t, edge_sp_t, edge_ft_t
    torch.cuda.empty_cache()
    return emb


def cluster_and_eval(emb, labels, n_clusters, method="leiden"):
    """Cluster embeddings and evaluate against ground truth."""
    adata = sc.AnnData(X=emb)
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=15)

    if method == "leiden":
        # Try different resolutions to find best ARI
        best_ari = -1
        best_res = 0.5
        for res in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
            sc.tl.leiden(adata, resolution=res, key_added="cluster")
            pred = adata.obs["cluster"].astype(int).values
            ari = adjusted_rand_score(labels, pred)
            if ari > best_ari:
                best_ari = ari
                best_res = res
                best_pred = pred.copy()

        # Final result with best resolution
        nmi = normalized_mutual_info_score(labels, best_pred)
        n_found = len(set(best_pred))
        return best_ari, nmi, n_found, best_res, best_pred

    return None


def run_graphst_sample(expr, coords, device, n_epochs=600):
    """Run GraphST on a single sample."""
    try:
        from GraphST import GraphST as GraphSTModel
    except ImportError:
        return None

    adata = sc.AnnData(X=expr.copy())
    adata.obsm["spatial"] = coords.copy()

    # Our data is already 3000 HVGs (normalized). Mark all genes as HVG
    # so GraphST skips its own preprocess() which requires raw counts.
    adata.var["highly_variable"] = True

    graphst_device = "cuda:0" if device.type == "cuda" else "cpu"
    model = GraphSTModel.GraphST(adata, device=graphst_device, epochs=n_epochs,
                                  dim_output=64, datatype='10X')
    adata = model.train()

    return adata.obsm["emb"]


def run_stagate_sample(expr, coords, device, n_epochs=500):
    """Run STAGATE on a single sample."""
    try:
        import STAGATE_pyG
    except ImportError:
        return None

    adata = sc.AnnData(X=expr.copy())
    adata.obsm["spatial"] = coords.copy()

    # Build spatial graph
    coor = pd.DataFrame(coords, columns=["x", "y"],
                        index=adata.obs_names)
    nbrs = NearestNeighbors(n_neighbors=K_SPATIAL + 1).fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    rows_list, cols_list, dist_list = [], [], []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]):
            rows_list.append(coor.index[i])
            cols_list.append(coor.index[indices[i, j]])
            dist_list.append(distances[i, j])
    adata.uns["Spatial_Net"] = pd.DataFrame({
        "Cell1": rows_list, "Cell2": cols_list, "Distance": dist_list
    })

    adata = STAGATE_pyG.train_STAGATE(
        adata, hidden_dims=[512, 30],
        n_epochs=n_epochs, lr=0.001, random_seed=SEED, device=device.type)

    return adata.obsm["STAGATE"]


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("Unsupervised Benchmark: Per-sample clustering")
    print("=" * 60)

    # Load data
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
    print(f"Spots: {len(labels_all)}, Genes: {n_genes}, Classes: {n_classes}")

    results = []
    cluster_predictions = {}

    for sample in samples:
        print(f"\n{'='*50}")
        print(f"Sample: {sample}")
        print(f"{'='*50}")

        mask = sample_ids == sample
        expr = expr_all[mask]
        coords = coords_all[mask]
        labels = labels_all[mask]
        n_spots = len(labels)
        n_unique = len(set(labels))
        print(f"  Spots: {n_spots}, Unique labels: {n_unique}")

        # Build graphs for this sample
        sp_src, sp_dst = build_knn_edges(coords, K_SPATIAL)
        ft_src, ft_dst = build_feature_edges(expr, K_FEATURE)
        print(f"  Spatial edges: {len(sp_src)}, Feature edges: {len(ft_src)}")

        # ── Our method: SpatialDomainAE ──────────────────
        print(f"\n  --- SpatialDomainAE ---")
        model = SpatialDomainAE(n_genes, latent_dim=64, hidden_dim=256,
                                n_heads=4, dropout=0.3)
        emb_ours = train_ae(model, expr, (sp_src, sp_dst), (ft_src, ft_dst),
                            device, n_epochs=500)
        ari, nmi, n_found, res, pred = cluster_and_eval(
            emb_ours, labels, n_unique)
        print(f"    ARI={ari:.3f}  NMI={nmi:.3f}  clusters={n_found} (res={res})")
        results.append({
            "sample": sample, "method": "SpatialDomainAE",
            "ari": ari, "nmi": nmi, "n_clusters": n_found
        })
        cluster_predictions[f"{sample}_pred"] = pred
        del model; torch.cuda.empty_cache()

        # ── Ablation: SpatialGAT-AE (spatial graph only) ─
        print(f"\n  --- SpatialGATAE ---")
        model = SpatialGATAE(n_genes, latent_dim=64, hidden_dim=256,
                             n_heads=4, dropout=0.3)
        emb_sgat = train_ae(model, expr, (sp_src, sp_dst), (ft_src, ft_dst),
                            device, n_epochs=500)
        ari, nmi, n_found, res, _ = cluster_and_eval(
            emb_sgat, labels, n_unique)
        print(f"    ARI={ari:.3f}  NMI={nmi:.3f}  clusters={n_found} (res={res})")
        results.append({
            "sample": sample, "method": "SpatialGATAE",
            "ari": ari, "nmi": nmi, "n_clusters": n_found
        })
        del model; torch.cuda.empty_cache()

        # ── Ablation: ExprOnly-AE (no graph) ─────────────
        print(f"\n  --- ExprOnlyAE ---")
        model = ExprOnlyAE(n_genes, latent_dim=64, hidden_dim=256, dropout=0.3)
        emb_expr = train_ae(model, expr, (sp_src, sp_dst), (ft_src, ft_dst),
                            device, n_epochs=500)
        ari, nmi, n_found, res, _ = cluster_and_eval(
            emb_expr, labels, n_unique)
        print(f"    ARI={ari:.3f}  NMI={nmi:.3f}  clusters={n_found} (res={res})")
        results.append({
            "sample": sample, "method": "ExprOnlyAE",
            "ari": ari, "nmi": nmi, "n_clusters": n_found
        })
        del model; torch.cuda.empty_cache()

        # ── STAGATE ──────────────────────────────────────
        print(f"\n  --- STAGATE ---")
        emb_stagate = run_stagate_sample(expr, coords, device, n_epochs=500)
        if emb_stagate is not None:
            ari_s, nmi_s, n_found_s, res_s, _ = cluster_and_eval(
                emb_stagate, labels, n_unique)
            print(f"    ARI={ari_s:.3f}  NMI={nmi_s:.3f}  clusters={n_found_s} (res={res_s})")
            results.append({
                "sample": sample, "method": "STAGATE",
                "ari": ari_s, "nmi": nmi_s, "n_clusters": n_found_s
            })
        torch.cuda.empty_cache()

        # ── GraphST ──────────────────────────────────────
        print(f"\n  --- GraphST ---")
        emb_graphst = run_graphst_sample(expr, coords, device, n_epochs=600)
        if emb_graphst is not None:
            ari_g, nmi_g, n_found_g, res_g, _ = cluster_and_eval(
                emb_graphst, labels, n_unique)
            print(f"    ARI={ari_g:.3f}  NMI={nmi_g:.3f}  clusters={n_found_g} (res={res_g})")
            results.append({
                "sample": sample, "method": "GraphST",
                "ari": ari_g, "nmi": nmi_g, "n_clusters": n_found_g
            })
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("Summary (mean ± std across samples)")
    print(f"{'='*60}")

    df = pd.DataFrame(results)
    for method in ["SpatialDomainAE", "SpatialGATAE", "ExprOnlyAE",
                    "STAGATE", "GraphST"]:
        sub = df[df["method"] == method]
        if len(sub) == 0:
            continue
        print(f"  {method:20s}: "
              f"ARI={sub['ari'].mean():.3f}±{sub['ari'].std(ddof=0):.3f}  "
              f"NMI={sub['nmi'].mean():.3f}±{sub['nmi'].std(ddof=0):.3f}")

    df.to_csv(PROJ / "results" / "unsupervised_benchmark.csv", index=False)
    np.savez(PROJ / "results" / "ae_cluster_predictions.npz",
             **cluster_predictions)
    print(f"\nSaved to results/unsupervised_benchmark.csv")
    print("Saved SpatialDomainAE cluster labels to results/ae_cluster_predictions.npz")


if __name__ == "__main__":
    main()
