#!/usr/bin/env python3
"""
10_biological_validation.py — DEG and marker gene validation.

For each spatial domain identified by SpatialDomainAE clustering,
compute differentially expressed genes and check against known markers.

This validates that our unsupervised clusters correspond to
biologically meaningful spatial domains in mouse MCAO stroke.

Known markers (from literature):
  - Ischemic core: Hif1a, Bnip3, Ndrg1, Vegfa, Slc2a1 (hypoxia)
  - Penumbra: Hmox1, Ccl2, Tnf, Il1b, Cxcl10 (inflammation)
  - Glial scar: Gfap, Vim, Serpina3n, Lcn2 (reactive astrocytes)
  - Cortex neurons: Slc17a7, Satb2, Tbr1, Camk2a (excitatory)
  - Striatum MSN: Drd1, Drd2, Ppp1r1b, Adora2a (medium spiny)
  - White matter/OL: Mbp, Plp1, Mog, Olig2
  - Microglia: Tmem119, Cx3cr1, P2ry12, Itgam
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
from model import SpatialDomainAE
from figure_style import apply_style, save_figure, SAMPLES_ORDERED, SAMPLE_DISPLAY

apply_style()

FIGDIR = PROJ / "figures"
FIGDIR.mkdir(exist_ok=True)
RESDIR = PROJ / "results"
RESDIR.mkdir(exist_ok=True)

K_SPATIAL = 15
K_FEATURE = 20
LATENT_DIM = 64
SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Known marker genes for stroke-relevant cell types / domains
MARKERS = {
    "Hypoxia/Ischemic core": ["Hif1a", "Bnip3", "Ndrg1", "Vegfa", "Slc2a1",
                               "Ldha", "Pgk1", "Aldoa"],
    "Inflammation/Penumbra": ["Hmox1", "Ccl2", "Tnf", "Il1b", "Cxcl10",
                               "Ccl3", "Ccl4", "Il6"],
    "Reactive astrocytes/Glial scar": ["Gfap", "Vim", "Serpina3n", "Lcn2",
                                        "S100a10", "Aqp4"],
    "Excitatory neurons (Cortex)": ["Slc17a7", "Satb2", "Tbr1", "Camk2a",
                                     "Neurod6", "Nrgn"],
    "Medium spiny neurons (Striatum)": ["Drd1", "Drd2", "Ppp1r1b", "Adora2a",
                                         "Gpr88", "Foxp1"],
    "Oligodendrocytes": ["Mbp", "Plp1", "Mog", "Olig2", "Mag", "Cldn11"],
    "Microglia": ["Tmem119", "Cx3cr1", "P2ry12", "Itgam", "Csf1r", "Hexb"],
}


def _save(fig, name):
    save_figure(fig, name, FIGDIR)
    print(f"  Saved {name}.pdf/.png")


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


def train_ae(expr, edge_sp, edge_ft, device, n_genes, n_epochs=500, lr=1e-3):
    """Train AE and return embeddings."""
    model = SpatialDomainAE(n_genes, latent_dim=LATENT_DIM, hidden_dim=256,
                            n_heads=4, dropout=0.3)
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
    model.eval()
    with torch.no_grad():
        z, alpha = model.encode(expr_t, edge_sp_t, edge_ft_t)
    emb = z.cpu().numpy()

    del expr_t, edge_sp_t, edge_ft_t, model
    torch.cuda.empty_cache()
    return emb


def cluster_embeddings(emb, n_clusters_hint):
    """Cluster embeddings using Leiden, return cluster labels."""
    adata = sc.AnnData(X=emb)
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=15)

    # Find resolution that gives approximately n_clusters_hint clusters
    best_diff = float("inf")
    best_res = 1.0
    best_labels = None
    for res in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        sc.tl.leiden(adata, resolution=res, key_added="cluster")
        n_found = adata.obs["cluster"].nunique()
        diff = abs(n_found - n_clusters_hint)
        if diff < best_diff:
            best_diff = diff
            best_res = res
            best_labels = adata.obs["cluster"].astype(int).values.copy()

    return best_labels, best_res


def compute_degs(adata, groupby="cluster", n_genes=50):
    """Compute DEGs per cluster using Wilcoxon test."""
    # Filter out groups with fewer than 2 spots
    counts = adata.obs[groupby].value_counts()
    keep = counts[counts >= 2].index
    adata_filt = adata[adata.obs[groupby].isin(keep)].copy()
    adata_filt.obs[groupby] = pd.Categorical(adata_filt.obs[groupby])
    sc.tl.rank_genes_groups(adata_filt, groupby=groupby, method="wilcoxon",
                            pts=True, n_genes=n_genes)
    return adata_filt


def marker_enrichment_score(deg_results, gene_names, markers_dict):
    """
    For each cluster, compute enrichment of known marker gene sets.
    Returns a DataFrame: clusters × marker categories, values = fraction of
    markers in top DEGs.
    """
    clusters = list(deg_results.uns["rank_genes_groups"]["names"].dtype.names)
    scores = {}

    for cluster in clusters:
        top_genes = [deg_results.uns["rank_genes_groups"]["names"][cluster][i]
                     for i in range(50)]
        scores[cluster] = {}
        for cat, markers in markers_dict.items():
            # Check how many markers appear in top 50 DEGs
            found = [g for g in markers if g in top_genes]
            scores[cluster][cat] = len(found) / len(markers)

    return pd.DataFrame(scores).T  # index=clusters, columns=categories




def fig_marker_spatial(adata_full):
    """
    Spatial expression of representative domain markers across all four
    samples. Each row is a marker gene; columns are time points.
    """
    # Make a working copy with raw counts → log-normalized
    adata = adata_full.copy()
    if "raw_counts" in adata.layers:
        adata.X = adata.layers["raw_counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    panels = [
        ("Spp1",      "Inflammation/core"),
        ("C1qa",      "Complement/microglia"),
        ("Serpina3n", "Reactive astrocytes"),
        ("Ppp1r1b",   "Striatal MSN"),
        ("Plp1",      "Oligodendrocytes"),
        ("Slc17a7",   "Cortical excitatory"),
    ]
    panels = [(g, d) for g, d in panels if g in adata.var_names]

    fig, axes = plt.subplots(len(panels), len(SAMPLES_ORDERED),
                              figsize=(7.5, 1.4 * len(panels)))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    last_sc = None

    for r, (gene, desc) in enumerate(panels):
        col = adata[:, gene].X
        expr = (col.toarray().flatten() if hasattr(col, "toarray")
                else np.asarray(col).flatten())
        vmax = np.percentile(expr, 99)
        for c, sample in enumerate(SAMPLES_ORDERED):
            ax = axes[r, c]
            mask = (adata.obs["sample"] == sample).values
            coords = adata.obsm["spatial"][mask]
            e = expr[mask]
            last_sc = ax.scatter(coords[:, 0], coords[:, 1], c=e,
                                  cmap="Reds", s=1.5, alpha=0.95,
                                  vmin=0, vmax=vmax,
                                  edgecolors="none", rasterized=True)
            ax.set_aspect("equal"); ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(SAMPLE_DISPLAY[sample], fontsize=10,
                              fontweight="bold", pad=4)
            if c == 0:
                ax.text(-0.15, 0.5, gene, transform=ax.transAxes,
                         fontsize=10, fontweight="bold", fontstyle="italic",
                         ha="right", va="center")
                ax.text(-0.15, 0.30, desc, transform=ax.transAxes,
                         fontsize=7, color="#555555",
                         ha="right", va="center")

    fig.subplots_adjust(right=0.93, left=0.10)
    cbar_ax = fig.add_axes([0.95, 0.30, 0.012, 0.4])
    cbar = fig.colorbar(last_sc, cax=cbar_ax)
    cbar.set_label("log-normalized\nexpression", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    _save(fig, "fig5_marker_spatial")


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("Biological Validation: DEG & Marker Gene Analysis")
    print("=" * 60)

    # Load full adata (has raw counts and all genes)
    adata_full = sc.read_h5ad(PROJ / "data/processed/adata_combined.h5ad")
    print(f"Full adata: {adata_full.shape}")

    # Load processed data (3000 HVGs, used for model training)
    with h5py.File(PROJ / "data/processed/spatial_dataset.h5", "r") as hf:
        expr_all = hf["expression"][:]
        labels_all = hf["label_indices"][:]
        label_names = [s.decode() for s in hf["label_names"][:]]
        gene_names_hvg = [s.decode() for s in hf["gene_names"][:]]
    with h5py.File(PROJ / "data/processed/spatial_coords.h5", "r") as hf:
        coords_all = hf["spatial_coords"][:]
        sample_ids = np.array([s.decode() for s in hf["sample_ids"][:]])

    n_genes = expr_all.shape[1]

    # Check which marker genes are in our HVG set
    print("\nMarker gene availability in HVG set:")
    available_markers = {}
    for cat, markers in MARKERS.items():
        found = [g for g in markers if g in gene_names_hvg]
        available_markers[cat] = found
        print(f"  {cat}: {len(found)}/{len(markers)} "
              f"({', '.join(found[:5])}{'...' if len(found) > 5 else ''})")

    # Focus on 3DPI (most complex pathology, best SpatialDomainAE performance)
    target_sample = "3DPI"
    print(f"\n{'='*50}")
    print(f"Primary analysis: {target_sample}")
    print(f"{'='*50}")

    mask = sample_ids == target_sample
    expr = expr_all[mask]
    coords = coords_all[mask]
    labels = labels_all[mask]
    n_spots = len(labels)
    n_unique = len(set(labels))
    print(f"  Spots: {n_spots}, Ground truth domains: {n_unique}")

    # Build graphs and get embeddings
    sp_src, sp_dst = build_knn_edges(coords, K_SPATIAL)
    ft_src, ft_dst = build_feature_edges(expr, K_FEATURE)
    emb = train_ae(expr, (sp_src, sp_dst), (ft_src, ft_dst), device, n_genes)

    # Cluster embeddings
    cluster_labels, best_res = cluster_embeddings(emb, n_unique)
    print(f"  Leiden clusters: {len(set(cluster_labels))} (res={best_res})")

    # Build adata for DEG analysis (use full gene set from adata_full)
    mask_full = adata_full.obs["sample"] == target_sample
    adata_sample = adata_full[mask_full].copy()
    adata_sample.obs["cluster"] = pd.Categorical(
        [str(c) for c in cluster_labels])
    adata_sample.obs["ground_truth"] = pd.Categorical(
        [label_names[l] for l in labels])

    # Use raw counts for DEG, then log-normalize
    adata_sample.X = adata_sample.layers["raw_counts"].copy()
    sc.pp.normalize_total(adata_sample, target_sum=1e4)
    sc.pp.log1p(adata_sample)

    # --- DEG analysis on predicted clusters ---
    print("\n  Computing DEGs per predicted cluster...")
    adata_deg = compute_degs(adata_sample, groupby="cluster", n_genes=50)

    # Save top DEGs
    deg_results = []
    clusters_found = list(
        adata_deg.uns["rank_genes_groups"]["names"].dtype.names)
    for cl in clusters_found:
        for i in range(20):
            gene = adata_deg.uns["rank_genes_groups"]["names"][cl][i]
            score = adata_deg.uns["rank_genes_groups"]["scores"][cl][i]
            pval = adata_deg.uns["rank_genes_groups"]["pvals_adj"][cl][i]
            logfc = adata_deg.uns["rank_genes_groups"]["logfoldchanges"][cl][i]
            deg_results.append({
                "cluster": cl, "rank": i + 1, "gene": gene,
                "score": score, "pval_adj": pval, "log2fc": logfc,
            })

    df_deg = pd.DataFrame(deg_results)
    df_deg.to_csv(RESDIR / f"deg_{target_sample}_clusters.csv", index=False)
    print(f"  Saved DEGs: {len(df_deg)} entries")

    # --- Marker enrichment for predicted clusters ---
    print("\n  Computing marker enrichment...")
    # Use full gene names for marker lookup
    enrichment = marker_enrichment_score(adata_deg, adata_sample.var_names,
                                         MARKERS)
    enrichment.to_csv(RESDIR / f"marker_enrichment_{target_sample}.csv")
    print(f"  Enrichment matrix: {enrichment.shape}")

    # --- Figure: spatial expression of representative markers across all samples ---
    print("\n  Generating marker spatial feature plot (fig5_marker_spatial)...")
    fig_marker_spatial(adata_full)

    # --- Cluster-to-domain mapping ---
    print("\n  Cluster-to-domain correspondence:")
    contingency = pd.crosstab(
        pd.Series(cluster_labels, name="cluster"),
        pd.Series([label_names[l] for l in labels], name="domain"),
    )
    # For each cluster, find dominant domain
    mapping = []
    for cl in contingency.index:
        dominant = contingency.loc[cl].idxmax()
        purity = contingency.loc[cl].max() / contingency.loc[cl].sum()
        mapping.append({
            "cluster": cl, "dominant_domain": dominant,
            "purity": purity, "n_spots": contingency.loc[cl].sum(),
        })
    df_map = pd.DataFrame(mapping)
    df_map.to_csv(RESDIR / f"cluster_domain_mapping_{target_sample}.csv",
                  index=False)
    print(df_map.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
