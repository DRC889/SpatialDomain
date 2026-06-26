#!/usr/bin/env python3
"""
09_attention_analysis.py — Extract and visualize attention fusion weights.

Core insight: The attention fusion module assigns per-spot weights
α_spatial and α_feature. These weights reveal which spots rely on
spatial context (tissue architecture) vs transcriptomic similarity
(cell-type identity).

In ischemic stroke, spatial organization is disrupted in the lesion:
- Control brain: spatial context dominates (well-organized tissue)
- Ischemic core: feature context increases (spatial structure disrupted)
- Penumbra: intermediate (partial disruption)

This "spatial disorganization" signal is a biological discovery unique
to the dual-graph attention architecture.

Generates:
  Fig_attention_maps: Spatial maps colored by α_feature (4 samples)
  Fig_attention_boxplot: α_feature distribution by domain category
  Fig_feature_graph_distance: Physical distance of feature graph edges
  Fig_embedding_umap: UMAP of embeddings colored by time point + domain
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
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
from model import SpatialDomainAE
from figure_style import (apply_style, save_figure,
                            SAMPLES_ORDERED, SAMPLE_DISPLAY)

apply_style()
FIGDIR = PROJ / "figures"
FIGDIR.mkdir(exist_ok=True)

K_SPATIAL = 15
K_FEATURE = 20
LATENT_DIM = 64
SEED = 42

# Backwards-compat alias used throughout this script
SAMPLES = SAMPLES_ORDERED


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save(fig, name):
    save_figure(fig, name, FIGDIR)


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


def train_ae_and_get_alpha(expr, edge_sp, edge_ft, device,
                           n_genes, n_epochs=500, lr=1e-3):
    """Train AE and return both embeddings and attention weights."""
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
    alpha_np = alpha.cpu().numpy()  # (N, 2): [α_spatial, α_feature]

    del expr_t, edge_sp_t, edge_ft_t
    torch.cuda.empty_cache()
    return emb, alpha_np, model


def fig_attention_maps(all_data):
    """
    Spatial maps colored by α_feature for each sample.
    High α_feature = spot relies more on transcriptomic identity
    (spatial organization disrupted).
    """
    # Colormap: blue (spatial-dominant) → white → red (feature-dominant)
    cmap = LinearSegmentedColormap.from_list(
        "spatial_feature",
        ["#1565C0", "#BBDEFB", "#FFFFFF", "#FFCDD2", "#C62828"],
        N=256,
    )

    fig, axes = plt.subplots(1, 4, figsize=(8, 2.5))
    fig.subplots_adjust(wspace=0.05, right=0.88)

    for j, sample in enumerate(SAMPLES):
        ax = axes[j]
        d = all_data[sample]
        coords = d["coords"]
        alpha_feat = d["alpha"][:, 1]  # α_feature

        sc_plot = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=alpha_feat, cmap=cmap, s=1.5, alpha=0.9,
            edgecolors="none", rasterized=True,
            vmin=0.3, vmax=0.7,
        )

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(SAMPLE_DISPLAY[sample], fontsize=10, fontweight="bold")

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc_plot, cax=cbar_ax)
    cbar.set_label(r"$\alpha_{\mathrm{feature}}$", fontsize=10)
    cbar.ax.tick_params(labelsize=7)

    _save(fig, "archive/fig_attention_maps_standalone")


def fig_attention_by_domain(all_data, label_names):
    """
    Box/violin plot: α_feature grouped by domain category.
    Categories: Anatomical, Lesioned cortex, ISD core, ISD penumbra, Glial scar.
    """
    # Classify domains into categories
    domain_cats = {}
    for name in label_names:
        if name.startswith("ISD") and name.endswith("c"):
            domain_cats[name] = "ISD core"
        elif name.startswith("ISD") and name.endswith("p"):
            domain_cats[name] = "ISD penumbra"
        elif name.startswith("lCTX"):
            domain_cats[name] = "Lesioned cortex"
        elif name == "GLS":
            domain_cats[name] = "Glial scar"
        else:
            domain_cats[name] = "Anatomical"

    rows = []
    for sample in SAMPLES:
        d = all_data[sample]
        alpha_feat = d["alpha"][:, 1]
        labels = d["labels"]
        for i in range(len(labels)):
            dname = label_names[labels[i]]
            cat = domain_cats[dname]
            rows.append({
                "sample": sample,
                "domain": dname,
                "category": cat,
                "alpha_feature": alpha_feat[i],
            })

    df = pd.DataFrame(rows)

    # Order categories biologically
    cat_order = ["Anatomical", "Lesioned cortex", "Glial scar",
                 "ISD penumbra", "ISD core"]
    cat_colors = {
        "Anatomical": "#4A90D9",
        "Lesioned cortex": "#7BB3E0",
        "Glial scar": "#8D6E63",
        "ISD penumbra": "#EF9A9A",
        "ISD core": "#D32F2F",
    }

    fig, ax = plt.subplots(figsize=(5, 3.5))

    positions = []
    labels_list = []
    bp_data = []
    colors = []
    for i, cat in enumerate(cat_order):
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        bp_data.append(sub["alpha_feature"].values)
        positions.append(i)
        labels_list.append(cat)
        colors.append(cat_colors[cat])

    bp = ax.boxplot(bp_data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.2))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("gray")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_list, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(r"$\alpha_{\mathrm{feature}}$", fontsize=10)
    ax.set_title("Feature graph reliance by domain category", fontsize=10,
                 fontweight="bold")

    # Add horizontal line at 0.5 (equal weighting)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(len(positions) - 0.5, 0.505, "equal", fontsize=7, color="gray",
            va="bottom", ha="right")

    fig.tight_layout()
    _save(fig, "archive/fig4_attention_by_domain")


def fig_feature_graph_distance(all_data):
    """
    KDE of physical distances for spatial vs feature graph edges.
    Curves are normalized to the same maximum so both distributions are
    visible despite the spatial graph's narrow concentration near the
    Visium spot pitch.
    """
    from scipy.stats import gaussian_kde
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.5), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for j, sample in enumerate(SAMPLES):
        ax = axes[j]
        d = all_data[sample]
        coords = d["coords"]
        sp_src, sp_dst = d["edge_sp"]
        ft_src, ft_dst = d["edge_ft"]

        sp_dist = np.sqrt(np.sum((coords[sp_src] - coords[sp_dst])**2,
                                   axis=1))
        ft_dist = np.sqrt(np.sum((coords[ft_src] - coords[ft_dst])**2,
                                   axis=1))

        # Subsample for KDE speed
        rng = np.random.default_rng(42)
        sp_sub = rng.choice(sp_dist, min(5000, len(sp_dist)), replace=False)
        ft_sub = rng.choice(ft_dist, min(5000, len(ft_dist)), replace=False)

        x_max = max(ft_dist.max(), sp_dist.max())
        x_range = np.linspace(0, x_max * 1.02, 500)
        kde_sp = gaussian_kde(sp_sub, bw_method=0.15)
        kde_ft = gaussian_kde(ft_sub, bw_method=0.15)
        sp_y = kde_sp(x_range); sp_y = sp_y / sp_y.max()
        ft_y = kde_ft(x_range); ft_y = ft_y / ft_y.max()

        ax.fill_between(x_range, sp_y, alpha=0.4, color="#1565C0",
                         label="Spatial graph", linewidth=0)
        ax.fill_between(x_range, ft_y, alpha=0.4, color="#D32F2F",
                         label="Feature graph", linewidth=0)
        ax.plot(x_range, sp_y, color="#1565C0", linewidth=1.0)
        ax.plot(x_range, ft_y, color="#D32F2F", linewidth=1.0)
        ax.axvline(np.median(sp_dist), color="#0D47A1",
                    linestyle="--", linewidth=0.8, alpha=0.8)
        ax.axvline(np.median(ft_dist), color="#B71C1C",
                    linestyle="--", linewidth=0.8, alpha=0.8)
        ax.text(np.median(sp_dist), 0.85, f"{np.median(sp_dist):.0f}",
                 fontsize=5.5, color="#0D47A1", ha="right",
                 va="bottom", rotation=90)
        ax.text(np.median(ft_dist), 0.85, f"{np.median(ft_dist):.0f}",
                 fontsize=5.5, color="#B71C1C", ha="left",
                 va="bottom", rotation=90)

        ax.set_title(SAMPLE_DISPLAY[sample], fontsize=10, fontweight="bold")
        ax.set_xlabel("Physical distance (px)", fontsize=8)
        if j == 0:
            ax.set_ylabel("Normalized density", fontsize=9)
            ax.legend(fontsize=6, frameon=False, loc="upper right")
        ax.set_xlim(0, x_range[-1] * 0.85)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])

        print(f"  {sample}: spatial median={np.median(sp_dist):.1f}, "
              f"feature median={np.median(ft_dist):.1f}")

    fig.tight_layout()
    _save(fig, "fig4_graph_distance")


def fig_alpha_timecourse(all_data, label_names):
    """
    Per-spot alpha_feature trajectory across time points,
    grouped by lesion-related domain category.
    """
    import pandas as pd
    from scipy import stats

    cat_colors = {
        "Ischemic core":     "#D32F2F",
        "Ischemic penumbra": "#EF9A9A",
        "Glial scar":        "#8D6E63",
        "Lesioned cortex":   "#7BB3E0",
        "Intact anatomy":    "#4A90D9",
    }

    rows = []
    for sample in SAMPLES:
        d = all_data[sample]
        alpha_feat = d["alpha"][:, 1]
        labels = d["labels"]
        for i in range(len(labels)):
            dname = label_names[labels[i]]
            if dname.startswith("ISD") and dname.endswith("c"):
                cat = "Ischemic core"
            elif dname.startswith("ISD") and dname.endswith("p"):
                cat = "Ischemic penumbra"
            elif dname.startswith("lCTX"):
                cat = "Lesioned cortex"
            elif dname == "GLS":
                cat = "Glial scar"
            else:
                cat = "Intact anatomy"
            rows.append({"sample": sample, "category": cat,
                          "alpha_feature": alpha_feat[i]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    time_points = [1, 3, 7]
    sample_labels = ["1DPI", "3DPI", "7DPI"]

    for cat in ["Ischemic core", "Ischemic penumbra", "Glial scar",
                 "Lesioned cortex", "Intact anatomy"]:
        means, sems, xs = [], [], []
        for tp, sample in zip(time_points, sample_labels):
            sub = df[(df["sample"] == sample) &
                      (df["category"] == cat)]["alpha_feature"]
            if len(sub) >= 5:
                means.append(sub.median())
                sems.append((sub.quantile(0.75) - sub.quantile(0.25)) / 2)
                xs.append(tp)
        if xs:
            ax.errorbar(xs, means, yerr=sems, marker="o", label=cat,
                         color=cat_colors[cat], linewidth=1.5,
                         markersize=5, capsize=3, elinewidth=0.8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xticks([1, 3, 7])
    ax.set_xticklabels(["1 DPI", "3 DPI", "7 DPI"])
    ax.set_xlabel("Days post-injury", fontsize=10)
    ax.set_ylabel(r"$\alpha_{\mathrm{feature}}$ (median $\pm$ IQR/2)",
                   fontsize=10)
    ax.legend(fontsize=6, frameon=False, loc="center left",
                bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()

    # Save summary CSV alongside figure
    summary = df.groupby(["sample", "category"])["alpha_feature"].agg(
        ["mean", "median", "std", "count"]).round(3)
    summary.to_csv(PROJ / "results/alpha_feature_timecourse.csv")

    # Print stats for manuscript
    print()
    print("  Mann-Whitney (one-sided): ischemic vs intact, per sample")
    for sample in ["1DPI", "3DPI", "7DPI"]:
        intact = df[(df["sample"] == sample) &
                     (df["category"] == "Intact anatomy")]["alpha_feature"]
        for cat in ["Ischemic core", "Ischemic penumbra"]:
            ischemic = df[(df["sample"] == sample) &
                           (df["category"] == cat)]["alpha_feature"]
            if len(ischemic) > 0:
                stat, pval = stats.mannwhitneyu(ischemic, intact,
                                                 alternative="greater")
                delta = sum(np.sign(x - y) for x in ischemic
                             for y in intact) / (len(ischemic) * len(intact))
                print(f"    {sample} {cat:18s}: n={len(ischemic):4d} "
                       f"median={ischemic.median():.3f} "
                       f"(intact={intact.median():.3f}) "
                       f"p={pval:.2e} delta={delta:.3f}")

    _save(fig, "archive/fig5_alpha_timecourse")


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("Attention Analysis & Biological Insights")
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

    all_data = {}

    for sample in SAMPLES:
        print(f"\n{'='*50}")
        print(f"Sample: {sample}")
        print(f"{'='*50}")

        mask = sample_ids == sample
        expr = expr_all[mask]
        coords = coords_all[mask]
        labels = labels_all[mask]

        # Build graphs
        sp_src, sp_dst = build_knn_edges(coords, K_SPATIAL)
        ft_src, ft_dst = build_feature_edges(expr, K_FEATURE)
        print(f"  Spots: {len(labels)}, Spatial edges: {len(sp_src)}, "
              f"Feature edges: {len(ft_src)}")

        # Train and extract attention
        emb, alpha, model = train_ae_and_get_alpha(
            expr, (sp_src, sp_dst), (ft_src, ft_dst), device, n_genes)

        print(f"  α_spatial: mean={alpha[:, 0].mean():.3f}±{alpha[:, 0].std():.3f}")
        print(f"  α_feature: mean={alpha[:, 1].mean():.3f}±{alpha[:, 1].std():.3f}")

        all_data[sample] = {
            "coords": coords,
            "labels": labels,
            "emb": emb,
            "alpha": alpha,
            "edge_sp": (sp_src, sp_dst),
            "edge_ft": (ft_src, ft_dst),
        }
        del model
        torch.cuda.empty_cache()

    # Save attention weights
    np.savez(PROJ / "results/attention_weights.npz",
             **{f"{s}_alpha": all_data[s]["alpha"] for s in SAMPLES},
             **{f"{s}_emb": all_data[s]["emb"] for s in SAMPLES})
    print("\nSaved attention weights to results/attention_weights.npz")

    # Generate figures
    print("\n--- Generating figures ---")

    print("\nFig 10: Attention fusion maps")
    fig_attention_maps(all_data)

    print("\nFig 11: Attention by domain category")
    fig_attention_by_domain(all_data, label_names)

    print("\nFig 12: Feature graph distance analysis")
    fig_feature_graph_distance(all_data)

    print("\nFig 13: Alpha_feature time-course")
    fig_alpha_timecourse(all_data, label_names)

    print("\nDone!")


if __name__ == "__main__":
    main()
