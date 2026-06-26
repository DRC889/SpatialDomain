#!/usr/bin/env python3
"""
09_biological_analysis.py — Downstream biological analyses for SpatialStroke.

Generates:
  Fig5: Per-fold per-domain accuracy heatmap (folds x domains)
  Fig6: Prediction spatial maps per sample showing correct/incorrect spots
  Fig7: Domain size vs accuracy scatter (shows which domains are hard)

All outputs in both PDF and PNG (300 dpi).
"""
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
FIGDIR = PROJ / "figures"
FIGDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style (match 08_figures.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DOMAIN_COLORS = {
    "CTX1-4": "#4A90D9", "CTX5": "#2C5FA1", "CTX6": "#1B3A6B",
    "lCTX4-5": "#7BB3E0", "lCTX6": "#A8CCE8",
    "ISD1c": "#D32F2F", "ISD1p": "#EF9A9A",
    "ISD3c": "#E65100", "ISD3p": "#FFB74D",
    "ISD7c": "#C62828", "ISD7p": "#FFCDD2",
    "HIP": "#7B1FA2", "AMY": "#CE93D8",
    "CP": "#2E7D32", "PAL": "#66BB6A",
    "TH": "#A5D6A7", "HY": "#00897B",
    "PIR": "#F9A825", "GLS": "#8D6E63",
    "CS": "#78909C", "FT": "#B0BEC5", "LV": "#ECEFF1",
}

SAMPLES = ["Ctrl", "1DPI", "3DPI", "7DPI"]
SAMPLE_DISPLAY = {"Ctrl": "Control", "1DPI": "1 DPI", "3DPI": "3 DPI", "7DPI": "7 DPI"}


def _save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  Saved {name}.pdf/.png")


def load_data():
    with h5py.File(PROJ / "data/processed/spatial_dataset.h5", "r") as f:
        labels_all = f["label_indices"][:]
        label_names = [s.decode() for s in f["label_names"][:]]
    with h5py.File(PROJ / "data/processed/spatial_coords.h5", "r") as f:
        coords_all = f["spatial_coords"][:]
        sample_ids = np.array([s.decode() for s in f["sample_ids"][:]])
    preds = np.load(PROJ / "results/predictions.npz")
    return labels_all, label_names, coords_all, sample_ids, preds


def fig5_perfold_heatmap(labels_all, label_names, sample_ids, preds):
    """
    Fig5: Per-fold per-domain accuracy heatmap.
    Rows = LOSO folds (test sample), Columns = 22 domains.
    Shows which domains are easy/hard in which fold.
    """
    n_domains = len(label_names)
    acc_matrix = np.full((4, n_domains), np.nan)
    fold_labels = []

    for i, sample in enumerate(SAMPLES):
        mask = sample_ids == sample
        gt = labels_all[mask]
        pred = preds[f"SpatialDomainNet_{sample}"]
        fold_labels.append(f"Test: {SAMPLE_DISPLAY[sample]}")

        for d in range(n_domains):
            d_mask = gt == d
            if d_mask.sum() > 0:
                acc_matrix[i, d] = (pred[d_mask] == d).mean()

    # Custom white-to-navy colormap
    cmap = LinearSegmentedColormap.from_list(
        "white_navy", ["#FFFFFF", "#E3F2FD", "#90CAF9", "#1565C0", "#0D2B5E"])
    cmap.set_bad(color="#F5F5F5")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.subplots_adjust(bottom=0.22, right=0.92)
    im = ax.imshow(acc_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_domains))
    ax.set_xticklabels(label_names, rotation=50, ha="right", fontsize=6.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels(fold_labels, fontsize=9)

    # Annotate cells
    for i in range(4):
        for j in range(n_domains):
            val = acc_matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=6, color="#BDBDBD")
            else:
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    # Grid lines
    for edge in range(n_domains + 1):
        ax.axvline(edge - 0.5, color="white", linewidth=0.5)
    for edge in range(5):
        ax.axhline(edge - 0.5, color="white", linewidth=0.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Per-class accuracy", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Per-fold Per-domain Classification Accuracy (SpatialDomainNet)",
                 fontsize=11, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    _save(fig, "fig5_perfold_heatmap")
    return acc_matrix


def fig6_error_maps(labels_all, label_names, coords_all, sample_ids, preds):
    """
    Fig6: Spatial error maps showing correct (gray) vs incorrect (red) predictions.
    Highlights where the model makes mistakes — typically at domain boundaries.
    """
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.8))
    fig.subplots_adjust(wspace=0.05, top=0.80, bottom=0.08)

    for j, sample in enumerate(SAMPLES):
        ax = axes[j]
        mask = sample_ids == sample
        coords = coords_all[mask]
        gt = labels_all[mask]
        pred = preds[f"SpatialDomainNet_{sample}"]

        correct = pred == gt
        acc = correct.mean()

        # Plot correct spots first (background), then errors on top
        ax.scatter(coords[correct, 0], coords[correct, 1],
                   c="#D0D0D0", s=0.8, alpha=0.6, edgecolors="none",
                   rasterized=True, zorder=1)
        ax.scatter(coords[~correct, 0], coords[~correct, 1],
                   c="#D32F2F", s=1.0, alpha=0.8, edgecolors="none",
                   rasterized=True, zorder=2)

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(f"{SAMPLE_DISPLAY[sample]}\n(acc={acc:.1%})",
                     fontsize=9, fontweight="bold", pad=6)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D0D0D0',
               markersize=5, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D32F2F',
               markersize=5, label='Misclassified'),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Spatial Distribution of Classification Errors",
                 fontsize=11, fontweight="bold", y=0.98)

    _save(fig, "fig6_error_maps")


def fig7_domain_accuracy_bar(labels_all, label_names, sample_ids, preds):
    """
    Fig7: Per-domain accuracy bar chart (pooled across all folds),
    with domain count as secondary info. Sorted by accuracy.
    """
    n_domains = len(label_names)

    # Pool all predictions
    all_gt, all_pred = [], []
    for sample in SAMPLES:
        mask = sample_ids == sample
        all_gt.append(labels_all[mask])
        all_pred.append(preds[f"SpatialDomainNet_{sample}"])
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    # Per-domain accuracy and count
    accs = []
    counts = []
    for d in range(n_domains):
        d_mask = all_gt == d
        count = d_mask.sum()
        counts.append(count)
        if count > 0:
            accs.append((all_pred[d_mask] == d).mean())
        else:
            accs.append(0)

    accs = np.array(accs)
    counts = np.array(counts)

    # Sort by accuracy
    order = np.argsort(accs)[::-1]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    fig.subplots_adjust(bottom=0.18)

    x = np.arange(n_domains)
    colors = [DOMAIN_COLORS.get(label_names[i], "#888888") for i in order]

    bars = ax1.bar(x, accs[order], color=colors, edgecolor="black",
                   linewidth=0.3, width=0.7)
    ax1.set_ylabel("Accuracy (pooled)", fontweight="bold")
    ax1.set_ylim(0, 1.12)
    ax1.axhline(y=accs.mean(), color="#666666", linestyle="--", linewidth=0.8,
                label=f"Mean = {accs.mean():.3f}")

    ax1.set_xticks(x)
    ax1.set_xticklabels([label_names[i] for i in order],
                         rotation=50, ha="right", fontsize=6.5)

    # Add count labels on bars — stagger vertically to avoid overlap
    for xi, idx in enumerate(order):
        # Alternate vertical offset to prevent adjacent labels from overlapping
        offset = 0.02 if xi % 2 == 0 else 0.06
        ax1.text(xi, accs[idx] + offset, f"n={counts[idx]}",
                 ha="center", va="bottom", fontsize=5, color="#666666",
                 rotation=45)

    ax1.legend(fontsize=8, loc="lower left")
    ax1.set_title("Per-domain Classification Accuracy (SpatialDomainNet, pooled LOSO)",
                  fontsize=11, fontweight="bold", pad=8)

    _save(fig, "fig7_domain_accuracy")


def fig8_temporal_confusion(labels_all, label_names, sample_ids, preds):
    """
    Fig8: Per-fold accuracy summary — showing how accuracy changes
    across test folds (temporal progression: Ctrl → 1DPI → 3DPI → 7DPI).
    Includes both overall accuracy and ISD-specific accuracy.
    """
    fold_accs = []
    isd_accs = []
    non_isd_accs = []

    isd_indices = [i for i, n in enumerate(label_names) if n.startswith("ISD")]
    non_isd_indices = [i for i, n in enumerate(label_names) if not n.startswith("ISD")]

    for sample in SAMPLES:
        mask = sample_ids == sample
        gt = labels_all[mask]
        pred = preds[f"SpatialDomainNet_{sample}"]

        fold_accs.append((pred == gt).mean())

        # ISD accuracy
        isd_mask = np.isin(gt, isd_indices)
        if isd_mask.sum() > 0:
            isd_accs.append((pred[isd_mask] == gt[isd_mask]).mean())
        else:
            isd_accs.append(np.nan)

        # Non-ISD accuracy
        non_isd_mask = np.isin(gt, non_isd_indices)
        if non_isd_mask.sum() > 0:
            non_isd_accs.append((pred[non_isd_mask] == gt[non_isd_mask]).mean())
        else:
            non_isd_accs.append(np.nan)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    fig.subplots_adjust(top=0.88, bottom=0.12)

    x = np.arange(4)
    w = 0.22

    ax.bar(x - w, fold_accs, w, label="Overall", color="#1565C0",
           edgecolor="black", linewidth=0.3)
    ax.bar(x, non_isd_accs, w, label="Anatomical regions", color="#66BB6A",
           edgecolor="black", linewidth=0.3)

    # Handle NaN for ISD bars
    isd_vals = [v if not np.isnan(v) else 0 for v in isd_accs]
    isd_bar = ax.bar(x + w, isd_vals, w, label="Ischemic domains", color="#D32F2F",
                     edgecolor="black", linewidth=0.3)
    # Mark Ctrl ISD as N/A
    if np.isnan(isd_accs[0]):
        ax.text(0 + w, 0.02, "N/A", ha="center", va="bottom",
                fontsize=7, color="#999999", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([SAMPLE_DISPLAY[s] for s in SAMPLES], fontsize=9)
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_ylim(0, 1.10)
    ax.legend(fontsize=7, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), framealpha=0.9)

    # Value labels — use smaller font and slight horizontal offsets to avoid overlap
    for xi, v in enumerate(fold_accs):
        ax.text(xi - w, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    for xi, v in enumerate(non_isd_accs):
        if not np.isnan(v):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    for xi, v in enumerate(isd_accs):
        if not np.isnan(v):
            ax.text(xi + w, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_title("Classification Accuracy by Domain Category per Fold",
                 fontsize=10, fontweight="bold", pad=10)

    _save(fig, "fig8_temporal_accuracy")


def fig9_method_comparison(labels_all, label_names, coords_all, sample_ids, preds):
    """
    Fig9: Multi-method spatial domain prediction comparison for 3DPI sample.
    2 rows x 3 columns: Ground Truth, SpatialDomainNet (Ours),
    SpatialGAT, DualGCN, ExprOnly, Random Forest.
    Each panel shows spots colored by predicted domain with accuracy subtitle.
    """
    import matplotlib.patches as mpatches

    sample = "3DPI"
    mask = sample_ids == sample
    coords = coords_all[mask]
    gt = labels_all[mask]

    methods = [
        ("Ground Truth", None),
        ("SpatialDomainNet (Ours)", f"SpatialDomainNet_{sample}"),
        ("SpatialGAT", f"SpatialGAT_{sample}"),
        ("DualGCN", f"DualGCN_{sample}"),
        ("ExprOnly", f"ExprOnly_{sample}"),
        ("Random Forest", f"RF_{sample}"),
    ]

    colors = [DOMAIN_COLORS.get(n, "#888888") for n in label_names]

    fig, axes = plt.subplots(2, 3, figsize=(9, 7.5))
    fig.subplots_adjust(hspace=0.30, wspace=0.06, bottom=0.13, top=0.88)

    for idx, (name, pred_key) in enumerate(methods):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        if pred_key is None:
            data = gt
            acc_text = ""
        else:
            data = preds[pred_key]
            acc = (data == gt).mean()
            acc_text = f"\n(acc = {acc:.1%})"

        spot_colors = [colors[int(c)] for c in data]
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=spot_colors, s=1.8, alpha=0.9,
            edgecolors="none", rasterized=True, linewidths=0,
        )

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Title with optional accuracy
        fontweight = "bold" if idx <= 1 else "normal"
        title_color = "#0D47A1" if idx == 1 else "black"
        ax.set_title(f"{name}{acc_text}", fontsize=9,
                     fontweight=fontweight, color=title_color, pad=4)

    # Compact legend at bottom
    legend_order = [
        "CTX1-4", "CTX5", "CTX6", "lCTX4-5", "lCTX6",
        "HIP", "AMY", "PIR",
        "CP", "PAL", "TH", "HY",
        "CS", "FT", "GLS", "LV",
        "ISD1c", "ISD1p", "ISD3c", "ISD3p", "ISD7c", "ISD7p",
    ]
    legend_order = [d for d in legend_order if d in label_names]
    for d in label_names:
        if d not in legend_order:
            legend_order.append(d)

    handles = [mpatches.Patch(facecolor=DOMAIN_COLORS.get(d, "#888888"),
                              edgecolor="none", label=d)
               for d in legend_order]

    fig.legend(
        handles=handles, loc="lower center",
        ncol=11, fontsize=5.5, frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=0.7, handleheight=0.7,
        columnspacing=0.7, labelspacing=0.3,
        handletextpad=0.3,
    )

    fig.suptitle("Multi-method Spatial Domain Prediction Comparison (3 DPI)",
                 fontsize=12, fontweight="bold", y=0.95)

    _save(fig, "fig9_method_comparison")


def main():
    print("=" * 60)
    print("Biological Analysis — Downstream Figures (Fig5-9)")
    print("=" * 60)

    labels_all, label_names, coords_all, sample_ids, preds = load_data()
    print(f"Spots: {len(labels_all)}, Domains: {len(label_names)}")

    print("\nFig5: Per-fold per-domain accuracy heatmap")
    fig5_perfold_heatmap(labels_all, label_names, sample_ids, preds)

    print("\nFig6: Spatial error maps")
    fig6_error_maps(labels_all, label_names, coords_all, sample_ids, preds)

    print("\nFig7: Per-domain accuracy bar chart")
    fig7_domain_accuracy_bar(labels_all, label_names, sample_ids, preds)

    print("\nFig8: Temporal accuracy breakdown (overall vs ISD vs anatomical)")
    fig8_temporal_accuracy = fig8_temporal_confusion(labels_all, label_names,
                                                     sample_ids, preds)

    print("\nFig9: Multi-method spatial domain comparison (3DPI)")
    fig9_method_comparison(labels_all, label_names, coords_all, sample_ids, preds)

    print(f"\nAll figures saved to {FIGDIR}/")


if __name__ == "__main__":
    main()
