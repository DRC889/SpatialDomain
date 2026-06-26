#!/usr/bin/env python3
"""rev13_fig3.py — Regenerate Figure 3 (attention fusion) with non-overlapping
panel labels and recompute the key statistics for Section 3.4.

Panel a: spatial alpha_feature maps (4 samples)
Panel b: alpha_feature by domain category (boxplot, labels placed clearly)
Panel c: alpha_feature time-course by lesion category (per-sample medians, IQR/2)
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from rev_common import (load_data, build_knn_edges, build_feature_edges,
                        train_ae, set_seed, K_SPATIAL, K_FEATURE, WORK)
sys.path.insert(0, str(WORK / "src"))
from model import SpatialDomainAE
from figure_style import apply_style, SAMPLE_DISPLAY, SAMPLES_ORDERED, DOMAIN_CATEGORY_COLORS
apply_style()
FIG = WORK / "figures_rev"; SEED = 42


def categorize(name):
    if name.startswith("ISD") and name.endswith("c"): return "Ischemic core"
    if name.startswith("ISD") and name.endswith("p"): return "Ischemic penumbra"
    if name.startswith("lCTX"): return "Lesioned cortex"
    if name == "GLS": return "Glial scar"
    return "Intact anatomy"


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((a[:, None] > b[None, :]).sum(1)); lt = sum((a[:, None] < b[None, :]).sum(1))
    return (gt - lt) / (len(a) * len(b))


def get_alpha(device):
    expr_all, labels_all, coords_all, sample_ids, _ = load_data()
    with h5py.File(WORK/"data/processed/spatial_dataset.h5") as hf:
        label_names = [s.decode() for s in hf["label_names"][:]]
    data = {}
    for s in SAMPLES_ORDERED:
        m = sample_ids == s
        expr, coords, labels = expr_all[m], coords_all[m], labels_all[m]
        set_seed(SEED)
        sp = build_knn_edges(coords, K_SPATIAL); ft = build_feature_edges(expr, K_FEATURE)
        set_seed(SEED)
        model = SpatialDomainAE(expr.shape[1], 64, 256, 4, 0.3)
        train_ae(model, expr, sp, ft, device, n_epochs=500)
        xt = torch.tensor(expr, dtype=torch.float32, device=device)
        et = torch.tensor(np.stack(sp), dtype=torch.long, device=device)
        eft = torch.tensor(np.stack(ft), dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            _, alpha = model.encode(xt, et, eft)
        af = alpha[:, 1].cpu().numpy()  # alpha_feature
        cats = np.array([categorize(label_names[l]) for l in labels])
        data[s] = dict(coords=coords, alpha=af, cats=cats)
        print(f"  {s}: alpha_feature mean {af.mean():.3f}", flush=True)
        del model; torch.cuda.empty_cache()
    return data


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = get_alpha(device)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.42, wspace=0.30)

    # panel a: maps
    for c, s in enumerate(SAMPLES_ORDERED):
        ax = fig.add_subplot(gs[0, c]); ax.axis("off"); ax.set_aspect("equal")
        co = d[s]["coords"]; af = d[s]["alpha"]
        sc = ax.scatter(co[:, 0], -co[:, 1], c=af, cmap="RdBu_r", vmin=0.30, vmax=0.70, s=4, linewidths=0)
        ax.set_title(SAMPLE_DISPLAY[s], fontsize=10)
        if c == 0: ax.text(-0.05, 1.05, "a", transform=ax.transAxes, fontsize=14, fontweight="bold")
    cax = fig.add_axes([0.92, 0.58, 0.012, 0.30])
    fig.colorbar(sc, cax=cax, label=r"$\alpha_{\mathrm{feature}}$")

    # panel b: boxplot by category (all samples)
    axb = fig.add_subplot(gs[1, :2])
    cats_order = ["Intact anatomy", "Lesioned cortex", "Glial scar", "Ischemic penumbra", "Ischemic core"]
    allcat = {c: [] for c in cats_order}
    for s in SAMPLES_ORDERED:
        for cat in cats_order:
            allcat[cat].extend(d[s]["alpha"][d[s]["cats"] == cat])
    bp = axb.boxplot([allcat[c] for c in cats_order], vert=True, patch_artist=True,
                     showfliers=False, widths=0.6)
    for patch, cat in zip(bp["boxes"], cats_order):
        patch.set_facecolor(DOMAIN_CATEGORY_COLORS.get(cat, "#999")); patch.set_alpha(0.85)
    axb.axhline(0.5, ls="--", c="gray", lw=0.8)
    axb.set_xticks(range(1, len(cats_order)+1))
    axb.set_xticklabels([c.replace(" ", "\n") for c in cats_order], fontsize=8)
    axb.set_ylabel(r"$\alpha_{\mathrm{feature}}$", fontsize=10)
    axb.set_ylim(0.1, 0.95)
    axb.text(-0.08, 1.04, "b", transform=axb.transAxes, fontsize=14, fontweight="bold")

    # panel c: timecourse (per-sample medians) for lesion categories
    axc = fig.add_subplot(gs[1, 2:])
    lc = ["Ischemic core", "Ischemic penumbra", "Glial scar", "Intact anatomy"]
    for cat in lc:
        xs, ys, es = [], [], []
        for i, s in enumerate(SAMPLES_ORDERED):
            vals = d[s]["alpha"][d[s]["cats"] == cat]
            if len(vals) >= 5:
                xs.append(i); ys.append(np.median(vals)); es.append((np.percentile(vals,75)-np.percentile(vals,25))/2)
        if xs:
            axc.errorbar(xs, ys, yerr=es, marker="o", ms=4, capsize=2,
                         color=DOMAIN_CATEGORY_COLORS.get(cat, "#999"), label=cat, lw=1.5)
    axc.axhline(0.5, ls="--", c="gray", lw=0.8)
    axc.set_xticks(range(4)); axc.set_xticklabels([SAMPLE_DISPLAY[s] for s in SAMPLES_ORDERED], fontsize=8)
    axc.set_ylabel(r"median $\alpha_{\mathrm{feature}}$", fontsize=10)
    axc.legend(fontsize=7, loc="upper right", framealpha=0.9)
    axc.text(-0.08, 1.04, "c", transform=axc.transAxes, fontsize=14, fontweight="bold")

    fig.savefig(FIG/"fig3_attention.pdf", bbox_inches="tight")
    fig.savefig(FIG/"fig3_attention.png", dpi=200, bbox_inches="tight")
    print("saved fig3")

    # ---- recompute Section 3.4 stats ----
    print("\n=== Section 3.4 stats (per sample, ischemic vs intact) ===")
    for s in ["1DPI", "3DPI", "7DPI"]:
        af = d[s]["alpha"]; ct = d[s]["cats"]
        intact = af[ct == "Intact anatomy"]
        for cat in ["Ischemic core", "Ischemic penumbra"]:
            v = af[ct == cat]
            if len(v) >= 5 and len(intact) >= 5:
                p = mannwhitneyu(v, intact, alternative="greater").pvalue
                print(f"  {s} {cat}: median {np.median(v):.3f} vs intact {np.median(intact):.3f}  delta={cliffs_delta(v,intact):+.3f}  p={p:.1e}")
    print("\n=== panel b medians by category ===")
    for cat in cats_order:
        print(f"  {cat}: {np.median(allcat[cat]):.3f}")


if __name__ == "__main__":
    main()
