#!/usr/bin/env python3
"""rev08_fig1.py — Regenerate Fig 1 architecture schematic.

Key fix: the MSE reconstruction loss is drawn as an EXTERNAL training objective
that compares the decoder's reconstructed expression to the input expression,
rather than as a block inside the decoder.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures_rev"
OUT.mkdir(parents=True, exist_ok=True)

C = dict(inp="#dbe9f6", graph="#e7f0d8", enc="#d8e4f0", gat="#cfe3f7",
         fuse="#fde6c8", lat="#e3d6ef", dec="#d8e4f0", out="#d7efe0",
         loss="#fde0e0")


def box(ax, x, y, w, h, text, fc, fs=8.5, ec="#5b6b7a", lw=1.2, style="round"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size=0.03",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, zorder=3)
    return (x, y, w, h)


def arrow(ax, p0, p1, color="#37474f", style="-|>", lw=1.4, ls="-", rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12,
                        linewidth=lw, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)


def rmid(b): return (b[0]+b[2], b[1]+b[3]/2)   # right-middle
def lmid(b): return (b[0], b[1]+b[3]/2)        # left-middle
def tmid(b): return (b[0]+b[2]/2, b[1]+b[3])   # top-middle
def bmid(b): return (b[0]+b[2]/2, b[1])        # bottom-middle


def main():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")

    yC = 2.6  # main row center
    h = 0.9
    inp  = box(ax, 0.2, yC-h/2, 1.5, h, "Input\n$\\mathbf{X}$ (HVG expr.)\n+ coords $\\mathbf{c}_i$", C["inp"])
    grf  = box(ax, 2.05, yC-h/2, 1.7, h, "Dual-graph\nconstruction\n$\\mathcal{G}_s$ (k=15)\n$\\mathcal{G}_f$ (k=20, corr)", C["graph"], fs=8)
    enc  = box(ax, 4.1, yC-h/2, 1.4, h, "Expression\nencoder\n(MLP)", C["enc"])
    # two GAT branches
    gA   = box(ax, 5.95, yC+0.30, 1.35, 0.62, "GAT on $\\mathcal{G}_s$", C["gat"], fs=8)
    gB   = box(ax, 5.95, yC-0.92, 1.35, 0.62, "GAT on $\\mathcal{G}_f$", C["gat"], fs=8)
    fuse = box(ax, 7.65, yC-h/2, 1.5, h, "Per-spot\nattention fusion\n$(\\alpha_s,\\alpha_f)$", C["fuse"], fs=8)
    lat  = box(ax, 9.5, yC-0.38, 0.95, 0.76, "Latent\n$\\mathbf{z}$", C["lat"])
    dec  = box(ax, 10.8, yC-h/2, 1.3, h, "Decoder\n(MLP)", C["dec"])
    rec  = box(ax, 12.2, yC-0.38, 0.7, 0.76, "$\\hat{\\mathbf{X}}$", C["dec"], fs=10)

    # main flow arrows
    arrow(ax, rmid(inp), lmid(grf)); arrow(ax, rmid(grf), lmid(enc))
    arrow(ax, rmid(enc), lmid(gA), rad=0.15); arrow(ax, rmid(enc), lmid(gB), rad=-0.15)
    arrow(ax, rmid(gA), (fuse[0], fuse[1]+fuse[3]*0.7), rad=-0.15)
    arrow(ax, rmid(gB), (fuse[0], fuse[1]+fuse[3]*0.3), rad=0.15)
    arrow(ax, rmid(fuse), lmid(lat)); arrow(ax, rmid(lat), lmid(dec))
    arrow(ax, rmid(dec), lmid(rec))

    # ---- MSE loss as EXTERNAL objective (the external-objective fix) ----
    loss = box(ax, 11.0, 0.45, 1.5, 0.66,
               "MSE loss\n$\\|\\mathbf{X}-\\hat{\\mathbf{X}}\\|^2$", C["loss"], fs=8,
               ec="#c0392b", lw=1.5)
    # reconstruction -> loss
    arrow(ax, bmid(rec), (loss[0]+loss[2]*0.75, loss[1]+loss[3]), color="#c0392b", ls="-", rad=0.0)
    # input expression -> loss (long external path along the bottom)
    ax.plot([inp[0]+inp[2]/2, inp[0]+inp[2]/2], [inp[1], 0.78], color="#c0392b", lw=1.3, ls="--", zorder=0)
    ax.plot([inp[0]+inp[2]/2, loss[0]], [0.78, 0.78], color="#c0392b", lw=1.3, ls="--", zorder=0)
    arrow(ax, (loss[0]-0.02, 0.78), (loss[0]+loss[2]*0.25, loss[1]+loss[3]*0.5), color="#c0392b", ls="--", rad=0.0)
    ax.text(6.5, 0.62, "reconstruction objective computed externally: decoder output vs.\\,input expression",
            ha="center", va="center", fontsize=7.5, color="#c0392b", style="italic")

    # ---- outputs from latent ----
    out1 = box(ax, 9.15, 4.05, 1.9, 0.7, "Leiden clustering\n$\\rightarrow$ spatial domains", C["out"], fs=8)
    out2 = box(ax, 7.0, 4.05, 1.9, 0.7, "$\\alpha_f$ map\n(disorganization)", C["out"], fs=8)
    arrow(ax, tmid(lat), bmid(out1), color="#2e7d57", rad=-0.1)
    arrow(ax, (fuse[0]+fuse[2]*0.5, fuse[1]+fuse[3]), bmid(out2), color="#2e7d57", rad=0.1)

    ax.text(0.2, 4.7, "SpatialDomainAE", fontsize=14, fontweight="bold", color="#2c3e50")
    fig.savefig(OUT / "fig1_framework.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_framework.png", dpi=200, bbox_inches="tight")
    print("saved", OUT / "fig1_framework.pdf")


if __name__ == "__main__":
    main()
