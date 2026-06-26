#!/usr/bin/env python3
"""rev09_analyze.py -- Reproduce the manuscript tables from results/*.csv.

Recomputes, directly from the saved per-seed CSVs, every number reported in:
  * Table 1  (benchmark, all methods over the SAME 10 seeds; paired Wilcoxon 10 vs 10)
  * Supplementary Table S2  (graph-structure ablation, our own variants, 20 runs)
  * Table 2  (feature-graph content controls, 10 seeds, paired + BH correction)
  * Supplementary Table S1  (deployment-realistic resolution selection)
  * Section 3.8  (external DLPFC benchmark)

Run after the experiment scripts (rev01/rev04/rev07/rev17/rev02/rev15/rev05) have
populated results/.  No GPU needed -- this is pure pandas/scipy aggregation.
"""
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
SAMP = ["Ctrl", "1DPI", "3DPI", "7DPI"]


def load(*names, src=False):
    """Concat the given result CSVs (skipping any that are absent).
    If src=True, tag each row with the file it came from (used to identify
    the two repeat-runs of our model for the 20-run ablation)."""
    frames = []
    for n in names:
        p = R / n
        if p.exists():
            d = pd.read_csv(p)
            if src:
                d["__src"] = n
            frames.append(d)
    if not frames:
        raise FileNotFoundError(f"none of {names} found under {R}")
    return pd.concat(frames, ignore_index=True)


def fmt(x):
    return f"{np.mean(x):.3f}±{np.std(x, ddof=0):.3f}"


def per_seed(df, method):
    """Collapse to one value per seed (mean over repeats / over the 4 samples)."""
    d = df[(df.method == method) & (df["sample"].isin(SAMP))]
    s3 = d[d["sample"] == "3DPI"].groupby("seed").ari.mean()
    pm = d.groupby(["seed", "sample"]).ari.mean().groupby("seed").mean()
    pn = d.groupby(["seed", "sample"]).nmi.mean().groupby("seed").mean()
    return s3, pm, pn


def paired_p(a, b):
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 2:
        return float("nan"), len(common)
    _, p = wilcoxon(a.loc[common].values, b.loc[common].values)
    return p, len(common)


def table1():
    print("=" * 80)
    print("TABLE 1 -- benchmark (same 10 seeds for every method; mean +/- SD)")
    print(f"{'Method':24s}{'3DPI ARI':>13s}{'Mean ARI':>13s}{'Mean NMI':>13s}{'p(3DPI vs ours)':>18s}")
    ours = load("cleanenv_multiseed_g0.csv", "cleanenv_multiseed_g1.csv")
    base = load("baselines.csv", "baselines_more.csv")
    mgcn = load("spatialmgcn_stroke.csv", "spatialmgcn_more.csv")
    spam = load("spamask_stroke.csv", "spamask_more.csv")
    o3, _, _ = per_seed(ours, "SpatialDomainAE")
    rows = [("SpatialDomainAE (Ours)", ours, "SpatialDomainAE"),
            ("Spatial-MGCN", mgcn, "Spatial-MGCN"), ("SpaMask", spam, "SpaMask"),
            ("STAGATE", base, "STAGATE"), ("GraphST", base, "GraphST"), ("SpaGCN", base, "SpaGCN")]
    for name, df, m in rows:
        s3, pm, pn = per_seed(df, m)
        p = "--" if m == "SpatialDomainAE" else (lambda pv, n: f"{pv:.3f} (n={n})")(*paired_p(o3, s3))
        print(f"{name:24s}{fmt(s3):>13s}{fmt(pm):>13s}{fmt(pn):>13s}{p:>18s}")


def ablation():
    print("\n" + "=" * 80)
    print("SUPPLEMENTARY TABLE S2 -- ablation (our variants, 20 runs = 10 seeds x 2)")
    print(f"{'Variant':28s}{'3DPI ARI':>13s}{'Mean ARI':>13s}{'Mean NMI':>13s}")
    df = load("cleanenv_multiseed_g0.csv", "cleanenv_multiseed_g1.csv",
              "multiseed_g0.csv", "multiseed_g1.csv", src=True)
    df = df[df["sample"].isin(SAMP)]
    perseed_mean = {}
    for m in ["SpatialDomainAE", "SpatialGATAE", "ExprOnlyAE"]:
        d = df[df.method == m]
        s3 = d[d["sample"] == "3DPI"].ari.values                       # 20 runs, for SD
        pm = d.groupby(["__src", "seed"]).ari.mean()                   # 20 four-sample means, for SD
        pn = d.groupby(["__src", "seed"]).nmi.mean()
        # for the paired test, collapse the 2 repeats per seed -> 10 seed-level values
        perseed_mean[m] = d.groupby(["seed", "sample"]).ari.mean().groupby("seed").mean()
        print(f"{m:28s}{fmt(s3):>13s}{fmt(pm.values):>13s}{fmt(pn.values):>13s}")
    for m in ["SpatialGATAE", "ExprOnlyAE"]:
        a, b = perseed_mean["SpatialDomainAE"], perseed_mean[m]
        common = sorted(set(a.index) & set(b.index))
        _, p = wilcoxon(a.loc[common].values, b.loc[common].values)
        print(f"  dual vs {m:14s} mean-ARI paired p = {p:.3f} (n={len(common)} seeds)")


def feature_graph():
    print("\n" + "=" * 80)
    print("TABLE 2 -- feature-graph content controls (3DPI, 10 seeds; paired vs full + BH)")
    try:
        fg = load("feature_graph.csv", "feature_graph_more.csv")
    except FileNotFoundError:
        print("  (feature_graph CSVs absent -- run rev02_feature_graph.py)"); return
    if "variant" not in fg.columns:
        print("  (no 'variant' column)"); return
    fg = fg[fg["sample"] == "3DPI"]
    full = fg[fg.variant == "full"].set_index("seed").ari
    # the four edge-CONTENT contrasts (BH-corrected); spatial_only is the remove-graph control
    content = ["prune_lr", "local_feat", "dist_matched", "rand_feat"]
    raw = {}
    for v in fg.variant.unique():
        if v == "full":
            continue
        b = fg[fg.variant == v].set_index("seed").ari
        c = sorted(set(full.index) & set(b.index))
        if len(c) < 2:
            continue
        _, p = wilcoxon(full.loc[c].values, b.loc[c].values)
        raw[v] = (b.mean(), p)
    # Benjamini-Hochberg step-up over the four content contrasts only
    cpv = np.array([raw[v][1] for v in content if v in raw])
    order = np.argsort(cpv); m = len(cpv)
    adj = (cpv[order] * m / (np.arange(m) + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1].clip(0, 1)
    bh = {};
    for i, idx in enumerate(order):
        bh[[v for v in content if v in raw][idx]] = adj[i]
    print(f"  full (correlation KNN): {full.mean():.3f}")
    for v in content + ["spatial_only"]:
        if v not in raw:
            continue
        mn, p = raw[v]
        tag = f"BH-adj={bh[v]:.3f}" if v in bh else "(remove-graph control, not BH-corrected)"
        print(f"  {v:26s} {mn:.3f}   p={p:.3f}  {tag}")


def deploy():
    print("\n" + "=" * 80)
    print("SUPPLEMENTARY TABLE S1 -- resolution selection without oracle labels")
    try:
        dp = load("deploy.csv")
    except FileNotFoundError:
        print("  (deploy.csv absent -- run rev15_deploy.py)"); return
    print(dp.to_string(index=False))


def dlpfc():
    print("\n" + "=" * 80)
    print("SECTION 3.8 -- external DLPFC benchmark (mean ARI over seeds)")
    try:
        alld = pd.concat([load("dlpfc_ours.csv"), load("dlpfc_baselines.csv")], ignore_index=True)
    except FileNotFoundError:
        print("  (dlpfc CSVs absent -- run rev05_dlpfc.py)"); return
    if "method" not in alld.columns:
        print("  (no 'method' column)"); return
    for m in sorted(alld.method.unique()):
        sub = alld[alld.method == m]
        ps = sub.groupby("seed").ari.mean() if "seed" in sub.columns else sub.ari
        print(f"  {m:18s} mean ARI = {ps.mean():.3f}+/-{ps.std(ddof=0):.3f}")


if __name__ == "__main__":
    table1()
    ablation()
    feature_graph()
    deploy()
    dlpfc()
    print("\nDone. These numbers reproduce Table 1, Table 2, and Supplementary Tables S1-S2.")
