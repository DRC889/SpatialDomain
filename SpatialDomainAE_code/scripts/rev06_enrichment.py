#!/usr/bin/env python3
"""rev06_enrichment.py — Pathway enrichment for lesion domains.

Groups SpatialDomainAE 3DPI clusters by lesion domain category (using the
cluster->domain purity mapping), collects up-regulated DEGs per category, and
runs GO/KEGG/Reactome over-representation analysis with Enrichr (mouse).
Outputs per-category enrichment tables and a summary dot/bar figure.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import gseapy as gp
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path(__file__).resolve().parent.parent
# DEG / cluster-mapping CSVs (deg_3DPI_clusters.csv, cluster_domain_mapping_3DPI.csv)
RES = WORK / "results" if (WORK / "results").exists() else WORK / "results_rev"
OUT = WORK / "results_rev" / "enrichment"; OUT.mkdir(parents=True, exist_ok=True)
FIGOUT = WORK / "figures_rev"; FIGOUT.mkdir(parents=True, exist_ok=True)

# Lesion domain categories -> dominant-domain labels in the mapping
CATEGORIES = {
    "Ischemic core (ISD3c)":   ["ISD3c"],
    "Penumbra (ISD3p)":        ["ISD3p"],
    "Glial scar (GLS)":        ["GLS"],
    "Lesioned cortex (lCTX)":  ["lCTX6", "lCTX4-5"],
}
GENE_SETS = ["GO_Biological_Process_2021", "KEGG_2019_Mouse", "Reactome_2022"]
TOP_GENES = 100


def main():
    deg = pd.read_csv(RES / "deg_3DPI_clusters.csv")
    mp = pd.read_csv(RES / "cluster_domain_mapping_3DPI.csv")
    dom_of = dict(zip(mp.cluster, mp.dominant_domain))
    deg["domain"] = deg.cluster.map(dom_of)

    all_terms = []
    for cat, doms in CATEGORIES.items():
        sub = deg[deg.domain.isin(doms)]
        sub = sub[(sub.pval_adj < 0.05) & (sub.log2fc > 1.0)]
        genes = (sub.sort_values("score", ascending=False)
                    .drop_duplicates("gene").gene.head(TOP_GENES).tolist())
        if len(genes) < 5:
            print(f"[skip] {cat}: only {len(genes)} genes"); continue
        print(f"\n=== {cat}: {len(genes)} up-genes (clusters {sorted(sub.cluster.unique())}) ===")
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=GENE_SETS,
                             organism="mouse", outdir=None)
            r = enr.results.copy()
        except Exception as e:
            print("  enrichr fail:", repr(e)[:150]); continue
        r["category"] = cat
        r = r.sort_values("Adjusted P-value")
        r.to_csv(OUT / f"enrichr_{doms[0]}.csv", index=False)
        sig = r[r["Adjusted P-value"] < 0.05]
        print(f"  {len(sig)} significant terms (FDR<0.05). Top:")
        for _, row in r.head(6).iterrows():
            print(f"    [{row.Gene_set.split('_')[0]:>8s}] {row.Term[:55]:55s} FDR={row['Adjusted P-value']:.1e}")
        all_terms.append(r)

    if all_terms:
        full = pd.concat(all_terms, ignore_index=True)
        full.to_csv(OUT / "enrichr_all.csv", index=False)
        make_figure(full)
        print(f"\nSaved enrichment tables + figure to {OUT} and {FIGOUT}")


def make_figure(full):
    cats = list(CATEGORIES.keys())
    fig, axes = plt.subplots(1, len(cats), figsize=(5*len(cats), 4.2), squeeze=False)
    for ax, cat in zip(axes[0], cats):
        sub = full[(full.category == cat) & (full["Adjusted P-value"] < 0.05)].copy()
        sub = sub.sort_values("Adjusted P-value").head(8)
        if len(sub) == 0:
            ax.set_title(cat, fontsize=10); ax.axis("off"); continue
        sub["nlp"] = -np.log10(sub["Adjusted P-value"])
        terms = [t[:42] for t in sub.Term][::-1]
        ax.barh(range(len(sub)), sub.nlp[::-1], color="#c0392b")
        ax.set_yticks(range(len(sub))); ax.set_yticklabels(terms, fontsize=7)
        ax.set_xlabel(r"$-\log_{10}$ FDR", fontsize=9)
        ax.set_title(cat, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGOUT / "fig_enrichment.pdf", bbox_inches="tight")
    fig.savefig(FIGOUT / "fig_enrichment.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
