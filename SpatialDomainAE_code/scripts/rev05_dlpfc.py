#!/usr/bin/env python3
"""rev05_dlpfc.py — External-dataset evaluation on human DLPFC.

Runs SpatialDomainAE + ablations (and optionally external baselines) on the
12-sample spatialLIBD DLPFC 10x Visium benchmark (Maynard et al. 2021), using
the SAME preprocessing (library-size normalize + log1p + 3000 HVG), the SAME
dual-graph construction (k_s=15, k_f=20 correlation), and the SAME oracle Leiden
grid as the stroke analysis. Ground truth = manual cortical layer annotations
(layer_guess_reordered: Layer1-6 + WM).

Data already on disk: /data/projects/11003054/changxu/Data/DLPFC/<sid>/
  filtered_feature_bc_matrix.h5  (CellRanger h5, read via h5py to avoid PIL)
  metadata.tsv                   (barcode, imagerow, imagecol, layer_guess_reordered)

Usage:
  python rev05_dlpfc.py --samples 151507,151508 --seeds 0,1,2 --device cuda:0 \
      --methods ours,ablations --out results_rev/dlpfc_g0.csv
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import scanpy as sc
import torch
from pathlib import Path
from rev_common import (build_knn_edges, build_feature_edges, train_ae,
                        cluster_and_eval, set_seed, MODELS, K_SPATIAL, K_FEATURE, WORK)

DLPFC_DIR = Path("/data/projects/11003054/changxu/Data/DLPFC")
ALL_SAMPLES = ["151507","151508","151509","151510","151669","151670",
               "151671","151672","151673","151674","151675","151676"]


def read_counts_h5(path):
    """Read a CellRanger filtered_feature_bc_matrix.h5 via h5py (no PIL)."""
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        data = g["data"][:]; indices = g["indices"][:]; indptr = g["indptr"][:]
        shape = g["shape"][:]  # (n_genes, n_cells)
        barcodes = [b.decode() for b in g["barcodes"][:]]
        genes = [b.decode() for b in g["features"]["name"][:]]
    # CSC: genes x cells -> transpose to cells x genes CSR
    M = sp.csc_matrix((data, indices, indptr), shape=tuple(shape))
    X = M.T.tocsr()
    return X, np.array(barcodes), np.array(genes)


def load_dlpfc_sample(sid, n_hvg=3000):
    meta = pd.read_csv(DLPFC_DIR / sid / "metadata.tsv", sep="\t", dtype=str)
    lay = meta["layer_guess_reordered"].values
    keep = ~pd.isna(lay) & (lay != "NA") & (lay != "")
    meta = meta[keep].reset_index(drop=True)
    coords = meta[["imagerow", "imagecol"]].astype(float).values
    labels_str = meta["layer_guess_reordered"].values
    uniq = sorted(set(labels_str))
    lab2i = {l: i for i, l in enumerate(uniq)}
    labels = np.array([lab2i[l] for l in labels_str])
    bc_meta = meta["barcode"].values

    X, bc_h5, genes = read_counts_h5(DLPFC_DIR / sid / "filtered_feature_bc_matrix.h5")
    pos = {b: i for i, b in enumerate(bc_h5)}
    idx = np.array([pos[b] for b in bc_meta])
    X = X[idx]

    adata = sc.AnnData(X=X)
    adata.var_names = genes; adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
    adata = adata[:, adata.var.highly_variable]
    expr = np.asarray(adata.X.todense() if sp.issparse(adata.X) else adata.X, dtype=np.float32)
    return expr, coords, labels, len(uniq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=",".join(ALL_SAMPLES))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--methods", default="ours,ablations")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--out", default="results_rev/dlpfc.csv")
    args = ap.parse_args()
    samples = args.samples.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device)
    want = args.methods.split(",")
    model_names = ["SpatialDomainAE"]
    if "ablations" in want:
        model_names += ["SpatialGATAE", "ExprOnlyAE"]
    out = WORK / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for sid in samples:
            expr, coords, labels, ncl_gt = load_dlpfc_sample(sid)
            n_genes = expr.shape[1]
            set_seed(seed)
            sp_e = build_knn_edges(coords, K_SPATIAL)
            ft_e = build_feature_edges(expr, K_FEATURE)
            for name in model_names:
                t0 = time.time(); set_seed(seed)
                model = MODELS[name](n_genes)
                emb = train_ae(model, expr, sp_e, ft_e, device, n_epochs=args.epochs)
                ari, nmi, ncl, res, _ = cluster_and_eval(emb, labels, seed=seed)
                rows.append(dict(seed=seed, sample=sid, method=name, n_spots=len(labels),
                                 n_layers=ncl_gt, ari=ari, nmi=nmi, n_clusters=ncl,
                                 res=res, sec=round(time.time()-t0,1)))
                print(f"  seed={seed} {sid} {name:16s} ARI={ari:.3f} NMI={nmi:.3f} "
                      f"layers={ncl_gt} ({time.time()-t0:.0f}s)", flush=True)
                del model; torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
