#!/usr/bin/env python3
"""rev04_baselines.py — External baselines across seeds.

Runs STAGATE, GraphST, SpaGCN per sample across multiple seeds with the SAME
preprocessing (3000-HVG normalized matrix), the SAME oracle Leiden grid, and
explicit per-method settings, so Table 1 can report mean+/-SD over seeds for
every method on equal footing.

Run with the baseline env:
  # (run inside an environment with PyG + STAGATE_pyG + GraphST + SpaGCN installed)
  python rev04_baselines.py --methods STAGATE,GraphST,SpaGCN \
      --seeds 0,1,2 --device cuda:1 --out results_rev/baselines.csv
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from rev_common import load_data, cluster_and_eval, set_seed, K_SPATIAL, WORK


def run_stagate(expr, coords, device, seed, n_epochs=500):
    import STAGATE_pyG
    adata = sc.AnnData(X=expr.copy()); adata.obsm["spatial"] = coords.copy()
    coor = pd.DataFrame(coords, columns=["x", "y"], index=adata.obs_names)
    nbrs = NearestNeighbors(n_neighbors=K_SPATIAL + 1).fit(coor)
    dist, idx = nbrs.kneighbors(coor)
    r, c, d = [], [], []
    for i in range(idx.shape[0]):
        for j in range(1, idx.shape[1]):
            r.append(coor.index[i]); c.append(coor.index[idx[i, j]]); d.append(dist[i, j])
    adata.uns["Spatial_Net"] = pd.DataFrame({"Cell1": r, "Cell2": c, "Distance": d})
    adata = STAGATE_pyG.train_STAGATE(adata, hidden_dims=[512, 30],
                                      n_epochs=n_epochs, lr=0.001,
                                      random_seed=seed, device=device.type)
    return adata.obsm["STAGATE"]


def run_graphst(expr, coords, device, seed, n_epochs=600):
    from GraphST import GraphST as G
    adata = sc.AnnData(X=expr.copy()); adata.obsm["spatial"] = coords.copy()
    adata.var["highly_variable"] = True
    dev = f"cuda:{device.index}" if device.type == "cuda" else "cpu"
    model = G.GraphST(adata, device=torch.device(dev), epochs=n_epochs,
                      dim_output=64, datatype="10X", random_seed=seed)
    adata = model.train()
    return adata.obsm["emb"]


def run_spagcn(expr, coords, device, seed, n_pcs=50):
    import SpaGCN as spg
    from SpaGCN.calculate_adj import calculate_adj_matrix
    adata = sc.AnnData(X=expr.copy()); adata.obsm["spatial"] = coords.copy()
    adata.obs["x_pixel"] = coords[:, 0].astype(int)
    adata.obs["y_pixel"] = coords[:, 1].astype(int)
    sc.pp.pca(adata, n_comps=n_pcs)
    adj = calculate_adj_matrix(x=adata.obs["x_pixel"].tolist(),
                               y=adata.obs["y_pixel"].tolist(), histology=False)
    l = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    clf = spg.SpaGCN(); clf.set_l(l)
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    clf.train(adata, adj, init_spa=True, init="kmeans", n_clusters=20,
              tol=5e-3, lr=0.05, max_epochs=200)
    z, q = clf.model.predict(clf.embed, clf.adj_exp)
    return z.detach().cpu().numpy()


RUNNERS = {"STAGATE": run_stagate, "GraphST": run_graphst, "SpaGCN": run_spagcn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="STAGATE,GraphST,SpaGCN")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dataset", default="stroke", choices=["stroke", "dlpfc"])
    ap.add_argument("--samples", default="")
    ap.add_argument("--out", default="results_rev/baselines.csv")
    args = ap.parse_args()
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device)
    if args.dataset == "stroke":
        expr_all, labels_all, coords_all, sample_ids, _ = load_data()
        samples = sorted(set(sample_ids))
        def get(s):
            m = sample_ids == s
            return expr_all[m], coords_all[m], labels_all[m]
    else:
        import rev05_dlpfc as d
        samples = args.samples.split(",") if args.samples else d.ALL_SAMPLES
        def get(s):
            e, c, l, _ = d.load_dlpfc_sample(s); return e, c, l
    out = WORK / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for sample in samples:
            expr, coords, labels = get(sample)
            for meth in methods:
                t0 = time.time(); set_seed(seed)
                try:
                    emb = RUNNERS[meth](expr, coords, device, seed)
                    ari, nmi, ncl, res, _ = cluster_and_eval(emb, labels, seed=seed)
                    rows.append(dict(seed=seed, sample=sample, method=meth,
                                     ari=ari, nmi=nmi, n_clusters=ncl, res=res,
                                     sec=round(time.time()-t0, 1)))
                    print(f"  seed={seed} {sample:5s} {meth:8s} ARI={ari:.3f} "
                          f"NMI={nmi:.3f} ({time.time()-t0:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  seed={seed} {sample:5s} {meth:8s} FAIL {repr(e)[:120]}", flush=True)
                torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
