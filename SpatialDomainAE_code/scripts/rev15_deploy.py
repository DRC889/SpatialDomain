#!/usr/bin/env python3
"""rev15_deploy.py — Deployment-realistic resolution selection.

Addresses the panel concern that all ARIs use an ORACLE (label-selected) Leiden
resolution. For every method we additionally report, on the same embeddings:
  - oracle : best-ARI resolution on the grid (uses labels; upper bound)
  - fixed  : a single fixed resolution gamma=1.0 for all samples/methods
  - silh   : resolution chosen by maximizing the silhouette score (LABEL-FREE,
             a realistic unsupervised selection criterion)
Run with the clean env (PYTHONPATH not needed; baselines import directly).
"""
import argparse, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from rev_common import (load_data, build_knn_edges, build_feature_edges, train_ae,
                        set_seed, MODELS, K_SPATIAL, K_FEATURE, RES_GRID, WORK)
import rev04_baselines as b
import rev07_spatialmgcn as smg


def cluster_modes(emb, labels, seed=0):
    adata = sc.AnnData(X=emb.astype(np.float32))
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=15, random_state=seed)
    rows = {}
    best_ari, fixed_ari, sil_best, sil_ari = -1, None, -1, None
    for res in RES_GRID:
        sc.tl.leiden(adata, resolution=res, key_added="cl", random_state=seed)
        pred = adata.obs["cl"].astype(int).values
        ari = adjusted_rand_score(labels, pred)
        if ari > best_ari: best_ari = ari
        if abs(res - 1.0) < 1e-9: fixed_ari = ari
        if len(set(pred)) > 1:
            try:
                s = silhouette_score(emb, pred)
                if s > sil_best: sil_best, sil_ari = s, ari
            except Exception: pass
    return best_ari, fixed_ari, (sil_ari if sil_ari is not None else best_ari)


def get_emb(method, expr, coords, sp, ft, device, seed, n_genes):
    if method == "SpatialDomainAE":
        set_seed(seed); m = MODELS[method](n_genes)
        return train_ae(m, expr, sp, ft, device, n_epochs=500)
    if method == "STAGATE": return b.run_stagate(expr, coords, device, seed)
    if method == "GraphST": return b.run_graphst(expr, coords, device, seed)
    if method == "SpaGCN":  return b.run_spagcn(expr, coords, device, seed)
    if method == "Spatial-MGCN": return smg.run_smgcn(expr, coords, device, seed, epochs=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="SpatialDomainAE,Spatial-MGCN,GraphST,SpaGCN,STAGATE")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results_rev/deploy.csv")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]; device = torch.device(args.device)
    expr_all, labels_all, coords_all, sample_ids, _ = load_data()
    samples = sorted(set(sample_ids)); n_genes = expr_all.shape[1]
    out = WORK / args.out; out.parent.mkdir(parents=True, exist_ok=True); rows = []
    for seed in seeds:
        for s in samples:
            mask = sample_ids == s; expr, coords, labels = expr_all[mask], coords_all[mask], labels_all[mask]
            set_seed(seed); sp = build_knn_edges(coords, K_SPATIAL); ft = build_feature_edges(expr, K_FEATURE)
            for meth in args.methods.split(","):
                t0 = time.time()
                try:
                    emb = get_emb(meth, expr, coords, sp, ft, device, seed, n_genes)
                    orc, fix, sil = cluster_modes(emb, labels, seed)
                    rows.append(dict(seed=seed, sample=s, method=meth, oracle=orc, fixed_g1=fix, silhouette=sil))
                    print(f"  seed={seed} {s:5s} {meth:15s} oracle={orc:.3f} fixed={fix:.3f} silh={sil:.3f} ({time.time()-t0:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  seed={seed} {s:5s} {meth:15s} FAIL {repr(e)[:100]}", flush=True)
                torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
