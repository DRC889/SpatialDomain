#!/usr/bin/env python3
"""rev03_sensitivity.py — Graph-construction sensitivity.

Varies one construction choice at a time around the defaults
(k_spatial=15, k_feature=20, feature metric=correlation) and reports ARI/NMI,
to justify the chosen values and show robustness.

Axes:
  ks      : k_spatial in {6,10,15,20,30}
  kf      : k_feature in {10,15,20,30,50}
  metric  : feature similarity in {correlation, cosine, euclidean}

Usage:
  python rev03_sensitivity.py --axis ks --seeds 0,1,2 --device cuda:0 \
      --out results_rev/sens_ks.csv
"""
import argparse, time
import numpy as np
import pandas as pd
import torch
from rev_common import (load_data, build_knn_edges, build_feature_edges,
                        train_ae, cluster_and_eval, set_seed, MODELS, WORK)

AXES = {
    "ks":     ("k_spatial", [6, 10, 15, 20, 30]),
    "kf":     ("k_feature", [10, 15, 20, 30, 50]),
    "metric": ("metric",    ["correlation", "cosine", "euclidean"]),
}
DEF_KS, DEF_KF, DEF_METRIC = 15, 20, "correlation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=list(AXES))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device)
    pname, values = AXES[args.axis]
    out = WORK / (args.out or f"results_rev/sens_{args.axis}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    expr_all, labels_all, coords_all, sample_ids, _ = load_data()
    samples = sorted(set(sample_ids))
    n_genes = expr_all.shape[1]
    rows = []
    for seed in seeds:
        for sample in samples:
            m = sample_ids == sample
            expr, coords, labels = expr_all[m], coords_all[m], labels_all[m]
            for val in values:
                ks, kf, metric = DEF_KS, DEF_KF, DEF_METRIC
                if pname == "k_spatial": ks = val
                elif pname == "k_feature": kf = val
                elif pname == "metric": metric = val
                t0 = time.time(); set_seed(seed)
                sp = build_knn_edges(coords, ks)
                ft = build_feature_edges(expr, kf, metric=metric)
                set_seed(seed)
                model = MODELS["SpatialDomainAE"](n_genes)
                emb = train_ae(model, expr, sp, ft, device, n_epochs=args.epochs)
                ari, nmi, ncl, res, _ = cluster_and_eval(emb, labels, seed=seed)
                rows.append(dict(seed=seed, sample=sample, axis=args.axis,
                                 param=pname, value=str(val), ari=ari, nmi=nmi,
                                 n_clusters=ncl, res=res, sec=round(time.time()-t0,1)))
                print(f"  seed={seed} {sample:5s} {pname}={str(val):11s} "
                      f"ARI={ari:.3f} NMI={nmi:.3f} ({time.time()-t0:.0f}s)", flush=True)
                del model; torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
