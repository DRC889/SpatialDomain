#!/usr/bin/env python3
"""rev01_multiseed.py — Multi-seed robustness.

Runs SpatialDomainAE + ablations across multiple random seeds on all 4 samples,
so we can report mean +/- SD over seeds and run paired statistical tests, rather
than relying on a single seed over only 4 samples.

Usage:
  python rev01_multiseed.py --seeds 0,1,2,3,4 --device cuda:0 --epochs 500 --out results_rev/multiseed_gpu0.csv
"""
import argparse, time
import numpy as np
import pandas as pd
import torch
from rev_common import (load_data, build_knn_edges, build_feature_edges,
                        train_ae, cluster_and_eval, set_seed, MODELS,
                        K_SPATIAL, K_FEATURE, WORK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--out", default="results_rev/multiseed.csv")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device)
    expr_all, labels_all, coords_all, sample_ids, _ = load_data()
    samples = sorted(set(sample_ids))
    n_genes = expr_all.shape[1]
    print(f"device={device} seeds={seeds} samples={samples} genes={n_genes}", flush=True)

    rows = []
    out = WORK / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for sample in samples:
            m = sample_ids == sample
            expr, coords, labels = expr_all[m], coords_all[m], labels_all[m]
            set_seed(seed)
            sp = build_knn_edges(coords, K_SPATIAL)
            ft = build_feature_edges(expr, K_FEATURE)
            for name, ctor in MODELS.items():
                t0 = time.time()
                set_seed(seed)
                model = ctor(n_genes)
                emb = train_ae(model, expr, sp, ft, device, n_epochs=args.epochs)
                ari, nmi, ncl, res, _ = cluster_and_eval(emb, labels, seed=seed)
                dt = time.time() - t0
                rows.append(dict(seed=seed, sample=sample, method=name,
                                 ari=ari, nmi=nmi, n_clusters=ncl, res=res, sec=round(dt, 1)))
                print(f"  seed={seed} {sample:5s} {name:16s} ARI={ari:.3f} NMI={nmi:.3f} "
                      f"ncl={ncl} res={res} ({dt:.0f}s)", flush=True)
                del model; torch.cuda.empty_cache()
            # incremental save
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
