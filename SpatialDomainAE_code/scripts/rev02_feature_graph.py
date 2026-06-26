#!/usr/bin/env python3
"""rev02_feature_graph.py — Feature-graph necessity.

Tests whether the long-range *transcriptomic* feature edges are necessary, by
replacing the feature graph in SpatialDomainAE with controlled variants and
comparing ARI to the full dual graph:

  full          : correlation-KNN feature graph (the model's design)
  prune_lr      : feature edges with physical distance below the per-sample
                  median feature-edge distance (long-range edges removed)
  local_feat    : feature-KNN restricted to spatially proximal candidates
                  (feature similarity among the 50 nearest spatial neighbors)
                  -> short-range feature edges only
  rand_feat     : degree-matched random feature graph (uniform random targets)
  dist_matched  : random targets resampled to match the *spatial-distance*
                  distribution of the real feature edges (controls for
                  long-range-ness; isolates transcriptomic content)
  spatial_only  : SpatialGATAE (no feature graph at all)  [reference]

Usage:
  python rev02_feature_graph.py --seeds 0,1,2 --device cuda:0 --epochs 500 \
      --out results_rev/feature_graph_g0.csv
"""
import argparse, time
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from rev_common import (load_data, build_knn_edges, build_feature_edges,
                        train_ae, cluster_and_eval, set_seed, MODELS,
                        K_SPATIAL, K_FEATURE, WORK)


def edge_phys_dist(coords, src, dst):
    d = coords[src] - coords[dst]
    return np.sqrt((d * d).sum(1))


def variant_prune_lr(coords, ft_src, ft_dst):
    """Keep only feature edges shorter than the median feature-edge distance."""
    dist = edge_phys_dist(coords, ft_src, ft_dst)
    keep = dist <= np.median(dist)
    return ft_src[keep], ft_dst[keep]


def variant_local_feat(expr, coords, k_feat, n_spatial_cand=50, rng=None):
    """Feature-KNN restricted to the n_spatial_cand nearest spatial neighbors."""
    n = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(n_spatial_cand + 1, n)).fit(coords)
    _, sp_idx = nbrs.kneighbors(coords)
    src, dst = [], []
    for i in range(n):
        cand = sp_idx[i, 1:]  # drop self
        # correlation distance to candidates
        ei = expr[i:i + 1]
        ec = expr[cand]
        d = cdist(ei, ec, metric="correlation")[0]
        order = np.argsort(d)[:k_feat]
        for j in cand[order]:
            src.append(i); dst.append(int(j))
    return np.array(src, np.int64), np.array(dst, np.int64)


def variant_rand(n, n_edges, rng):
    """Degree-matched uniform-random directed edges (no self loops)."""
    src = rng.integers(0, n, size=n_edges)
    dst = rng.integers(0, n, size=n_edges)
    bad = src == dst
    while bad.any():
        dst[bad] = rng.integers(0, n, size=int(bad.sum()))
        bad = src == dst
    return src.astype(np.int64), dst.astype(np.int64)


def variant_dist_matched(coords, ft_src, ft_dst, rng, n_bins=20):
    """Random edges whose physical-distance distribution matches the feature
    graph (controls for long-range-ness; content is destroyed)."""
    n = coords.shape[0]
    target = edge_phys_dist(coords, ft_src, ft_dst)
    # precompute pairwise distance per source on the fly (sample)
    src_out, dst_out = [], []
    # bin target distances
    bins = np.quantile(target, np.linspace(0, 1, n_bins + 1))
    for i in range(n):
        di = np.sqrt(((coords - coords[i]) ** 2).sum(1))
        di[i] = np.inf
        # how many edges this source had
        m = (ft_src == i).sum()
        if m == 0:
            continue
        tdist = target[ft_src == i]
        for td in tdist:
            b = np.searchsorted(bins, td) - 1
            b = max(0, min(b, n_bins - 1))
            lo, hi = bins[b], bins[b + 1]
            cand = np.where((di >= lo) & (di <= hi))[0]
            if len(cand) == 0:
                cand = np.where(np.isfinite(di))[0]
            j = cand[rng.integers(0, len(cand))]
            src_out.append(i); dst_out.append(int(j))
    return np.array(src_out, np.int64), np.array(dst_out, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--out", default="results_rev/feature_graph.csv")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device)
    expr_all, labels_all, coords_all, sample_ids, _ = load_data()
    samples = sorted(set(sample_ids))
    n_genes = expr_all.shape[1]
    out = WORK / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for sample in samples:
            m = sample_ids == sample
            expr, coords, labels = expr_all[m], coords_all[m], labels_all[m]
            n = expr.shape[0]
            set_seed(seed)
            sp = build_knn_edges(coords, K_SPATIAL)
            ft = build_feature_edges(expr, K_FEATURE)
            rng = np.random.default_rng(seed)
            variants = {
                "full":         ft,
                "prune_lr":     variant_prune_lr(coords, ft[0], ft[1]),
                "local_feat":   variant_local_feat(expr, coords, K_FEATURE, 50, rng),
                "rand_feat":    variant_rand(n, len(ft[0]), rng),
                "dist_matched": variant_dist_matched(coords, ft[0], ft[1], rng),
            }
            for vname, ftv in variants.items():
                t0 = time.time()
                set_seed(seed)
                model = MODELS["SpatialDomainAE"](n_genes)
                emb = train_ae(model, expr, sp, ftv, device, n_epochs=args.epochs)
                ari, nmi, ncl, res, _ = cluster_and_eval(emb, labels, seed=seed)
                rows.append(dict(seed=seed, sample=sample, variant=vname,
                                 n_feat_edges=len(ftv[0]), ari=ari, nmi=nmi,
                                 n_clusters=ncl, res=res, sec=round(time.time()-t0,1)))
                print(f"  seed={seed} {sample:5s} {vname:13s} edges={len(ftv[0]):6d} "
                      f"ARI={ari:.3f} NMI={nmi:.3f} ({time.time()-t0:.0f}s)", flush=True)
                del model; torch.cuda.empty_cache()
            # spatial-only reference
            t0 = time.time(); set_seed(seed)
            model = MODELS["SpatialGATAE"](n_genes)
            emb = train_ae(model, expr, sp, ft, device, n_epochs=args.epochs)
            ari, nmi, ncl, res, _ = cluster_and_eval(emb, labels, seed=seed)
            rows.append(dict(seed=seed, sample=sample, variant="spatial_only",
                             n_feat_edges=0, ari=ari, nmi=nmi, n_clusters=ncl,
                             res=res, sec=round(time.time()-t0,1)))
            print(f"  seed={seed} {sample:5s} {'spatial_only':13s} edges={0:6d} "
                  f"ARI={ari:.3f} NMI={nmi:.3f}", flush=True)
            del model; torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
