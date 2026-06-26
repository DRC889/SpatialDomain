#!/usr/bin/env python3
"""rev07_spatialmgcn.py — Spatial-MGCN dual-view baseline.

Runs Spatial-MGCN (Wang et al. 2023) on the stroke and/or DLPFC data using the
SAME normalized 3000-HVG input and the SAME oracle Leiden grid as every other
method, for an apples-to-apples comparison. Spatial-MGCN's own multi-view GCN +
attention + ZINB objective is used to learn the embedding (model code from the
authors' repo); we cluster the resulting embedding with our common protocol.

Run:
  # (run inside an environment with Spatial-MGCN dependencies installed)
  python rev07_spatialmgcn.py --dataset stroke --seeds 0,1,2 --device cuda:0 \
      --out results_rev/spatialmgcn_stroke.csv
"""
import os, argparse, time, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.optim as optim
from sklearn.neighbors import kneighbors_graph
from rev_common import (load_data, cluster_and_eval, set_seed, K_SPATIAL, K_FEATURE, WORK)

SMGCN = os.environ.get("SPATIALMGCN_DIR", "Spatial-MGCN")  # local clone of the Spatial-MGCN repository
sys.path.insert(0, SMGCN)
from models import Spatial_MGCN
from utils import (regularization_loss, consistency_loss, ZINB,
                   normalize_sparse_matrix, sparse_mx_to_torch_sparse_tensor)


def build_adj(mat, k, metric):
    A = kneighbors_graph(mat, k + 1, mode="connectivity", metric=metric, include_self=True).toarray()
    r, c = np.diag_indices_from(A); A[r, c] = 0
    adj = sp.coo_matrix(A, dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj, A


def to_norm_tensor(adj, device):
    n = normalize_sparse_matrix(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(n).to(device)


def run_smgcn(expr, coords, device, seed, epochs=200,
              nhid1=128, nhid2=64, lr=1e-3, wd=5e-4,
              alpha=1.0, beta=10.0, gamma=0.1):
    set_seed(seed)
    # graphs: feature = cosine KNN (k_f), spatial = euclidean KNN (k_s)
    fadj_sp, _ = build_adj(expr, K_FEATURE, "cosine")
    sadj_sp, A_sp = build_adj(coords, K_SPATIAL, "euclidean")
    features = torch.FloatTensor(expr).to(device)
    fadj = to_norm_tensor(fadj_sp, device)
    sadj = to_norm_tensor(sadj_sp, device)
    graph_nei = torch.LongTensor(A_sp).to(device)
    graph_neg = (torch.ones_like(graph_nei) - graph_nei).to(device)
    model = Spatial_MGCN(nfeat=expr.shape[1], nhid1=nhid1, nhid2=nhid2, dropout=0.0).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_emb, best_loss = None, float("inf")
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        com1, com2, emb, pi, disp, mean = model(features, sadj, fadj)
        zinb = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
        reg = regularization_loss(emb, graph_nei, graph_neg)
        con = consistency_loss(com1, com2)
        loss = alpha * zinb + beta * con + gamma * reg
        loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_emb = pd.DataFrame(emb.detach().cpu().numpy()).fillna(0).values
    return best_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="stroke", choices=["stroke", "dlpfc"])
    ap.add_argument("--samples", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="results_rev/spatialmgcn.csv")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device(args.device)
    out = WORK / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.dataset == "stroke":
        expr_all, labels_all, coords_all, sample_ids, _ = load_data()
        samples = sorted(set(sample_ids))
        loader = lambda s: (expr_all[sample_ids == s], coords_all[sample_ids == s], labels_all[sample_ids == s])
    else:
        import rev05_dlpfc as d
        samples = (args.samples.split(",") if args.samples else d.ALL_SAMPLES)
        def loader(s):
            e, c, l, _ = d.load_dlpfc_sample(s); return e, c, l

    for seed in seeds:
        for s in samples:
            expr, coords, labels = loader(s)
            t0 = time.time()
            try:
                emb = run_smgcn(expr, coords, device, seed, epochs=args.epochs)
                ari, nmi, ncl, res, _ = cluster_and_eval(emb, labels, seed=seed)
                rows.append(dict(seed=seed, sample=str(s), method="Spatial-MGCN",
                                 ari=ari, nmi=nmi, n_clusters=ncl, res=res,
                                 sec=round(time.time()-t0, 1)))
                print(f"  seed={seed} {str(s):6s} Spatial-MGCN ARI={ari:.3f} NMI={nmi:.3f} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"  seed={seed} {str(s):6s} Spatial-MGCN FAIL {repr(e)[:140]}", flush=True)
            torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
