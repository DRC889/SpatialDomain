#!/usr/bin/env python3
"""rev16_alpha_anchor.py — Anchor the alpha_feature readout to a LABEL-INDEPENDENT
disorganization measure (addresses 'alpha is only validated on the model's own labels').

For each sample we compute, per spot:
  - alpha_feature  (from the trained SpatialDomainAE fusion module)
  - local label entropy: Shannon entropy of ground-truth domain labels among the
    spot's spatial k-NN (high = locally heterogeneous tissue = disorganized)
and report the Spearman correlation alpha vs entropy (per sample + pooled).
This provides an external anchor that does NOT reuse the domain identity of the
spot itself, only the local heterogeneity of annotations around it.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, sys
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from rev_common import (load_data, build_knn_edges, build_feature_edges, train_ae,
                        set_seed, K_SPATIAL, K_FEATURE, WORK)
sys.path.insert(0, str(WORK / "src"))
from model import SpatialDomainAE
SEED = 42


def local_entropy(coords, labels, k=15):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    ent = np.zeros(len(labels))
    for i in range(len(labels)):
        nb = labels[idx[i, 1:]]
        _, c = np.unique(nb, return_counts=True)
        p = c / c.sum()
        ent[i] = -(p * np.log(p)).sum()
    return ent


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    expr_all, labels_all, coords_all, sample_ids, _ = load_data()
    rows = []; pooled_a, pooled_e = [], []
    for s in ["Ctrl", "1DPI", "3DPI", "7DPI"]:
        m = sample_ids == s
        expr, coords, labels = expr_all[m], coords_all[m], labels_all[m]
        set_seed(SEED)
        sp = build_knn_edges(coords, K_SPATIAL); ft = build_feature_edges(expr, K_FEATURE)
        set_seed(SEED)
        model = SpatialDomainAE(expr.shape[1], 64, 256, 4, 0.3)
        train_ae(model, expr, sp, ft, device, n_epochs=500)
        xt = torch.tensor(expr, dtype=torch.float32, device=device)
        et = torch.tensor(np.stack(sp), dtype=torch.long, device=device)
        eft = torch.tensor(np.stack(ft), dtype=torch.long, device=device)
        model.eval()
        with torch.no_grad():
            _, alpha = model.encode(xt, et, eft)
        af = alpha[:, 1].cpu().numpy()
        ent = local_entropy(coords, labels, K_SPATIAL)
        rho, p = spearmanr(af, ent)
        rows.append(dict(sample=s, n=len(labels), spearman_rho=round(rho, 3), p=p))
        pooled_a.append(af); pooled_e.append(ent)
        print(f"  {s}: Spearman(alpha_feature, local_label_entropy) rho={rho:+.3f} p={p:.2e} (n={len(labels)})", flush=True)
        del model; torch.cuda.empty_cache()
    a = np.concatenate(pooled_a); e = np.concatenate(pooled_e)
    rho, p = spearmanr(a, e)
    print(f"\n  POOLED: rho={rho:+.3f} p={p:.2e} (n={len(a)})", flush=True)
    rows.append(dict(sample="POOLED", n=len(a), spearman_rho=round(rho, 3), p=p))
    pd.DataFrame(rows).to_csv(WORK / "results_rev/alpha_anchor.csv", index=False)
    print("saved results_rev/alpha_anchor.csv")


if __name__ == "__main__":
    main()
