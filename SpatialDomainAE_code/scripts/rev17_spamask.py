#!/usr/bin/env python3
"""rev17_spamask.py — SpaMask dual-masking baseline (2nd dual-view comparator).

SpaMask (Min et al., PLOS Comput Biol 2025): masked graph autoencoder + masked
graph contrastive learning. Run on the same normalized 3000-HVG input and same
spatial KNN as every other method; its embedding is clustered with our shared
oracle Leiden grid for a fair comparison.
"""
import os, argparse, time, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, scanpy as sc, h5py, scipy.sparse as sps
from pathlib import Path
from rev_common import cluster_and_eval, set_seed, K_SPATIAL, WORK

SPM = os.environ.get("SPAMASK_DIR", "SpaMask")  # local clone of the SpaMask repository
sys.path.insert(0, SPM)
from SpaMask.spaMask import SPAMASK

RAW = Path(os.environ.get("STROKE_H5AD", "data/processed/adata_combined.h5ad"))


def load_raw():
    with h5py.File(RAW) as f:
        rc = f["layers"]["raw_counts"]; shape = tuple(rc.attrs["shape"])
        X = sps.csr_matrix((rc["data"][:], rc["indices"][:], rc["indptr"][:]), shape=shape).toarray().astype(np.float32)
        def cat(n):
            g = f["obs"][n]; cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
            return np.array([cats[c] for c in g["codes"][:]])
        return X, cat("Condition"), cat("DetailedRegionAnnoShort"), f["obsm"]["spatial"][:]


def run_spamask(raw_counts, coords, labels, device, seed):
    """Fair run: give SpaMask its intended raw-count input and let it do its own
    HVG selection / normalization / scaling (top_genes=3000, hvg)."""
    n_clusters = len(set(labels.tolist()))
    adata = sc.AnnData(X=raw_counts.astype(np.float32))
    adata.obsm["spatial"] = coords.astype(np.float64)
    s = SPAMASK(adata, tissue_name="MCAO", num_clusters=n_clusters,
                top_genes=3000, genes_model="hvg",
                graph_model="KNN", k_cutoff=K_SPATIAL, device=device,
                random_seed=seed, max_epoch=1000)
    s.train()
    s.process(method="kmeans")
    ad = s._SPAMASK__adata
    emb = np.asarray(ad.obsm["eval_pred"], dtype=np.float32)
    native = np.asarray(ad.obs["kmeans"]).astype(int) if "kmeans" in ad.obs else None
    return emb, native


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results_rev/spamask_stroke.csv")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]; device = torch.device(args.device)
    # all aligned from adata_combined: raw counts + labels + coords (same spot order)
    X_raw, cond_raw, lab_raw, coords_raw = load_raw()
    samples = ["Ctrl", "1DPI", "3DPI", "7DPI"]
    out = WORK / args.out; out.parent.mkdir(parents=True, exist_ok=True); rows = []
    for seed in seeds:
        for s in samples:
            rm = cond_raw == s; raw = X_raw[rm]; coords = coords_raw[rm]
            uniq = sorted(set(lab_raw[rm])); l2i = {l: i for i, l in enumerate(uniq)}
            labels = np.array([l2i[x] for x in lab_raw[rm]])
            t0 = time.time(); set_seed(seed)
            try:
                from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                emb, native = run_spamask(raw, coords, labels, device, seed)
                ari, nmi, ncl, res, _ = cluster_and_eval(emb, labels, seed=seed)   # shared Leiden protocol
                nat_ari = adjusted_rand_score(labels, native) if native is not None else float("nan")
                nat_nmi = normalized_mutual_info_score(labels, native) if native is not None else float("nan")
                rows.append(dict(seed=seed, sample=s, method="SpaMask", ari=ari, nmi=nmi,
                                 native_kmeans_ari=nat_ari, native_kmeans_nmi=nat_nmi, n_clusters=ncl, res=res))
                print(f"  seed={seed} {s:5s} SpaMask leiden_ARI={ari:.3f} nativeKMeans_ARI={nat_ari:.3f} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"  seed={seed} {s:5s} SpaMask FAIL {repr(e)[:160]}", flush=True)
            torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
