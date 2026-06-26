#!/usr/bin/env python3
"""
train.py — Benchmark: SpatialDomainNet + ablations + STAGATE baseline.

Main model — SpatialDomainNet (Dual-graph GAT + Attention Fusion):
  Spatial k-NN graph + Feature (correlation) k-NN graph
  Single-layer GAT per graph → Attention fusion → Classifier

Ablations:
  SpatialGAT   — spatial graph only
  FeatureGAT   — feature graph only
  DualGCN      — dual-graph GCN (not GAT) + attention fusion
  ExprOnly     — expression MLP only
  RF           — RandomForest baseline

External:
  STAGATE      — established spatial domain method

Graph: k=15 spatial, k=20 feature (correlation metric)
Evaluation: Leave-one-sample-out CV (4 folds)
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import h5py
import warnings
import scanpy as sc

warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
from model import (SpatialDomainNet, ExprOnlyNet, SpatialGATNet,
                    FeatureGATNet, DualGCNNet)

RESULT_DIR = PROJ / "results"
RESULT_DIR.mkdir(exist_ok=True)

K_SPATIAL = 15
K_FEATURE = 20
SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================
# Data loading + graph construction
# =========================================================================

def load_dataset():
    """Load expression, labels, coordinates, PCA."""
    with h5py.File(PROJ / "data/processed/spatial_dataset.h5", "r") as hf:
        data = {
            "expression": hf["expression"][:],
            "label_indices": hf["label_indices"][:],
            "label_names": [s.decode() for s in hf["label_names"][:]],
            "gene_names": [s.decode() for s in hf["gene_names"][:]],
        }
    with h5py.File(PROJ / "data/processed/spatial_coords.h5", "r") as hf:
        data["coords"] = hf["spatial_coords"][:]
        data["sample_ids"] = np.array([s.decode() for s in hf["sample_ids"][:]])
    with h5py.File(PROJ / "data/processed/pca_features.h5", "r") as hf:
        data["pca"] = hf["pca"][:]
    return data


def build_knn_graph(coords, sample_ids, k):
    """Build per-sample k-NN spatial graph."""
    samples = np.unique(sample_ids)
    all_src, all_dst = [], []
    for s in samples:
        idx = np.where(sample_ids == s)[0]
        local_coords = coords[idx]
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(idx)),
                              metric="euclidean")
        nn.fit(local_coords)
        _, neighbors = nn.kneighbors(local_coords)
        for i in range(len(idx)):
            for j in range(1, neighbors.shape[1]):
                all_src.append(idx[i])
                all_dst.append(idx[neighbors[i, j]])
    return np.array(all_src, dtype=np.int64), np.array(all_dst, dtype=np.int64)


def build_feature_graph(expr, sample_ids, k=20):
    """Build per-sample feature-correlation k-NN graph."""
    samples = np.unique(sample_ids)
    all_src, all_dst = [], []
    for s in samples:
        idx = np.where(sample_ids == s)[0]
        local_expr = expr[idx]
        adj = kneighbors_graph(local_expr, k, mode="connectivity",
                               metric="correlation", include_self=False)
        rows, cols = adj.nonzero()
        all_src.extend(idx[rows].tolist())
        all_dst.extend(idx[cols].tolist())
    return np.array(all_src, dtype=np.int64), np.array(all_dst, dtype=np.int64)


def prepare_fold(data, sp_src, sp_dst, ft_src, ft_dst, test_sample):
    """Prepare train/test split for one LOSO fold."""
    sample_arr = data["sample_ids"]
    test_mask = sample_arr == test_sample
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    def subgraph(idx_set, src, dst):
        idx_map = {old: new for new, old in enumerate(idx_set)}
        mask = np.array([s in idx_map and d in idx_map
                         for s, d in zip(src, dst)])
        new_src = np.array([idx_map[s] for s, m in zip(src, mask) if m])
        new_dst = np.array([idx_map[d] for d, m in zip(dst, mask) if m])
        return new_src, new_dst

    tr_sp_src, tr_sp_dst = subgraph(set(train_idx), sp_src, sp_dst)
    te_sp_src, te_sp_dst = subgraph(set(test_idx), sp_src, sp_dst)
    tr_ft_src, tr_ft_dst = subgraph(set(train_idx), ft_src, ft_dst)
    te_ft_src, te_ft_dst = subgraph(set(test_idx), ft_src, ft_dst)

    return {
        "train_expr": data["expression"][train_idx],
        "train_pca": data["pca"][train_idx],
        "train_labels": data["label_indices"][train_idx],
        "train_edge_sp": np.stack([tr_sp_src, tr_sp_dst]),
        "train_edge_ft": np.stack([tr_ft_src, tr_ft_dst]),
        "test_expr": data["expression"][test_idx],
        "test_pca": data["pca"][test_idx],
        "test_labels": data["label_indices"][test_idx],
        "test_edge_sp": np.stack([te_sp_src, te_sp_dst]),
        "test_edge_ft": np.stack([te_ft_src, te_ft_dst]),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_coords": data["coords"][train_idx],
        "test_coords": data["coords"][test_idx],
    }


def _class_weights(labels, n_classes, device):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1
    w = 1.0 / counts
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32).to(device)


def _metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted",
                                zero_division=0),
        "ari": adjusted_rand_score(labels, preds),
        "nmi": normalized_mutual_info_score(labels, preds),
    }


# =========================================================================
# Training
# =========================================================================

def train_model(model, fold_data, device, n_epochs=100, lr=1e-3,
                patience=20, use_dual_graph=True):
    """Train any model with unified interface."""
    model.to(device)
    n_classes = int(max(fold_data["train_labels"].max(),
                        fold_data["test_labels"].max())) + 1

    expr = torch.tensor(fold_data["train_expr"],
                        dtype=torch.float32).to(device)
    labels = torch.tensor(fold_data["train_labels"],
                          dtype=torch.long).to(device)
    edge_sp = torch.tensor(fold_data["train_edge_sp"],
                           dtype=torch.long).to(device)
    edge_ft = torch.tensor(fold_data["train_edge_ft"],
                           dtype=torch.long).to(device)

    cw = _class_weights(fold_data["train_labels"], n_classes, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_loss = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        if use_dual_graph:
            logits = model(expr, edge_sp, edge_ft)
        else:
            logits = model(expr, edge_sp, edge_ft)  # model ignores unused

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience_ctr = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    del expr, labels
    torch.cuda.empty_cache()
    return model


def evaluate_model(model, fold_data, device, use_dual_graph=True):
    """Evaluate any model."""
    model.eval().to(device)
    expr = torch.tensor(fold_data["test_expr"],
                        dtype=torch.float32).to(device)
    edge_sp = torch.tensor(fold_data["test_edge_sp"],
                           dtype=torch.long).to(device)
    edge_ft = torch.tensor(fold_data["test_edge_ft"],
                           dtype=torch.long).to(device)
    labels = fold_data["test_labels"]

    with torch.no_grad():
        if use_dual_graph:
            logits = model(expr, edge_sp, edge_ft)
        else:
            logits = model(expr, edge_sp, edge_ft)
        preds = logits.argmax(-1).cpu().numpy()

    del expr
    torch.cuda.empty_cache()
    return _metrics(labels, preds), preds


def train_rf(fold_data):
    clf = RandomForestClassifier(
        n_estimators=500, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(fold_data["train_expr"], fold_data["train_labels"])
    preds = clf.predict(fold_data["test_expr"])
    return _metrics(fold_data["test_labels"], preds), preds


# =========================================================================
# STAGATE baseline
# =========================================================================

def run_stagate_fold(fold_data, data, n_classes, test_sample):
    """Run STAGATE on one fold using PCA features."""
    try:
        import STAGATE_pyG
    except ImportError:
        print("    STAGATE not installed, skipping")
        return None, None

    # Build adata for training spots
    train_pca = fold_data["train_pca"]
    train_coords = fold_data["train_coords"]
    train_labels = fold_data["train_labels"]

    adata_train = sc.AnnData(X=train_pca)
    adata_train.obsm["spatial"] = train_coords

    # Build adata for test spots
    test_pca = fold_data["test_pca"]
    test_coords = fold_data["test_coords"]
    test_labels = fold_data["test_labels"]

    adata_test = sc.AnnData(X=test_pca)
    adata_test.obsm["spatial"] = test_coords

    # STAGATE on full sample to get embeddings, then use for classification
    # Build combined adata for the test sample
    full_pca = np.vstack([train_pca, test_pca])
    full_coords = np.vstack([train_coords, test_coords])
    full_labels = np.concatenate([train_labels, test_labels])
    is_test = np.array([False]*len(train_labels) + [True]*len(test_labels))

    adata_full = sc.AnnData(X=full_pca)
    adata_full.obsm["spatial"] = full_coords

    # STAGATE spatial graph
    STAGATE_pyG.Cal_Spatial_Net(adata_full, rad_cutoff=None, k_cutoff=K_SPATIAL)

    # Train STAGATE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adata_full = STAGATE_pyG.train_STAGATE(
        adata_full, hidden_dims=[512, 30],
        n_epochs=500, lr=0.001, random_seed=42, device=device)

    # Get embeddings
    emb = adata_full.obsm["STAGATE"]  # (N, 30)

    # Train simple classifier on STAGATE embeddings
    from sklearn.neural_network import MLPClassifier
    train_emb = emb[~is_test]
    test_emb = emb[is_test]

    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                        random_state=42, early_stopping=True)
    clf.fit(train_emb, train_labels)
    preds = clf.predict(test_emb)

    return _metrics(test_labels, preds), preds


# =========================================================================
# Main
# =========================================================================

def main():
    set_seed()
    print("=" * 60)
    print("SpatialDomainNet Benchmark")
    print("  Dual-graph GAT + Attention Fusion")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data = load_dataset()
    n_genes = data["expression"].shape[1]
    n_classes = len(data["label_names"])
    samples = sorted(set(data["sample_ids"]))
    print(f"Samples: {samples}")
    print(f"Spots: {data['expression'].shape[0]}, Genes: {n_genes}, "
          f"Classes: {n_classes}")

    # Build graphs
    print(f"\nBuilding spatial graph (k={K_SPATIAL})...")
    sp_src, sp_dst = build_knn_graph(
        data["coords"], data["sample_ids"], k=K_SPATIAL)
    print(f"  Spatial edges: {len(sp_src)}")

    print(f"Building feature graph (k={K_FEATURE}, correlation)...")
    ft_src, ft_dst = build_feature_graph(
        data["expression"], data["sample_ids"], k=K_FEATURE)
    print(f"  Feature edges: {len(ft_src)}")

    # Model configs
    model_configs = [
        ("SpatialDomainNet",
         lambda: SpatialDomainNet(n_genes, n_classes, hidden_dim=256,
                                  n_heads=4, dropout=0.3)),
        ("DualGCN",
         lambda: DualGCNNet(n_genes, n_classes, hidden_dim=256, dropout=0.3)),
        ("SpatialGAT",
         lambda: SpatialGATNet(n_genes, n_classes, hidden_dim=256,
                               n_heads=4, dropout=0.3)),
        ("FeatureGAT",
         lambda: FeatureGATNet(n_genes, n_classes, hidden_dim=256,
                               n_heads=4, dropout=0.3)),
        ("ExprOnly",
         lambda: ExprOnlyNet(n_genes, n_classes, dropout=0.3)),
    ]

    all_results = {}
    all_preds = {}

    for test_sample in samples:
        print(f"\n{'='*55}")
        print(f"Fold: test = {test_sample}")
        print(f"{'='*55}")

        fold_data = prepare_fold(data, sp_src, sp_dst, ft_src, ft_dst,
                                 test_sample)
        print(f"  Train: {len(fold_data['train_labels'])}, "
              f"Test: {len(fold_data['test_labels'])}")

        # ── Our models ──────────────────────────────────────
        for mname, model_fn in model_configs:
            print(f"\n  --- {mname} ---")
            model = model_fn()
            model = train_model(model, fold_data, device)
            r, preds = evaluate_model(model, fold_data, device)
            all_results[f"{mname}_{test_sample}"] = {
                "model": mname, "test_sample": test_sample, **r}
            all_preds[f"{mname}_{test_sample}"] = preds
            print(f"    Acc={r['accuracy']:.3f}  F1w={r['f1_weighted']:.3f}  "
                  f"ARI={r['ari']:.3f}  NMI={r['nmi']:.3f}")
            del model; torch.cuda.empty_cache()

        # ── RandomForest ─────────────────────────────────────
        print(f"\n  --- RandomForest ---")
        r, preds = train_rf(fold_data)
        all_results[f"RF_{test_sample}"] = {
            "model": "RF", "test_sample": test_sample, **r}
        all_preds[f"RF_{test_sample}"] = preds
        print(f"    Acc={r['accuracy']:.3f}  F1w={r['f1_weighted']:.3f}  "
              f"ARI={r['ari']:.3f}  NMI={r['nmi']:.3f}")

        # ── STAGATE ──────────────────────────────────────────
        print(f"\n  --- STAGATE ---")
        r, preds = run_stagate_fold(fold_data, data, n_classes, test_sample)
        if r is not None:
            all_results[f"STAGATE_{test_sample}"] = {
                "model": "STAGATE", "test_sample": test_sample, **r}
            all_preds[f"STAGATE_{test_sample}"] = preds
            print(f"    Acc={r['accuracy']:.3f}  F1w={r['f1_weighted']:.3f}  "
                  f"ARI={r['ari']:.3f}  NMI={r['nmi']:.3f}")

    # =================================================================
    # Summary
    # =================================================================
    print(f"\n{'='*60}")
    print("Summary (mean ± std across 4 folds)")
    print(f"{'='*60}")

    results_df = pd.DataFrame(all_results).T
    model_order = [c[0] for c in model_configs] + ["RF", "STAGATE"]
    summary = []
    for mname in model_order:
        sub = results_df[results_df["model"] == mname]
        if len(sub) == 0:
            continue
        row = {"model": mname}
        for metric in ["accuracy", "f1_macro", "f1_weighted", "ari", "nmi"]:
            vals = sub[metric].astype(float)
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std()
        summary.append(row)
        print(f"  {mname:20s}: "
              f"Acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}  "
              f"F1w={row['f1_weighted_mean']:.3f}±{row['f1_weighted_std']:.3f}  "
              f"ARI={row['ari_mean']:.3f}±{row['ari_std']:.3f}  "
              f"NMI={row['nmi_mean']:.3f}±{row['nmi_std']:.3f}")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULT_DIR / "benchmark_summary.csv", index=False)
    results_df.to_csv(RESULT_DIR / "all_results.csv")

    json_results = {k: {kk: float(vv) if isinstance(vv, (np.floating, float))
                        else vv for kk, vv in v.items()}
                    for k, v in all_results.items()}
    with open(RESULT_DIR / "benchmark_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    np.savez(RESULT_DIR / "predictions.npz",
             **{k: v for k, v in all_preds.items()})

    print(f"\nResults saved to {RESULT_DIR}/")
    print("=== Training complete ===")


if __name__ == "__main__":
    main()
