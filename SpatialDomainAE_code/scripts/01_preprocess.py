#!/usr/bin/env python3
"""
01_preprocess.py — Preprocess Visium spatial data + H&E images (pure Python).

Uses scanpy + squidpy for standard spatial transcriptomics processing.
Reads GEO 10x matrix files + Seurat-exported metadata (annotations + coords).

Steps:
1. Load expression from GEO matrix.mtx.gz per sample
2. Merge samples, attach annotations from metadata CSV
3. Attach image coordinates from Seurat export
4. Crop H&E patches per spot (224x224)
5. Build spatial neighbor graph (squidpy)
6. Normalize expression (scanpy)
7. Save everything to HDF5 for training

Output:
    data/processed/spatial_dataset.h5
    data/processed/adata_combined.h5ad
"""

import gzip
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy.io import mmread
from scipy.sparse import csr_matrix
from pathlib import Path
from PIL import Image
from collections import OrderedDict
import h5py
import warnings

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

PROJ = Path(__file__).resolve().parent.parent
RAW = PROJ / "data" / "raw" / "geo_spatial"
PROC = PROJ / "data" / "processed"

SAMPLES = OrderedDict([
    ("Ctrl", {
        "barcode": "GSM7437221_C1-control_barcodes.tsv.gz",
        "feature": "GSM7437221_C1-control_features.tsv.gz",
        "matrix": "GSM7437221_C1-control_matrix.mtx.gz",
        "tif": "GSM7437221_C1_mouse_control_full_c1+2+3.tif",
        "coords": "coords_Ctrl.csv",
        "orig_ident": "Spatial2_sham",
    }),
    ("1DPI", {
        "barcode": "GSM7437222_B1-D1_barcodes.tsv.gz",
        "feature": "GSM7437222_B1-D1_features.tsv.gz",
        "matrix": "GSM7437222_B1-D1_matrix.mtx.gz",
        "tif": "GSM7437222_Mouse_1day_lesion_c1+2+3.tif",
        "coords": "coords_1DPI.csv",
        "orig_ident": "Spatial_D1",
    }),
    ("3DPI", {
        "barcode": "GSM7437223_D1-D3_barcodes.tsv.gz",
        "feature": "GSM7437223_D1-D3_features.tsv.gz",
        "matrix": "GSM7437223_D1-D3_matrix.mtx.gz",
        "tif": "GSM7437223_Mouse_3day_lesion_c1+2+3.tif",
        "coords": "coords_3DPI.csv",
        "orig_ident": "Spatial_D3",
    }),
    ("7DPI", {
        "barcode": "GSM7437224_C1-D7_barcodes.tsv.gz",
        "feature": "GSM7437224_C1-D7_features.tsv.gz",
        "matrix": "GSM7437224_C1-D7_matrix.mtx.gz",
        "tif": "GSM7437224_Mouse_7day_lesion_c1+2+3.tif",
        "coords": "coords_7DPI.csv",
        "orig_ident": "Spatial_D7",
    }),
])

PATCH_SIZE = 224


def load_10x_mtx(barcode_f, feature_f, matrix_f):
    """Load 10x-style sparse matrix from GEO files."""
    with gzip.open(barcode_f, "rt") as f:
        barcodes = [line.strip() for line in f]
    with gzip.open(feature_f, "rt") as f:
        features = [line.strip().split("\t") for line in f]
    gene_ids = [feat[0] for feat in features]
    gene_names = [feat[1] for feat in features]

    mat = mmread(str(matrix_f)).T  # transpose: spots x genes
    mat = csr_matrix(mat)

    adata = sc.AnnData(X=mat)
    adata.obs_names = barcodes
    adata.var_names = gene_names
    adata.var["gene_ids"] = gene_ids
    return adata


def decompress_tif(tif_path):
    """Decompress .tif.gz if needed."""
    gz_path = Path(str(tif_path) + ".gz")
    if not tif_path.exists() and gz_path.exists():
        import subprocess
        print(f"    Decompressing {gz_path.name}...")
        subprocess.run(["gunzip", "-k", str(gz_path)], check=True)
    return tif_path


def crop_patches(tif_path, imagerow, imagecol, patch_size=224):
    """Crop H&E patches centered on each spot."""
    img = Image.open(str(tif_path))
    img_w, img_h = img.size
    half = patch_size // 2
    n = len(imagerow)

    patches = np.zeros((n, patch_size, patch_size, 3), dtype=np.uint8)
    for i in range(n):
        cx, cy = int(imagecol[i]), int(imagerow[i])
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(img_w, cx + half), min(img_h, cy + half)
        patch = np.array(img.crop((x1, y1, x2, y2)))
        ph, pw = patch.shape[:2]
        if ph < patch_size or pw < patch_size:
            padded = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 255
            padded[:ph, :pw] = patch
            patch = padded
        patches[i] = patch[:patch_size, :patch_size]

    img.close()
    return patches


def main():
    print("=" * 60)
    print("Preprocessing: Visium + H&E (scanpy/squidpy)")
    print("=" * 60)

    # Load metadata with annotations
    meta = pd.read_csv(PROC / "spot_metadata.csv", index_col=0)
    print(f"Metadata: {meta.shape[0]} spots, {meta.shape[1]} cols")

    adata_list = []
    all_patches = []

    for sname, sinfo in SAMPLES.items():
        print(f"\n--- {sname} ---")

        # 1. Load expression
        adata = load_10x_mtx(
            RAW / sinfo["barcode"],
            RAW / sinfo["feature"],
            RAW / sinfo["matrix"],
        )
        print(f"  Expression: {adata.shape}")

        # 2. Load image coordinates
        coords = pd.read_csv(PROC / sinfo["coords"], index_col=0)

        # Match barcodes: GEO barcodes are base (no suffix), coords have suffix
        geo_bc_set = set(adata.obs_names)
        coord_base = {bc.split("_")[0]: bc for bc in coords.index}

        matched_geo = []
        matched_coord = []
        for base_bc, full_bc in coord_base.items():
            if base_bc in geo_bc_set:
                matched_geo.append(base_bc)
                matched_coord.append(full_bc)

        print(f"  Matched: {len(matched_geo)} spots")

        # Subset and rename
        adata = adata[matched_geo].copy()
        adata.obs_names = matched_coord  # use full barcodes to match metadata

        # Attach image coords
        coords_matched = coords.loc[matched_coord]
        adata.obs["imagerow"] = coords_matched["imagerow"].values
        adata.obs["imagecol"] = coords_matched["imagecol"].values
        adata.obs["array_row"] = coords_matched["row"].values
        adata.obs["array_col"] = coords_matched["col"].values

        # Attach spatial coords for scanpy
        adata.obsm["spatial"] = coords_matched[["imagecol", "imagerow"]].values.astype(float)

        # Attach metadata (labels, RCTD, etc.)
        meta_sub = meta.loc[meta.index.isin(matched_coord)]
        for col in ["DetailedRegionAnnoShort", "BrainAreas", "Condition",
                     "Timepoint", "IntegratedAnno"]:
            if col in meta_sub.columns:
                adata.obs[col] = meta_sub.loc[adata.obs_names, col].values

        # RCTD cell type proportions
        rctd_cols = [c for c in meta_sub.columns if c.startswith("RCTD_")]
        for col in rctd_cols:
            adata.obs[col] = meta_sub.loc[adata.obs_names, col].values

        adata.obs["sample"] = sname

        # 3. Crop H&E patches
        tif_path = decompress_tif(RAW / sinfo["tif"])
        print(f"  Cropping {len(matched_coord)} patches from {tif_path.name}...")
        patches = crop_patches(
            tif_path,
            coords_matched["imagerow"].values,
            coords_matched["imagecol"].values,
            patch_size=PATCH_SIZE,
        )
        print(f"  Patches: {patches.shape}")
        all_patches.append(patches)

        # Make var_names unique (some genes appear multiple times)
        adata.var_names_make_unique()
        adata_list.append(adata)

    # --- Merge all samples ---
    print("\n=== Merging samples ===")
    # Ensure var_names are unique in all adatas
    for ad in adata_list:
        ad.var_names_make_unique()
    adata = sc.concat(adata_list, merge="same")
    print(f"  Combined: {adata.shape}")

    # --- Normalize ---
    print("\n=== Normalizing expression ===")
    adata.layers["raw_counts"] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"  After filtering: {adata.shape}")

    # HVG selection for model input
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key="sample")
    print(f"  HVGs: {adata.var['highly_variable'].sum()}")

    # PCA
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50)

    # --- Build spatial graph ---
    print("\n=== Building spatial graph ===")
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
    adj = adata.obsp["spatial_connectivities"]
    src, dst = adj.nonzero()
    print(f"  Spatial edges: {len(src)}")

    # --- Save anndata ---
    print("\n=== Saving ===")
    adata.write(PROC / "adata_combined.h5ad")
    print(f"  Saved adata_combined.h5ad")

    # --- Save HDF5 for PyTorch training ---
    patches_all = np.concatenate(all_patches, axis=0)

    # Align patches with adata obs order
    # (concat preserves order, so patches should already be aligned)
    assert patches_all.shape[0] == adata.shape[0], \
        f"Patch count {patches_all.shape[0]} != spot count {adata.shape[0]}"

    # Expression matrix (HVG only, dense)
    hvg_mask = adata.var["highly_variable"].values
    expr_hvg = adata.X[:, hvg_mask]
    if hasattr(expr_hvg, "toarray"):
        expr_hvg = expr_hvg.toarray()
    expr_hvg = np.array(expr_hvg, dtype=np.float32)

    # Labels
    labels = adata.obs["DetailedRegionAnnoShort"].values
    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    label_indices = np.array([label_to_idx[l] for l in labels], dtype=np.int64)

    brain_areas = adata.obs["BrainAreas"].values
    unique_brain = sorted(set(brain_areas))
    brain_to_idx = {l: i for i, l in enumerate(unique_brain)}
    brain_indices = np.array([brain_to_idx[l] for l in brain_areas], dtype=np.int64)

    out_path = PROC / "spatial_dataset.h5"
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("expression", data=expr_hvg, compression="gzip")
        hf.create_dataset("patches", data=patches_all, compression="gzip",
                          chunks=(1, PATCH_SIZE, PATCH_SIZE, 3))
        hf.create_dataset("graph_src", data=src.astype(np.int64))
        hf.create_dataset("graph_dst", data=dst.astype(np.int64))
        hf.create_dataset("label_indices", data=label_indices)
        hf.create_dataset("brain_area_indices", data=brain_indices)

        dt = h5py.string_dtype()
        hf.create_dataset("sample_ids",
                          data=np.array(adata.obs["sample"].values, dtype="S"),
                          dtype=dt)
        hf.create_dataset("label_names",
                          data=np.array(unique_labels, dtype="S"), dtype=dt)
        hf.create_dataset("brain_area_names",
                          data=np.array(unique_brain, dtype="S"), dtype=dt)
        hf.create_dataset("gene_names",
                          data=np.array(adata.var_names[hvg_mask], dtype="S"),
                          dtype=dt)

    print(f"  Saved spatial_dataset.h5 ({out_path.stat().st_size / 1e9:.2f} GB)")

    # Print label distribution
    print("\n=== Label distribution ===")
    for l in unique_labels:
        n = (labels == l).sum()
        print(f"  {l:12s}: {n:5d}")

    # Save example patches
    patch_dir = PROC / "he_patches_examples"
    patch_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(patches_all), size=min(30, len(patches_all)),
                         replace=False)
    for i in indices:
        Image.fromarray(patches_all[i]).save(
            patch_dir / f"patch_{i}_{labels[i]}_{adata.obs['sample'].iloc[i]}.png"
        )
    print(f"  Saved {len(indices)} example patches")

    print("\n=== Preprocessing complete ===")


if __name__ == "__main__":
    main()
