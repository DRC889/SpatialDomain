# SpatialDomainAE

**Dual-Graph Attention Autoencoder for Spatial Domain Identification in Ischemic Stroke**

SpatialDomainAE is an unsupervised model for spatial domain identification in spatial transcriptomics. It builds two per-spot *k*-nearest-neighbor graphs — a spatial-proximity graph and an expression-correlation (feature) graph — processes each with a single-layer multi-head graph attention network (GAT), and fuses the two views with a learned **per-spot attention weight** `(α_spatial, α_feature)`. The fused embedding is decoded to reconstruct expression (MSE) and clustered with Leiden. The per-spot fusion weights double as an interpretable, region-level readout that separates lesion-associated domains.

The method is targeted at **pathological tissue with disrupted architecture**: on a mouse MCAO ischemic-stroke 10x Visium time course it significantly outperforms STAGATE, GraphST, SpaGCN, and the dual-view Spatial-MGCN at the most disrupted time point (3 DPI; all paired Wilcoxon p ≤ 0.037 over 10 seeds) and is competitive on average; on intact laminar cortex (DLPFC) it is comparable to most spatial baselines but below GraphST.

**Code repository:** https://github.com/DRC889/SpatialDomain

---

## Repository layout

```
src/
  model.py            # SpatialDomainAE + ablation variants (pure-PyTorch GAT, no torch_geometric)
  figure_style.py     # shared matplotlib style / palettes
  train.py            # training loop
scripts/
  01_preprocess.py            # raw -> normalized 3000-HVG matrix + graphs (Scanpy)
  07_unsupervised_benchmark.py# per-sample benchmark (ours + ablations + STAGATE/GraphST)
  11_run_spagcn.py            # SpaGCN baseline
  rev_common.py               # shared utilities for the revision experiments
  rev01_multiseed.py          # 10-seed robustness (ours + ablations)        [R1.5]
  rev02_feature_graph.py      # feature-graph necessity controls             [R1.3]
  rev03_sensitivity.py        # k_spatial / k_feature / metric sensitivity   [R1.4,R2.4]
  rev04_baselines.py          # STAGATE/GraphST/SpaGCN across seeds (stroke + DLPFC)
  rev05_dlpfc.py              # external DLPFC evaluation                     [R1.1]
  rev06_enrichment.py         # GO/KEGG/Reactome enrichment (gseapy)         [R1.7,R2.5]
  rev07_spatialmgcn.py        # Spatial-MGCN dual-view baseline              [R1.2,R2.1]
  rev08_fig1.py rev13_fig3.py   # figures
  rev09_analyze.py            # aggregate all CSVs -> tables / stats
manuscript/                   # LaTeX source (main.tex), references.bib, response_letter.md
results/                      # output CSVs
figures/                      # output figures
```

## Environment

Tested on Linux, NVIDIA A100 (driver CUDA 12.8), Python 3.10, PyTorch 2.5.1 + CUDA 12.4.
Our model and all ablations are **pure PyTorch** (custom scatter-based GAT) and do not need
`torch_geometric`; only the external baselines (STAGATE/GraphST/SpaGCN/Spatial-MGCN) do.

Create the environment with conda/mamba:

```bash
mamba create -y -p ./envs/spatialdomainae python=3.10
ENV=./envs/spatialdomainae

# scientific + single-cell stack
$ENV/bin/pip install numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 h5py \
    scikit-learn==1.5.2 scanpy==1.10.3 numba==0.60.0 \
    leidenalg python-igraph python-louvain matplotlib gseapy==1.1.3 requests

# PyTorch (CUDA 12.4)
$ENV/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# PyG (only needed for external baselines)
$ENV/bin/pip install torch_geometric==2.6.1
$ENV/bin/pip install torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# baselines: SpaGCN from PyPI; STAGATE_pyG and GraphST from GitHub (GraphST needs POT)
$ENV/bin/pip install SpaGCN POT
$ENV/bin/pip install "git+https://github.com/QIFEIDKN/STAGATE_pyG.git"
$ENV/bin/pip install "git+https://github.com/JinmiaoChenLab/GraphST.git"
```

> **Note.** `numpy` must stay `<2.1` (numba/scanpy compatibility). If you hit a
> `GLIBCXX_3.4.29 not found` error from Pillow/matplotlib, prepend the env's lib to
> the loader path: `export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH`.

## Data

Mouse MCAO ischemic-stroke 10x Visium time course: **GEO accession GSE233815**
(Zucha et al., 2024). Run `scripts/01_preprocess.py` to produce the normalized
3000-HVG matrix, coordinates, and KNN graphs under `data/processed/`.

External benchmark: human DLPFC (Maynard et al., 2021), 12 sections with manual
cortical-layer annotations (count matrices on the spatialLIBD S3 bucket; coordinates
and `barcode_level_layer_map.tsv` from the `LieberInstitute/HumanPilot` repository).

## Reproducing the results

Results are reported as **mean ± SD over 10 random seeds for every method**, using a shared oracle Leiden resolution grid
`{0.3,0.5,0.8,1.0,1.5,2.0,3.0,5.0}`.

```bash
ENV=./envs/spatialdomainae
# main benchmark + ablations (ours, SpatialGATAE, ExprOnlyAE), 10 seeds
$ENV/bin/python scripts/rev01_multiseed.py --seeds 0,1,2,3,4,5,6,7,8,9 --device cuda:0
# external baselines across seeds (stroke)
$ENV/bin/python scripts/rev04_baselines.py --dataset stroke --seeds 0,1,2,3,4,5,6,7,8,9
# dual-view baseline Spatial-MGCN
$ENV/bin/python scripts/rev07_spatialmgcn.py --dataset stroke --seeds 0,1,2,3,4,5,6,7,8,9
# feature-graph necessity controls (prune/local/distance-matched/random/spatial-only)
$ENV/bin/python scripts/rev02_feature_graph.py --seeds 0,1,2,3,4,5,6,7,8,9
# graph-construction sensitivity
$ENV/bin/python scripts/rev03_sensitivity.py --axis ks
$ENV/bin/python scripts/rev03_sensitivity.py --axis kf
$ENV/bin/python scripts/rev03_sensitivity.py --axis metric
# external DLPFC dataset (ours + ablations, and baselines)
$ENV/bin/python scripts/rev05_dlpfc.py --seeds 0,1,2
$ENV/bin/python scripts/rev04_baselines.py --dataset dlpfc --seeds 0,1,2
# pathway enrichment + figures
$ENV/bin/python scripts/rev06_enrichment.py
$ENV/bin/python scripts/rev08_fig1.py
$ENV/bin/python scripts/rev13_fig3.py
# aggregate everything into tables / statistics
$ENV/bin/python scripts/rev09_analyze.py
```

### One-step table reproduction

`python scripts/rev09_analyze.py` recomputes **every number in Table 1, Table 2, and
Supplementary Tables S1–S2** (benchmark, feature-graph controls, deployment, ablation),
including the paired Wilcoxon tests and Benjamini–Hochberg correction, directly from the
included `results/*.csv` — no GPU or model re-training required. The benchmark uses the
same 10 seeds for every method; the internal ablation additionally uses 2 repeats/seed.

### Note on reproducibility and run-to-run variance

Domain-identification ARI on the 3 DPI lesion has non-trivial run-to-run variance
(≈ ±0.03). A single run can therefore over- or under-state performance; we report the
10-seed mean (3 DPI ARI 0.700 ± 0.025 for SpatialDomainAE) with paired Wilcoxon tests
rather than any single run, and recommend the same protocol for re-evaluation.

## Citation

Chen Y-Y, Hu W-T, Zhang G, Rao X-L, Jiang T. *Dual-Graph Attention Autoencoder for
Spatial Domain Identification in Ischemic Stroke.* (under revision).
