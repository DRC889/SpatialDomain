#!/bin/bash
# Build a clean, self-contained conda env for SpatialDomainAE (repo-ready).
set -e
MAMBA=/home/project/11003054/dmeng/softs/miniconda3/bin/mamba
ENV=/scratch/users/nus/e1503317/envs/spatialdomainae
SRC=/scratch/users/nus/e1503317/rev_src

echo "===== [1/6] create env (python 3.10) ====="
$MAMBA create -y -p "$ENV" python=3.10
PY="$ENV/bin/python"
PIP="$ENV/bin/pip"
$PIP install --no-cache-dir --upgrade pip

echo "===== [2/6] scientific + single-cell stack ====="
$PIP install --no-cache-dir \
  "numpy==1.26.4" "scipy==1.13.1" "pandas==2.2.2" h5py \
  "scikit-learn==1.5.2" "scanpy==1.10.3" "numba==0.60.0" \
  leidenalg python-igraph python-louvain matplotlib "gseapy==1.1.3" requests

echo "===== [3/6] torch 2.5.1 + cu124 ====="
$PIP install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo "===== [4/6] PyG + compiled ops ====="
$PIP install --no-cache-dir torch_geometric==2.6.1
$PIP install --no-cache-dir torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

echo "===== [5/6] spatial-domain baselines ====="
$PIP install --no-cache-dir SpaGCN || echo "WARN SpaGCN"
# STAGATE_pyG and GraphST are GitHub-only (no PyPI); install from extracted source
for repo_pkg in "QIFEIDKN/STAGATE_pyG:STAGATE_pyG" "JinmiaoChenLab/GraphST:GraphST"; do
  repo="${repo_pkg%%:*}"; pkg="${repo_pkg##*:}"
  root=$(ls -d "$SRC/$pkg-extract"/*/ 2>/dev/null | head -1)
  if [ -n "$root" ] && [ -d "$root/$pkg" ]; then
    cp -r "$root/$pkg" "$ENV/lib/python3.10/site-packages/$pkg"
    echo "installed $pkg from cached source"
  else
    echo "WARN: cached source for $pkg not found at $SRC"
  fi
done

echo "===== [6/6] verify ====="
"$PY" - <<'PYV'
mods=["numpy","scipy","pandas","sklearn","scanpy","numba","leidenalg","igraph",
      "matplotlib","gseapy","torch","torch_geometric","torch_scatter","torch_sparse",
      "STAGATE_pyG","GraphST","SpaGCN"]
import importlib
for m in mods:
    try:
        x=importlib.import_module(m); print(f"OK   {m:18s} {getattr(x,'__version__','')}")
    except Exception as e:
        print(f"FAIL {m:18s} {repr(e)[:90]}")
import torch; print("CUDA available:", torch.cuda.is_available(), "| torch", torch.__version__)
PYV
echo "===== DONE -> $ENV ====="
