#!/bin/bash
# build_env.sh -- create a clean, self-contained conda/mamba environment for SpatialDomainAE.
# Requirements: conda or mamba on PATH; a CUDA 12.x GPU for the external baselines
# (our model itself is pure PyTorch and also runs on CPU). Creates ./envs/spatialdomainae.
set -e

ENV="./envs/spatialdomainae"
CREATE="$(command -v mamba || command -v conda)"
if [ -z "$CREATE" ]; then echo "ERROR: conda or mamba is required on PATH"; exit 1; fi

echo "===== [1/5] create env (python 3.10) ====="
"$CREATE" create -y -p "$ENV" python=3.10
PIP="$ENV/bin/pip"
"$PIP" install --no-cache-dir --upgrade pip

echo "===== [2/5] scientific + single-cell stack ====="
"$PIP" install --no-cache-dir \
  "numpy==1.26.4" "scipy==1.13.1" "pandas==2.2.2" h5py \
  "scikit-learn==1.5.2" "scanpy==1.10.3" "numba==0.60.0" \
  leidenalg python-igraph python-louvain matplotlib "gseapy==1.1.3" requests

echo "===== [3/5] PyTorch 2.5.1 (CUDA 12.4 wheels; use the cpu index-url on CPU-only hosts) ====="
"$PIP" install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo "===== [4/5] PyG + compiled ops (only needed for the external baselines) ====="
"$PIP" install --no-cache-dir torch_geometric==2.6.1
"$PIP" install --no-cache-dir torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

echo "===== [5/5] spatial-domain baselines (optional; only for the Table 1 comparisons) ====="
"$PIP" install --no-cache-dir SpaGCN POT || echo "WARN: SpaGCN/POT install failed"
"$PIP" install --no-cache-dir "git+https://github.com/QIFEIDKN/STAGATE_pyG.git" || echo "WARN: STAGATE_pyG"
"$PIP" install --no-cache-dir "git+https://github.com/JinmiaoChenLab/GraphST.git" || echo "WARN: GraphST"
# Spatial-MGCN and SpaMask have no PyPI package: clone their repositories and set
# SPATIALMGCN_DIR / SPAMASK_DIR to the clones when running rev07/rev17 (see those scripts).

echo "===== verify ====="
"$ENV/bin/python" - <<'PYV'
import importlib
for m in ["numpy","scipy","pandas","sklearn","scanpy","numba","leidenalg","igraph",
          "matplotlib","gseapy","torch","torch_geometric"]:
    try:
        x = importlib.import_module(m); print(f"OK   {m:16s} {getattr(x,'__version__','')}")
    except Exception as e:
        print(f"FAIL {m:16s} {repr(e)[:80]}")
try:
    import torch; print("CUDA available:", torch.cuda.is_available(), "| torch", torch.__version__)
except Exception:
    pass
PYV
echo "Done. Activate with:  conda activate $ENV   (or: source activate $ENV)"
# Note: if Pillow/matplotlib raise 'GLIBCXX_3.4.29 not found', prepend the env lib:
#   export LD_LIBRARY_PATH="$ENV/lib:$LD_LIBRARY_PATH"
