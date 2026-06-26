#!/bin/bash
# runcap.sh <nthreads> -- <python args...>
# Caps BLAS/OpenMP thread counts, then runs python with the given arguments.
# Activate your SpatialDomainAE environment first (see build_env.sh).
NTH="$1"; shift
export OMP_NUM_THREADS="$NTH" OPENBLAS_NUM_THREADS="$NTH" MKL_NUM_THREADS="$NTH" NUMEXPR_NUM_THREADS="$NTH"
exec python "$@"
