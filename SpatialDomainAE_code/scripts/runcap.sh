#!/bin/bash
# runcap.sh <nthreads> -- <python args...>
# Sets the working env (lin_MultiVI + rev_pkgs + libstdc++ fix + thread cap), then execs python.
NTH=$1; shift
export OMP_NUM_THREADS=$NTH OPENBLAS_NUM_THREADS=$NTH MKL_NUM_THREADS=$NTH NUMEXPR_NUM_THREADS=$NTH
export LD_LIBRARY_PATH=/home/project/11003054/dmeng/softs/miniconda3/envs/lin_MultiVI/lib
export PYTHONPATH=/scratch/users/nus/e1503317/rev_pkgs
exec /home/project/11003054/dmeng/softs/miniconda3/envs/lin_MultiVI/bin/python "$@"
