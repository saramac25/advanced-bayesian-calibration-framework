#!/bin/bash
# Run the whole workflow on one example folder: bash run_all.sh toy_scalar
set -e
cd "$(dirname "$0")/$1"
W=../../workflows
for s in 01_sensitivity_morris 02_correlation_analysis 03_train_surrogates 04_run_standard 05_run_hierarchical 06_diagnostics 07_postprocessing; do
  echo "=================== $s"
  python $W/$s.py config.yaml $( [[ $s == 04* || $s == 05* ]] && echo --no-progress )
done
