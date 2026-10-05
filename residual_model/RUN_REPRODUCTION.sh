#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON=${RESEARCH_PYTHON:-/usr/bin/python3}
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$(pwd)/qa/matplotlib"
mkdir -p logs data qa/matplotlib
# Deterministic local workflow. No paid or remote computation. Typical total
# measured runtime is recorded in data/*.json; direct source quadrature is slowest.
for script in compact_airy analytic_series generate_high_order study complete_predictions direct_check direct_events high_precision decomposition save_raw_curves minimum_derivatives summarize plot_evidence; do
  "$PYTHON" "code/$script.py" > "logs/${script}_reproduction.log" 2>&1
done
