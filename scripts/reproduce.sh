#!/usr/bin/env bash
# Rebuild everything from scratch in a brand-new virtual environment and
# check the results match the committed metrics.
#
# Usage (from the project root):
#     bash scripts/reproduce.sh
set -euo pipefail

VENV_DIR=".venv-repro"
PYTHON_BIN="${PYTHON_BIN:-python3}"
step() { printf '\n=== %s ===\n' "$1"; }

step "Python version"
"$PYTHON_BIN" --version

step "Creating a clean virtual environment ($VENV_DIR)"
rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
PY="$VENV_DIR/bin/python"

step "Installing pinned dependencies"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt

step "Saving current metrics for comparison"
HAVE_BASELINE=0
if [ -f reports/metrics.json ]; then
  cp reports/metrics.json reports/metrics_before.json
  HAVE_BASELINE=1
fi

step "Downloading and cleaning data"
"$PY" -m src.data --force

step "Training (no registry update)"
"$PY" -m src.train --no-register

step "Predicting on sample patients"
"$PY" -m src.predict --input samples/patients.json

if [ "$HAVE_BASELINE" -eq 1 ]; then
  step "Comparing with previous metrics"
  "$PY" scripts/compare_metrics.py reports/metrics_before.json reports/metrics.json
  rm reports/metrics_before.json
fi

step "Done. Remove the test environment with: rm -rf $VENV_DIR"
