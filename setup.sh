#!/bin/bash
# Setup script for cached-summaries-ehr-inference
# Creates a virtual environment and installs all dependencies.
#
# Usage:
#   bash setup.sh
#
# Requirements: Python 3.11+

set -euo pipefail

VENV_DIR="${1:-venv}"
PYTHON="${PYTHON:-python3.11}"

echo "==> Checking Python version..."
$PYTHON --version || { echo "ERROR: $PYTHON not found. Install Python 3.11+ first."; exit 1; }

echo "==> Creating virtual environment in ./$VENV_DIR ..."
$PYTHON -m venv "$VENV_DIR"

echo "==> Activating virtual environment..."
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Done! Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then install meds-torch (required framework):"
echo "  pip install meds-torch"
echo ""
echo "To run a smoke test of the hybrid summary encoder:"
echo "  python -c \"from src.meds_torch.input_encoder.hybrid_summary_encoder import HybridSummaryEncoder; print('OK')\""
