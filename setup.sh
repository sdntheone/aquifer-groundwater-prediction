#!/usr/bin/env bash
# setup.sh
# Creates a local virtual environment, installs dependencies, and runs the
# training + prediction pipeline so the app has fresh model artifacts.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo ">> Creating virtual environment (.venv)..."
$PYTHON_BIN -m venv .venv

echo ">> Activating virtual environment..."
source .venv/bin/activate

echo ">> Upgrading pip..."
pip install --upgrade pip

echo ">> Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ">> Training model..."
python train.py

echo ">> Scoring prediction points + building map GeoJSON..."
python predict.py

echo ""
echo "Setup complete. To run the app locally:"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo "Then open http://localhost:5000"
