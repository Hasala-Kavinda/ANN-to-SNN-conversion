#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  SNN-DOOM Setup Script
#  Creates a Python venv and installs all dependencies.
#  Works on both local dev machine and department training server.
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "==> Installing PyTorch (CUDA 12.4 wheels — compatible with CUDA 12.x and 13.x drivers)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing VizDoom and core packages..."
pip install "vizdoom>=1.2.4" opencv-python-headless

echo "==> Installing SNN and training utilities..."
pip install "snntorch>=0.9" tensorboard imageio imageio-ffmpeg tqdm

echo "==> Installing analysis and testing utilities..."
pip install matplotlib pytest

echo ""
echo "✓ Setup complete!"
echo "  Activate the environment with:  source venv/bin/activate"
echo "  Start training with:            python training/train_doom.py"
echo "  Run tests with:                 pytest tests/"
