# ─────────────────────────────────────────────────────────────
#  SNN-DOOM Windows Setup Script
#  Creates a Python 3.13 virtual environment (venv_win) and
#  installs all dependencies for native Windows development.
# ─────────────────────────────────────────────────────────────
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

# Prefer CPython 3.13 (supports PyTorch CUDA & ViZDoom wheels)
if (Get-Command py -ErrorAction SilentlyContinue) {
    Write-Host "==> Creating Python 3.13 virtual environment (venv_win)..."
    py -3.13 -m venv venv_win
} else {
    Write-Host "==> Creating Python virtual environment (venv_win)..."
    python -m venv venv_win
}

$Py = Join-Path $PSScriptRoot "venv_win\Scripts\python.exe"
$Pip = Join-Path $PSScriptRoot "venv_win\Scripts\pip.exe"

Write-Host "==> Upgrading pip..."
& $Py -m pip install --upgrade pip setuptools wheel

Write-Host "==> Installing PyTorch (CUDA 12.4 wheels)..."
& $Pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "==> Installing VizDoom and core packages..."
& $Pip install "vizdoom>=1.2.4" opencv-python-headless

Write-Host "==> Installing SNN and training utilities..."
& $Pip install "snntorch>=0.9" tensorboard imageio imageio-ffmpeg tqdm

Write-Host "==> Installing analysis and testing utilities..."
& $Pip install matplotlib pytest

Write-Host ""
Write-Host "Setup complete!"
Write-Host "  Activate the environment with:  .\venv_win\Scripts\Activate.ps1"
Write-Host "  Start training with:            python training\train_doom.py"
Write-Host "  Run tests with:                 pytest tests\"
