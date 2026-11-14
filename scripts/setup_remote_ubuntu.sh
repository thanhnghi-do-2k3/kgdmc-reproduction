#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh Ubuntu/CUDA setup for the compute-adapted Paper-2/KGDMC runner.
# Run this from the extracted ~/paper_2 directory.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project: $PROJECT_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Installing Python 3..."
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-dev build-essential
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found; CUDA verification may fail."
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "Installing python3-venv..."
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-dev build-essential
fi

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install CUDA-enabled PyTorch wheels. These include the required CUDA runtime;
# a system CUDA toolkit is not required for this project.
python -m pip install --upgrade \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install --upgrade -e '.[dev]' transformers pillow

export PYTHONPATH="$PROJECT_ROOT/src"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

python - <<'PY'
import sys
import torch
import kgdmc

print("python=", sys.executable)
print("torch=", torch.__version__)
print("cuda=", torch.cuda.is_available())
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check the NVIDIA driver and CUDA PyTorch wheel.")
print("kgdmc=", kgdmc.__file__)
PY

python -m py_compile src/kgdmc/training.py scripts/run_recovery_subset.py
PYTHONPATH=src pytest -q

echo
echo "Setup complete. Activate later with:"
echo "  cd $PROJECT_ROOT && source .venv/bin/activate && export PYTHONPATH=\"\$PWD/src\""
echo "Then login to W&B with:"
echo "  export WANDB_BASE_URL=https://api.forge.coreweave.com"
echo "  .venv/bin/wandb login"
