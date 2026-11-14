$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
# Install the project dependencies, then install the CUDA build explicitly.
# The default PyPI resolution can select a CPU-only torch wheel on Windows.
& .\.venv\Scripts\python.exe -m pip install -e '.[dev]'
& .\.venv\Scripts\python.exe -m pip install transformers pillow
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m kgdmc.cli env doctor --device cuda
nvidia-smi

Write-Host 'Windows GPU environment is ready.'
