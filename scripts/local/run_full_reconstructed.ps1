param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath,
    [int]$Steps = 30,
    [int]$BatchSize = 32,
    [int]$TextBatchSize = 32,
    [int]$ImageBatchSize = 16,
    [ValidateSet('disabled', 'offline', 'online')]
    [string]$WandbMode = 'disabled'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Cli = @('-m', 'kgdmc.cli')
$env:WANDB_MODE = $WandbMode

if (-not (Test-Path $Python)) { throw 'Missing .venv. Run scripts/local/setup_windows.ps1 first.' }
if (-not (Test-Path $ArchivePath)) { throw "Archive not found: $ArchivePath" }
nvidia-smi | Out-Host

function Invoke-Kgdmc {
    param([string[]]$Arguments)
    & $Python @Cli @Arguments
    if ($LASTEXITCODE -ne 0) { throw "kgdmc failed: $($Arguments -join ' ')" }
}

function Invoke-KgdmcJson {
    param([string[]]$Arguments)
    $raw = (& $Python @Cli @Arguments | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "kgdmc failed: $($Arguments -join ' ')" }
    return ($raw | ConvertFrom-Json)
}

$ImageRoot = 'datasets/modality/fb15k237'
$ImageManifest = Join-Path $ImageRoot 'images_full.tsv'
$ImportOutput = Join-Path $ImageRoot 'full'

Write-Host '[1/8] Importing the local combined image archive'
& $Python scripts/local/import_fb15k_full_archive.py $ArchivePath --output $ImportOutput --allow-incomplete
if ($LASTEXITCODE -ne 0) { throw 'Image archive import failed.' }
if (-not (Test-Path $ImageManifest)) { throw 'Full image manifest was not created.' }

Write-Host '[2/8] Validating graph/text and image inventory'
Invoke-Kgdmc -Arguments @('data', 'verify', '--dataset', 'fb15k237', '--strict')
Invoke-Kgdmc -Arguments @('data', 'readiness', '--dataset', 'fb15k237')

Write-Host '[3/8] Encoding text and images with CUDA (cached)'
$TextResult = Invoke-KgdmcJson -Arguments @('data', 'encode-text', '--dataset', 'fb15k237', '--batch-size', "$TextBatchSize", '--device', 'cuda')
$ImageResult = Invoke-KgdmcJson -Arguments @('data', 'encode-images', '--dataset', 'fb15k237', '--images-from', $ImageRoot, '--image-manifest', $ImageManifest, '--batch-size', "$ImageBatchSize", '--device', 'cuda')

$CorruptionManifest = 'datasets/manifests/fb15k237/full_reconstructed_corruptions_train_seed42.json'
$Checkpoint = 'runs/fb15k237_full_reconstructed_seed42.pt'

Write-Host '[4/8] Creating 5-percent generic corruption manifest'
Invoke-Kgdmc -Arguments @('data', 'corrupt', '--dataset', 'fb15k237', '--split', 'train', '--fraction', '0.05', '--seed', '42', '--output', $CorruptionManifest)

Write-Host '[5/8] Training'
Invoke-Kgdmc -Arguments @('experiments', 'run', '--dataset', 'fb15k237', '--profile', 'full_reconstructed', '--text-cache', $TextResult.cache, '--image-cache', $ImageResult.cache, '--corruption-manifest', $CorruptionManifest, '--graph-file', 'datasets/raw/fb15k237/train.txt', '--output', $Checkpoint, '--steps', "$Steps", '--batch-size', "$BatchSize", '--device', 'cuda')

Write-Host '[6/8] Inference'
Invoke-Kgdmc -Arguments @('experiments', 'infer', '--checkpoint', $Checkpoint, '--text-cache', $TextResult.cache, '--image-cache', $ImageResult.cache, '--corruption-manifest', $CorruptionManifest, '--graph-file', 'datasets/raw/fb15k237/train.txt', '--output', 'runs/fb15k237_full_reconstructed_predictions.jsonl', '--batch-size', "$BatchSize", '--device', 'cuda')

Write-Host '[7/8] Regression tests'
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed.' }

Write-Host '[8/8] Completed: DATA_RECONSTRUCTED_COMPUTE_ADAPTED'
