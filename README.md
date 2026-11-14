# KGDMC reproduction study

> Scope: this is a public code scaffold for the group's compute-adapted KGDMC implementation, not evidence of a complete paper reproduction. Dataset/image provenance, a clean installation, held-out evaluation, and full GPU reruns remain unverified.

Install from the source checkout in editable mode as shown below. A regular non-editable wheel install was checked and resolves the CLI data root inside the Python environment rather than the checkout; that installation mode is not supported by this package layout.

This folder contains the DMR/MTC reconstruction and scripts; datasets, image manifests, feature caches and results are not included. The available image source and corruption universe are reconstructed; outputs must be labeled `DATA_RECONSTRUCTED_COMPUTE_ADAPTED` unless the official assets are verified.

## Reported hardware and environment limits

The group's rental record for the KGDMC attempt lists Ubuntu 24.04, an Intel Core i5 (generation not recorded), and an NVIDIA RTX 5060 Ti. This is the reported rental configuration, not per-run telemetry. Historical Python, PyTorch, and CUDA versions were not recovered from W&B. The package declares Python 3.10 or later; the Ubuntu setup script installs a CUDA 12.8 PyTorch wheel for a new run, which does not establish the historical software versions.

The recovered historical W&B group name includes `rtx5070`, which conflicts with that rental screenshot. A group label is not hardware telemetry; the run metadata records only CUDA, so the GPU used by each historical run remains unverified.

For a fresh Ubuntu setup from the repository root, run:

```bash
bash scripts/setup_remote_ubuntu.sh
```

That setup script installs dependencies, checks for an NVIDIA GPU, and runs the local test suite. It prepares an execution environment; it does not validate the official KGDMC datasets or reproduce the published benchmark.

## Windows GPU setup

```powershell
cd C:\Users\<USER>\kgdmc-reproduction
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install torch torchvision transformers pillow numpy pandas tqdm wandb
wandb login
```

## Prepare and verify data

```powershell
python -m kgdmc.cli data fetch --dataset fb15k237 --components graph,text --resume
python -m kgdmc.cli data verify --dataset fb15k237 --strict
python -m kgdmc.cli data inspect --dataset fb15k237
```

After extracting the local image archive and creating a valid manifest, encode caches:

```powershell
python -m kgdmc.cli data encode-text --dataset fb15k237 --batch-size 16 --device cuda
python -m kgdmc.cli data encode-images --dataset fb15k237 `
  --images-from .\datasets\modality\fb15k237 `
  --image-manifest .\datasets\modality\fb15k237\images_full_clean.tsv `
  --batch-size 16 --device cuda
```

## Adapted suite and evaluation warning

The script `scripts/run_recovery_subset.py` passes the same corruption manifest to training and inference. The `experiments suite` implementation also passes the same manifest to training and inference for its triplet experiments. Their triplet precision/recall measures training-pair ranking, not held-out performance. Do not report those metrics as independent evaluation or paper reproduction. Historical subset runs were saved locally with W&B disabled; changing the project name does not upload or recreate them.

The following is the original adapted-suite invocation for reference, not a verified clean-environment or official benchmark rerun. New main and ablation runs belong in the [KGDMC main/ablation W&B project](https://forge.coreweave.com/wandb/nghidt-hcmus/kgdmc-training-ablation/workspace?nw=8ct0en9rizi); subset diagnostics belong in the separate [KGDMC subset project](https://forge.coreweave.com/wandb/nghidt-hcmus/kgdmc-subset-diagnostics/workspace?nw=jsdwd6ilo). Both projects were checked in an incognito session on 14 November 2026: the main/ablation project showed 19 runs and the subset project showed four entries, with runs and charts visible. These counts describe logged entries, not benchmark completeness. The subset experiment reuses corruption pairs between training and evaluation, so its scores are not held-out results. W&B's move operation did not transfer associated artifacts. Replace the cache paths below with actual paths and verify the evaluation protocol before interpreting any new metrics.

```powershell
$env:WANDB_MODE="online"
$env:WANDB_PROJECT="kgdmc-training-ablation"
$env:WANDB_RUN_GROUP="kgdmc-main-seeds42-46"
python -m kgdmc.cli experiments suite `
  --dataset fb15k237 `
  --text-cache <TEXT_CACHE> `
  --image-cache <IMAGE_CACHE> `
  --graph-file .\datasets\raw\fb15k237\train.txt `
  --groups main,ablation,hyperparam,robustness,baselines `
  --main-seeds 42,43,44,45,46 `
  --secondary-seed 42 `
  --steps 200 --batch-size 128 --device cuda `
  --output-dir .\runs\paper2_full_suite_rtx5070 2>&1 |
  Tee-Object -FilePath .\paper2_full_suite_rtx5070.log
```

The suite runs sequentially and can take a long time. Check progress from another terminal with `Get-Process python`, `nvidia-smi`, and `Get-Content .\paper2_full_suite_rtx5070.log -Tail 30 -Wait`. Do not compare reconstructed subset metrics directly with the paper's official tables.

## Smoke test

Provide existing aligned feature caches and a compatible corruption manifest; paths below are placeholders:

```powershell
$env:WANDB_MODE="disabled"
python -m kgdmc.cli experiments run --dataset fb15k237 --profile real_smoke `
  --text-cache <TEXT_CACHE> --image-cache <IMAGE_CACHE> `
  --corruption-manifest <MANIFEST> --output .\cpu_smoke.pt `
  --steps 1 --batch-size 32 --device cpu
python -m kgdmc.cli experiments infer --checkpoint .\cpu_smoke.pt `
  --text-cache <TEXT_CACHE> --image-cache <IMAGE_CACHE> `
  --corruption-manifest <MANIFEST> --output .\cpu_smoke_predictions.jsonl `
  --batch-size 32 --device cpu
```

This example reuses the training manifest for inference to check execution only. Its metrics are not independent held-out results and are not paper-level results. The caches and manifest are not distributed in this repository.

The `WANDB_PROJECT` setting above selects the destination for new runs, including recovery scripts when set in the same shell. It does not move existing runs. The strict graph/text verifier does not certify image-corpus availability.

## Local tests

```powershell
python -m pip install -e ".[dev,ml]"
$env:WANDB_MODE="disabled"
python -m pytest -q -p no:cacheprovider
python -m kgdmc.cli --help
```

The test suite is an execution check, not a full reproduction. The image manifest, aligned caches and permission to use them must be established separately.

