#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

ARCHIVE_PATH="${1:?Usage: bash scripts/local/run_full_reconstructed.sh /path/to/fb15k_wn18_images.zip}"
STEPS="${STEPS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"

IMAGE_ROOT="datasets/modality/fb15k237"
IMAGE_MANIFEST="$IMAGE_ROOT/images_full.tsv"
TEXT_CACHE="$(kgdmc data encode-text --dataset fb15k237 --batch-size 128 --device auto | python3 -c 'import json,sys; print(json.load(sys.stdin)["cache"])')"

echo '[1/8] Importing and validating the local combined image archive'
python3 scripts/local/import_fb15k_full_archive.py \
  "$ARCHIVE_PATH" \
  --output "$IMAGE_ROOT/full" \
  --allow-incomplete

test -s "$IMAGE_MANIFEST"

echo '[2/8] Validating graph/text and imported image inventory'
kgdmc data verify --dataset fb15k237 --strict
kgdmc data readiness --dataset fb15k237

echo '[3/8] Encoding all imported entity images with ViT (cached)'
IMAGE_CACHE="$(kgdmc data encode-images \
  --dataset fb15k237 \
  --images-from "$IMAGE_ROOT" \
  --image-manifest "$IMAGE_MANIFEST" \
  --batch-size 32 \
  --device auto | python3 -c 'import json,sys; print(json.load(sys.stdin)["cache"])')"

echo '[4/8] Generating deterministic 5-percent generic triplet corruptions'
CORRUPTION_MANIFEST="datasets/manifests/fb15k237/full_reconstructed_corruptions_train_seed42.json"
kgdmc data corrupt --dataset fb15k237 --split train \
  --fraction 0.05 --seed 42 \
  --output "$CORRUPTION_MANIFEST"

echo '[5/8] Training the batched reconstructed full-available run'
CHECKPOINT="runs/fb15k237_full_reconstructed_seed42.pt"
kgdmc experiments run --dataset fb15k237 --profile full_reconstructed \
  --text-cache "$TEXT_CACHE" \
  --image-cache "$IMAGE_CACHE" \
  --corruption-manifest "$CORRUPTION_MANIFEST" \
  --graph-file datasets/raw/fb15k237/train.txt \
  --output "$CHECKPOINT" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --device auto

echo '[6/8] Running inference and ranking metrics'
kgdmc experiments infer \
  --checkpoint "$CHECKPOINT" \
  --text-cache "$TEXT_CACHE" \
  --image-cache "$IMAGE_CACHE" \
  --corruption-manifest "$CORRUPTION_MANIFEST" \
  --graph-file datasets/raw/fb15k237/train.txt \
  --output runs/fb15k237_full_reconstructed_predictions.jsonl \
  --batch-size "$BATCH_SIZE" \
  --device auto

echo '[7/8] Running regression tests'
pytest -q

echo '[8/8] Completed reconstructed full-available run'
echo "label=DATA_RECONSTRUCTED_COMPUTE_ADAPTED"
echo "checkpoint=$CHECKPOINT"
echo "predictions=runs/fb15k237_full_reconstructed_predictions.jsonl"
