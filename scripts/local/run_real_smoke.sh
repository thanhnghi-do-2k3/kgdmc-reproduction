#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

echo '[1/9] Checking local CLI and required real-data manifest'
command -v kgdmc >/dev/null
test -f datasets/modality/fb15k237/images.tsv || {
  echo 'BLOCKED_DATA: provide datasets/modality/fb15k237/images.tsv' >&2
  exit 2
}

echo '[2/9] Fetching graph/text only when absent'
if [[ ! -f datasets/raw/fb15k237/train.txt || ! -f datasets/raw/fb15k237/entity2text.txt ]]; then
  kgdmc data fetch --dataset fb15k237 --components graph,text --resume
fi

echo '[3/9] Validating real graph/text'
kgdmc data verify --dataset fb15k237 --strict

echo '[4/9] Encoding/cache full real text with BERT'
TEXT_CACHE="$(kgdmc data encode-text --dataset fb15k237 --batch-size 16 --device auto | python3 -c 'import json,sys; print(json.load(sys.stdin)["cache"])')"

echo '[5/9] Encoding/cache explicitly mapped local images with ViT'
IMAGE_CACHE="$(kgdmc data encode-images --dataset fb15k237 --images-from datasets/modality/fb15k237 --image-manifest datasets/modality/fb15k237/images.tsv --batch-size 2 --device auto | python3 -c 'import json,sys; print(json.load(sys.stdin)["cache"])')"

echo '[6/9] Creating deterministic real-subgraph corruption manifest'
kgdmc data corrupt --dataset fb15k237 --split train \
  --triples-file datasets/slices/fb15k237/real_smoke.txt \
  --entity-ids-file datasets/slices/fb15k237/real_smoke_entities.txt \
  --fraction 1.0 --seed 42

echo '[7/9] Training DMR→MTC and saving checkpoint'
kgdmc experiments run --dataset fb15k237 --profile real_smoke \
  --text-cache "$TEXT_CACHE" \
  --image-cache "$IMAGE_CACHE" \
  --corruption-manifest datasets/manifests/fb15k237/corruptions_train_seed42.json \
  --output runs/fb15k237_real_smoke_seed42.pt --steps 2 --device auto

echo '[8/9] Running inference and rank metrics'
kgdmc experiments infer \
  --checkpoint runs/fb15k237_real_smoke_seed42.pt \
  --text-cache "$TEXT_CACHE" \
  --image-cache "$IMAGE_CACHE" \
  --corruption-manifest datasets/manifests/fb15k237/corruptions_train_seed42.json \
  --output runs/fb15k237_real_smoke_predictions.jsonl --device auto

echo '[9/9] Readiness report and tests'
kgdmc data readiness --dataset fb15k237
pytest -q

echo "DONE: checkpoint=runs/fb15k237_real_smoke_seed42.pt"
echo "DONE: predictions=runs/fb15k237_real_smoke_predictions.jsonl"

