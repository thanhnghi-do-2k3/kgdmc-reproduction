import json

import torch

from kgdmc.inference import run_inference
from kgdmc.training import run_training


def test_inference_persists_predictions(tmp_path):
    ids = ["a", "b", "c"]
    text_cache = tmp_path / "text.pt"
    image_cache = tmp_path / "image.pt"
    manifest = tmp_path / "corruption.json"
    torch.save({"entity_ids": ids, "embeddings": torch.randn(3, 8)}, text_cache)
    torch.save({"entity_ids": ids, "embeddings": torch.randn(3, 8)}, image_cache)
    manifest.write_text(json.dumps({"records": [{"source_split": "train", "row_index": 0, "original": ["a", "r", "b"], "corrupted": ["c", "r", "b"], "error_type": "head", "label": 1}]}))
    checkpoint = tmp_path / "run.pt"
    run_training(text_cache, image_cache, manifest, checkpoint, steps=1)
    result = run_inference(checkpoint, text_cache, image_cache, manifest, tmp_path / "predictions.jsonl")
    assert result["rows"] == 1
    assert (tmp_path / "predictions.jsonl").read_text()

