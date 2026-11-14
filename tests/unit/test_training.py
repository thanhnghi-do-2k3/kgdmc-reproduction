import json

import torch

from kgdmc.training import run_training


def test_training_uses_aligned_feature_caches(tmp_path):
    text_cache = tmp_path / "text.pt"
    image_cache = tmp_path / "image.pt"
    manifest = tmp_path / "corruption.json"
    ids = ["a", "b", "c"]
    torch.save({"entity_ids": ids, "embeddings": torch.randn(3, 8)}, text_cache)
    torch.save({"entity_ids": ids, "embeddings": torch.randn(3, 8)}, image_cache)
    manifest.write_text(json.dumps({"records": [{"source_split": "train", "row_index": 0, "original": ["a", "r", "b"], "corrupted": ["c", "r", "b"], "error_type": "head", "label": 1}]}))
    result = run_training(text_cache, image_cache, manifest, tmp_path / "run.pt", steps=1)
    assert result["status"].startswith("TRAINED")
    assert (tmp_path / "run.pt").exists()
    resumed = run_training(text_cache, image_cache, manifest, tmp_path / "run2.pt", steps=1, resume_from=tmp_path / "run.pt")
    assert resumed["completed_steps"] == 2


def test_training_accepts_batched_full_graph_contexts(tmp_path):
    text_cache = tmp_path / "text.pt"
    image_cache = tmp_path / "image.pt"
    graph = tmp_path / "train.txt"
    manifest = tmp_path / "corruption.json"
    ids = ["a", "b", "c", "d"]
    torch.save({"entity_ids": ids, "embeddings": torch.randn(4, 8)}, text_cache)
    torch.save({"entity_ids": ids, "embeddings": torch.randn(4, 8)}, image_cache)
    graph.write_text("a\tr\tb\nb\tr\tc\nc\tr\td\n")
    manifest.write_text(json.dumps({"records": [
        {"source_split": "train", "row_index": 0, "original": ["a", "r", "b"], "corrupted": ["c", "r", "b"], "error_type": "head", "label": 1},
        {"source_split": "train", "row_index": 1, "original": ["b", "r", "c"], "corrupted": ["d", "r", "c"], "error_type": "head", "label": 1},
    ]}))
    result = run_training(text_cache, image_cache, manifest, tmp_path / "full.pt", steps=1, batch_size=1, graph_file=graph)
    assert result["batch_size"] == 1
    assert result["records"] == 2

