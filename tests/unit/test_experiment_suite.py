import json

import torch

from kgdmc.experiment_suite import (
    generate_modality_corruptions,
    run_baseline_modality_matrix,
    run_baseline_triplet_matrix,
    run_experiment_suite,
)


def _caches(tmp_path):
    ids = ["a", "b", "c", "d"]
    torch.manual_seed(3)
    text = tmp_path / "text.pt"
    image = tmp_path / "image.pt"
    torch.save({"entity_ids": ids, "embeddings": torch.randn(4, 8)}, text)
    torch.save({"entity_ids": ids, "embeddings": torch.randn(4, 8)}, image)
    return ids, text, image


def test_modality_noise_and_baseline_matrices(tmp_path):
    ids, text, image = _caches(tmp_path)
    text_vectors = dict(zip(ids, torch.load(text, weights_only=False)["embeddings"], strict=True))
    image_vectors = dict(zip(ids, torch.load(image, weights_only=False)["embeddings"], strict=True))
    rows = generate_modality_corruptions(ids, text_vectors, image_vectors, fraction=1.0, strategy="inter_similar")
    assert len(rows) == len(ids)
    assert all(row["entity_id"] != row["wrong_image_entity_id"] for row in rows)
    modality_manifest = tmp_path / "modality.json"
    modality_manifest.write_text(json.dumps({"records": rows}))
    result = run_baseline_modality_matrix(modality_manifest, text, image, tmp_path / "modality.jsonl")
    assert result["baselines"] == 4

    triplet_manifest = tmp_path / "triplets.json"
    triplet_manifest.write_text(json.dumps({"records": [{
        "original": ["a", "r", "b"], "corrupted": ["c", "r", "b"], "error_type": "random_head",
    }]}))
    result = run_baseline_triplet_matrix(triplet_manifest, text, image, tmp_path / "triplets.jsonl")
    assert result["baselines"] == 12


def test_suite_writes_complete_plan_and_can_run_bounded(tmp_path):
    ids, text, image = _caches(tmp_path)
    graph = tmp_path / "train.txt"
    graph.write_text("a\tr\tb\nb\tr\tc\nc\tr\td\n")
    result = run_experiment_suite(
        tmp_path,
        "fb15k237",
        text,
        image,
        graph,
        groups=("main", "baselines"),
        main_seeds=(42,),
        steps=1,
        batch_size=2,
        device="cpu",
        max_runs=1,
    )
    assert result["runs_attempted"] == 1
    output_dir = tmp_path / "runs" / "paper2_suite" / "fb15k237"
    assert (output_dir / "experiment_plan.json").exists()
    assert (output_dir / "coverage.json").exists()
    assert (output_dir / "summary.csv").exists()
