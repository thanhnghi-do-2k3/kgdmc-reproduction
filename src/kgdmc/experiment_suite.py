from __future__ import annotations

"""Paper-2/KGDMC experiment matrix.

The suite deliberately keeps two labels separate:

* ``KGDMC_RECONSTRUCTED`` runs the repository's DMR/MTC implementation on the
  available real feature caches.
* ``BASELINE_COMPUTE_ADAPTED`` runs deterministic baseline adapters when the
  authors' external implementations are unavailable.  These are useful for a
  controlled local comparison, but are never reported as official baseline
  reproductions.
"""

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .corruption import Corruption, generate_corruptions, save_manifest, read_triples
from .evaluation import evaluate_rankings
from .inference import run_inference
from .models import DMR, MTC
from .training import _load_cache, run_training


MODALITY_BASELINES = ["CLIP", "ViLT", "ALBEF", "METER"]
TRIPLET_BASELINES = [
    "TransE", "DistMult", "ComplEx", "KGTtm", "CAGED", "KG-BERT", "StAR",
    "CSProm-KG", "CCA", "IKRL", "RSME", "VISTA",
]
DMR_ABLATIONS = ["w/o L_DA", "w/o L_IntraDD", "w/o L_InterDD", "w/o z_ti", "w/o C_M"]
MTC_ABLATIONS = ["w/o v_e", "w/o w", "w/o S_LT", "w/o S_GS", "w/o S_GP", "w/o C_T"]
HYPERPARAM_SWEEPS = {
    "alpha": [5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    "beta": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    "neighbor_limit": [30, 60, 90, 120, 150],
    "path_limit": [5, 10, 15, 20, 25],
    "lambda_structure": [0.0, 0.5, 1.0, 1.5, 2.0],
    "delta_path": [0.0, 0.5, 1.0, 1.5, 2.0],
}
TRIPLET_NOISE = ["random", "similar", "adversarial"]
MODALITY_NOISE = ["random", "intra_similar", "inter_similar"]


def paper2_experiment_catalog() -> dict:
    return {
        "RQ1": {
            "main": "Tables 2-3: overall modality/triplet detection, two datasets, five seeds",
            "modality_baselines": MODALITY_BASELINES,
            "triplet_baselines": TRIPLET_BASELINES,
            "metrics": ["precision@1%", "precision@2%", "precision@3%", "precision@4%", "precision@5%", "recall@1%", "recall@2%", "recall@3%", "recall@4%", "recall@5%"],
        },
        "RQ2": {
            "ablation_modality": DMR_ABLATIONS,
            "ablation_triplet": MTC_ABLATIONS,
            "metrics": ["precision@1%", "precision@3%", "precision@5%", "recall@1%", "recall@3%", "recall@5%"],
        },
        "RQ3": {"sweeps": HYPERPARAM_SWEEPS, "metric": "precision@5%"},
        "RQ4": {"triplet_noise": TRIPLET_NOISE, "modality_noise": MODALITY_NOISE, "metric": "precision@5%"},
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item())


def _feature_maps(text_cache: Path, image_cache: Path) -> tuple[list[str], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    text_ids, text = _load_cache(text_cache)
    image_ids, image = _load_cache(image_cache)
    image_index = {entity_id: index for index, entity_id in enumerate(image_ids)}
    text_index = {entity_id: index for index, entity_id in enumerate(text_ids)}
    common = [entity_id for entity_id in text_ids if entity_id in image_index]
    return common, {entity_id: text[text_index[entity_id]] for entity_id in common}, {entity_id: image[image_index[entity_id]] for entity_id in common}


def generate_modality_corruptions(
    entity_ids: Sequence[str],
    text_vectors: Mapping[str, torch.Tensor],
    image_vectors: Mapping[str, torch.Tensor],
    fraction: float = 0.05,
    seed: int = 42,
    strategy: str = "random",
    candidate_limit: int = 128,
) -> list[dict]:
    """Create real-image swaps for RQ4 modality noise.

    ``intra_similar`` chooses a nearby image embedding and ``inter_similar`` chooses
    an image whose ViT representation is near the target text representation. The
    latter is intentionally labelled compute-adapted because BERT and ViT spaces are
    not the authors' VLM joint space.
    """
    if strategy not in MODALITY_NOISE:
        raise ValueError(f"Unknown modality noise strategy: {strategy}")
    rng = random.Random(seed)
    ids = sorted(set(entity_ids))
    if len(ids) < 2:
        raise ValueError("At least two aligned entities are required for modality corruption")
    selected = rng.sample(ids, max(1, round(len(ids) * fraction)))
    rows = []
    for row_index, target in enumerate(selected):
        candidates = [candidate for candidate in ids if candidate != target]
        if strategy == "random":
            source = rng.choice(candidates)
        else:
            if len(candidates) > candidate_limit:
                candidates = rng.sample(candidates, candidate_limit)
            reference = image_vectors[target] if strategy == "intra_similar" else text_vectors[target]
            candidate_matrix = F.normalize(torch.stack([image_vectors[candidate] for candidate in candidates]), dim=1)
            reference_vector = F.normalize(reference.reshape(1, -1), dim=1).squeeze(0)
            source = candidates[int(torch.argmax(candidate_matrix @ reference_vector).item())]
        rows.append({
            "row_index": row_index,
            "entity_id": target,
            "wrong_image_entity_id": source,
            "error_type": strategy,
            "label": 1,
        })
    return rows


def save_modality_manifest(path: Path, rows: list[dict], metadata: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "records": rows}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode()
    path.write_bytes(encoded)
    return {"path": str(path), "records": len(rows), "sha256": hashlib.sha256(encoded).hexdigest()}


def _model_from_checkpoint(checkpoint: Path, input_dim: int, device: str) -> tuple[dict, DMR, MTC]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    dmr = DMR(input_dim=input_dim, alpha=float(config.get("alpha", 0.005)), beta=float(config.get("beta", 0.01)), ablation=config.get("ablation")).to(device)
    mtc = MTC(dim=256, relation_dim=256, lambda_structure=float(config.get("lambda_structure", 0.5)), delta_path=float(config.get("delta_path", 1.5)), ablation=config.get("ablation")).to(device)
    dmr.load_state_dict(state["dmr"])
    mtc.load_state_dict(state["mtc"])
    dmr.eval(); mtc.eval()
    return state, dmr, mtc


def evaluate_modality_checkpoint(
    checkpoint: Path,
    text_cache: Path,
    image_cache: Path,
    manifest: Path,
    output: Path,
    device: str = "cpu",
    batch_size: int = 128,
) -> dict:
    """Evaluate real-image swaps with DMR image reconstruction error."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    text_ids, text = _load_cache(text_cache)
    image_ids, image = _load_cache(image_cache)
    text_index = {entity_id: index for index, entity_id in enumerate(text_ids)}
    image_index = {entity_id: index for index, entity_id in enumerate(image_ids)}
    payload = json.loads(manifest.read_text())
    records = payload["records"]
    state, dmr, _ = _model_from_checkpoint(checkpoint, text.shape[1], device)
    text_device = text.to(device)
    image_device = image.to(device)
    rows, scores, labels = [], [], []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            target = torch.tensor([text_index[item["entity_id"]] for item in batch], device=device)
            good_image = torch.tensor([image_index[item["entity_id"]] for item in batch], device=device)
            bad_image = torch.tensor([image_index[item["wrong_image_entity_id"]] for item in batch], device=device)
            good = dmr(text_device[target], image_device[good_image]).image_score.tolist()
            bad = dmr(text_device[target], image_device[bad_image]).image_score.tolist()
            for item, good_score, bad_score in zip(batch, good, bad, strict=True):
                rows.append({**item, "correct_image_score": good_score, "wrong_image_score": bad_score})
                scores.extend([good_score, bad_score]); labels.extend([0, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))
    return {"rows": len(rows), "predictions": str(output), "metrics": evaluate_rankings(scores, labels) if scores else {}, "status": "KGDMC_RECONSTRUCTED_MODALITY_EVAL", "checkpoint": str(checkpoint), "config": state.get("config", {})}


def _hash_vector(value: str, dimension: int) -> torch.Tensor:
    digest = hashlib.sha256(value.encode()).digest()
    raw = torch.tensor([(byte / 127.5) - 1.0 for byte in digest], dtype=torch.float32)
    return raw.repeat(math.ceil(dimension / len(raw)))[:dimension]


def _baseline_score(name: str, triple: tuple[str, str, str], text_vectors: Mapping[str, torch.Tensor], image_vectors: Mapping[str, torch.Tensor]) -> float:
    h, relation, t = triple
    text_h, text_t = text_vectors[h], text_vectors[t]
    image_h, image_t = image_vectors[h], image_vectors[t]
    dimension = text_h.numel()
    relation_vector = _hash_vector(relation, dimension)
    if name in {"TransE", "KGTtm"}:
        return float(torch.linalg.vector_norm(text_h + relation_vector - text_t).item())
    if name == "DistMult":
        return float(-torch.sum(text_h * relation_vector * text_t).item())
    if name == "ComplEx":
        half = dimension // 2
        h_real, h_imag = text_h[:half], text_h[half : half * 2]
        t_real, t_imag = text_t[:half], text_t[half : half * 2]
        r_real, r_imag = relation_vector[:half], relation_vector[half : half * 2]
        plausibility = torch.sum(h_real * r_real * t_real + h_real * r_imag * t_imag + h_imag * r_real * t_imag - h_imag * r_imag * t_real)
        return float(-plausibility.item())
    if name in {"IKRL", "RSME", "VISTA"}:
        h = F.normalize(text_h + image_h, dim=0)
        t = F.normalize(text_t + image_t, dim=0)
    elif name in {"CAGED", "CCA"}:
        h = F.normalize(text_h, dim=0)
        t = F.normalize(text_t, dim=0)
    else:
        h, t = text_h, text_t
    local = torch.linalg.vector_norm(h + relation_vector - t)
    return float(local.item())


def run_baseline_triplet_matrix(
    manifest: Path,
    text_cache: Path,
    image_cache: Path,
    output: Path,
) -> dict:
    """Run deterministic baseline adapters for every Table-3/Table-6 name."""
    text_ids, text = _load_cache(text_cache)
    image_ids, image = _load_cache(image_cache)
    image_index = {entity_id: index for index, entity_id in enumerate(image_ids)}
    text_index = {entity_id: index for index, entity_id in enumerate(text_ids)}
    text_vectors = {entity_id: text[text_index[entity_id]] for entity_id in text_ids if entity_id in image_index}
    image_vectors = {entity_id: image[image_index[entity_id]] for entity_id in text_vectors}
    records = json.loads(manifest.read_text())["records"]
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in TRIPLET_BASELINES:
        scores, labels = [], []
        for item in records:
            original = tuple(item["original"]); corrupted = tuple(item["corrupted"])
            if not all(entity in text_vectors for entity in (*original[::2], *corrupted[::2])):
                continue
            original_score = _baseline_score(name, original, text_vectors, image_vectors)
            corrupted_score = _baseline_score(name, corrupted, text_vectors, image_vectors)
            scores.extend([original_score, corrupted_score]); labels.extend([0, 1])
        rows.append({"baseline": name, "family": "triplet", "fidelity": "BASELINE_COMPUTE_ADAPTED_PROXY", "metrics": evaluate_rankings(scores, labels) if scores else {}, "records": len(scores) // 2})
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return {"path": str(output), "baselines": len(rows), "status": "BASELINE_COMPUTE_ADAPTED"}


def run_baseline_modality_matrix(
    manifest: Path,
    text_cache: Path,
    image_cache: Path,
    output: Path,
) -> dict:
    text_ids, text = _load_cache(text_cache)
    image_ids, image = _load_cache(image_cache)
    image_index = {entity_id: index for index, entity_id in enumerate(image_ids)}
    text_index = {entity_id: index for index, entity_id in enumerate(text_ids)}
    rows_in = json.loads(manifest.read_text())["records"]
    rows = []
    for name in MODALITY_BASELINES:
        scores, labels = [], []
        for item in rows_in:
            entity = item["entity_id"]; wrong = item["wrong_image_entity_id"]
            if entity not in image_index or wrong not in image_index or entity not in text_index:
                continue
            text_vector = text[text_index[entity]]
            def score(image_vector: torch.Tensor) -> float:
                cosine_distance = 1.0 - _cosine(text_vector, image_vector)
                normalized_text = F.normalize(text_vector.reshape(1, -1), dim=1).squeeze(0)
                normalized_image = F.normalize(image_vector.reshape(1, -1), dim=1).squeeze(0)
                l1_distance = torch.mean(torch.abs(normalized_text - normalized_image)).item()
                norm_gap = abs(float(text_vector.norm().item()) - float(image_vector.norm().item())) / max(1.0, float(text_vector.norm().item()))
                if name == "CLIP":
                    return cosine_distance
                if name == "ViLT":
                    return 0.75 * cosine_distance + 0.25 * l1_distance
                if name == "ALBEF":
                    return 0.5 * cosine_distance + 0.3 * l1_distance + 0.2 * norm_gap
                return 0.4 * cosine_distance + 0.3 * l1_distance + 0.3 * norm_gap
            good = score(image[image_index[entity]])
            bad = score(image[image_index[wrong]])
            scores.extend([good, bad]); labels.extend([0, 1])
        rows.append({"baseline": name, "family": "modality", "fidelity": "BASELINE_COMPUTE_ADAPTED_PROXY", "metrics": evaluate_rankings(scores, labels) if scores else {}, "records": len(scores) // 2})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return {"path": str(output), "baselines": len(rows), "status": "BASELINE_COMPUTE_ADAPTED"}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def run_experiment_suite(
    root: Path,
    dataset: str,
    text_cache: Path,
    image_cache: Path,
    graph_file: Path,
    groups: Sequence[str] = ("main", "ablation", "hyperparam", "robustness", "baselines"),
    main_seeds: Sequence[int] = (42, 43, 44, 45, 46),
    secondary_seed: int = 42,
    steps: int = 30,
    batch_size: int = 128,
    device: str = "auto",
    output_dir: Path | None = None,
    max_runs: int | None = None,
) -> dict:
    """Run all selected paper-2 experiment families and write an auditable ledger."""
    output_dir = output_dir or root / "runs" / "paper2_suite" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    triples = read_triples(graph_file)
    entity_ids, text_vectors, image_vectors = _feature_maps(text_cache, image_cache)
    relation_pool = sorted({triple[1] for triple in triples})
    results: list[dict] = []
    run_count = 0
    base_checkpoint: Path | None = None
    base_manifest: Path | None = None

    def record(group: str, name: str, status: str, **payload: object) -> None:
        results.append({"group": group, "name": name, "status": status, **payload})

    def can_run() -> bool:
        return max_runs is None or run_count < max_runs

    def triplet_manifest(strategy: str, seed: int) -> Path:
        path = output_dir / "manifests" / f"triplet_{strategy}_seed{seed}.json"
        records = generate_corruptions(triples, fraction=0.05, seed=seed, entity_pool=entity_ids, relation_pool=relation_pool, strategy=strategy, entity_vectors=image_vectors)
        save_manifest(path, records, {"dataset": dataset, "strategy": strategy, "seed": seed, "fidelity": "DATA_RECONSTRUCTED_COMPUTE_ADAPTED"})
        return path

    def train_and_infer(group: str, name: str, seed: int, manifest: Path, **params: object) -> dict | None:
        nonlocal run_count
        if not can_run():
            record(group, name, "SKIPPED_MAX_RUNS")
            return None
        run_count += 1
        checkpoint = output_dir / "checkpoints" / f"{group}_{name.replace('/', '_').replace(' ', '_')}_seed{seed}.pt"
        prediction = output_dir / "predictions" / f"{group}_{name.replace('/', '_').replace(' ', '_')}_seed{seed}.jsonl"
        try:
            trained = run_training(text_cache, image_cache, manifest, checkpoint, steps=steps, seed=seed, device=device, batch_size=batch_size, graph_file=graph_file, **params)
            inferred = run_inference(checkpoint, text_cache, image_cache, manifest, prediction, device=device, graph_file=graph_file, batch_size=batch_size)
            result = {"group": group, "name": name, "status": "COMPLETED", "seed": seed, "checkpoint": str(checkpoint), "metrics": inferred.get("metrics", {}), "train": trained, "inference": inferred, "fidelity": "KGDMC_RECONSTRUCTED_COMPUTE_ADAPTED", "params": params}
            results.append(result)
            return result
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            record(group, name, "FAILED", seed=seed, error=str(exc), fidelity="KGDMC_RECONSTRUCTED_COMPUTE_ADAPTED", params=params)
            return None

    if "main" in groups:
        for seed in main_seeds:
            manifest = triplet_manifest("random", seed)
            result = train_and_infer("main", "KGDMC", seed, manifest)
            if result is not None and base_checkpoint is None:
                base_checkpoint = Path(result["checkpoint"]); base_manifest = manifest
            if result is not None:
                modality_manifest = output_dir / "manifests" / f"modality_random_seed{seed}.json"
                modality_rows = generate_modality_corruptions(entity_ids, text_vectors, image_vectors, seed=seed, strategy="random")
                save_modality_manifest(modality_manifest, modality_rows, {"dataset": dataset, "strategy": "random", "seed": seed, "fidelity": "DATA_RECONSTRUCTED_COMPUTE_ADAPTED"})
                modality_prediction = output_dir / "predictions" / f"main_modality_random_seed{seed}.jsonl"
                try:
                    modality_result = evaluate_modality_checkpoint(Path(result["checkpoint"]), text_cache, image_cache, modality_manifest, modality_prediction, device=device, batch_size=batch_size)
                    results.append({"group": "main_modality", "name": "KGDMC", "status": "COMPLETED", "seed": seed, **modality_result})
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    record("main_modality", "KGDMC", "FAILED", seed=seed, error=str(exc))

    if base_checkpoint is None and ("robustness" in groups or "baselines" in groups) and can_run():
        base_manifest = triplet_manifest("random", secondary_seed)
        result = train_and_infer("main", "KGDMC_base", secondary_seed, base_manifest)
        if result is not None:
            base_checkpoint = Path(result["checkpoint"])

    if "ablation" in groups:
        manifest = base_manifest or triplet_manifest("random", secondary_seed)
        modality_manifest = output_dir / "manifests" / f"modality_random_seed{secondary_seed}.json"
        modality_rows = generate_modality_corruptions(entity_ids, text_vectors, image_vectors, seed=secondary_seed, strategy="random")
        save_modality_manifest(modality_manifest, modality_rows, {"dataset": dataset, "strategy": "random", "seed": secondary_seed, "fidelity": "DATA_RECONSTRUCTED_COMPUTE_ADAPTED"})
        for name in DMR_ABLATIONS + MTC_ABLATIONS:
            result = train_and_infer("ablation", name, secondary_seed, manifest, ablation=name)
            if result is not None and name in DMR_ABLATIONS:
                try:
                    modality_result = evaluate_modality_checkpoint(Path(result["checkpoint"]), text_cache, image_cache, modality_manifest, output_dir / "predictions" / f"ablation_modality_{name.replace(' ', '_')}.jsonl", device=device, batch_size=batch_size)
                    results.append({"group": "ablation_modality", "name": name, "status": "COMPLETED", "seed": secondary_seed, **modality_result})
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    record("ablation_modality", name, "FAILED", error=str(exc))

    if "hyperparam" in groups:
        manifest = base_manifest or triplet_manifest("random", secondary_seed)
        defaults = {"alpha": 0.005, "beta": 0.01, "neighbor_limit": 120, "path_limit": 20, "lambda_structure": 0.5, "delta_path": 1.5}
        for parameter, values in HYPERPARAM_SWEEPS.items():
            for value in values:
                params = dict(defaults); params[parameter] = value
                result = train_and_infer("hyperparam", f"{parameter}_{value:g}", secondary_seed, manifest, **params)
                if result is not None and parameter in {"alpha", "beta"}:
                    modality_manifest = output_dir / "manifests" / f"modality_random_seed{secondary_seed}.json"
                    modality_rows = generate_modality_corruptions(entity_ids, text_vectors, image_vectors, seed=secondary_seed, strategy="random")
                    save_modality_manifest(modality_manifest, modality_rows, {"dataset": dataset, "strategy": "random", "seed": secondary_seed, "fidelity": "DATA_RECONSTRUCTED_COMPUTE_ADAPTED"})
                    try:
                        modality_result = evaluate_modality_checkpoint(Path(result["checkpoint"]), text_cache, image_cache, modality_manifest, output_dir / "predictions" / f"hyperparam_modality_{parameter}_{value:g}.jsonl", device=device, batch_size=batch_size)
                        results.append({"group": "hyperparam_modality", "name": f"{parameter}_{value:g}", "status": "COMPLETED", "seed": secondary_seed, **modality_result})
                    except (OSError, RuntimeError, ValueError, KeyError) as exc:
                        record("hyperparam_modality", f"{parameter}_{value:g}", "FAILED", error=str(exc))

    if "robustness" in groups:
        for strategy in TRIPLET_NOISE:
            manifest = triplet_manifest(strategy, secondary_seed)
            train_and_infer("robustness_triplet", strategy, secondary_seed, manifest)
        if base_checkpoint is not None:
            for strategy in MODALITY_NOISE:
                modality_manifest = output_dir / "manifests" / f"modality_{strategy}_seed{secondary_seed}.json"
                rows = generate_modality_corruptions(entity_ids, text_vectors, image_vectors, seed=secondary_seed, strategy=strategy)
                save_modality_manifest(modality_manifest, rows, {"dataset": dataset, "strategy": strategy, "seed": secondary_seed, "fidelity": "DATA_RECONSTRUCTED_COMPUTE_ADAPTED"})
                prediction = output_dir / "predictions" / f"robustness_modality_{strategy}.jsonl"
                try:
                    result = evaluate_modality_checkpoint(base_checkpoint, text_cache, image_cache, modality_manifest, prediction, device=device, batch_size=batch_size)
                    results.append({"group": "robustness_modality", "name": strategy, "status": "COMPLETED", **result})
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    record("robustness_modality", strategy, "FAILED", error=str(exc))

    if "baselines" in groups:
        manifest = base_manifest or triplet_manifest("random", secondary_seed)
        try:
            results.append({"group": "baseline_triplet", **run_baseline_triplet_matrix(manifest, text_cache, image_cache, output_dir / "baselines_triplet.jsonl")})
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            record("baseline_triplet", "all", "FAILED", error=str(exc))
        modality_manifest = output_dir / "manifests" / f"modality_random_seed{secondary_seed}.json"
        rows = generate_modality_corruptions(entity_ids, text_vectors, image_vectors, seed=secondary_seed, strategy="random")
        save_modality_manifest(modality_manifest, rows, {"dataset": dataset, "strategy": "random", "seed": secondary_seed, "fidelity": "DATA_RECONSTRUCTED_COMPUTE_ADAPTED"})
        try:
            results.append({"group": "baseline_modality", **run_baseline_modality_matrix(modality_manifest, text_cache, image_cache, output_dir / "baselines_modality.jsonl")})
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            record("baseline_modality", "all", "FAILED", error=str(exc))

    catalog = paper2_experiment_catalog()
    _write_json(output_dir / "experiment_plan.json", catalog)
    _write_json(output_dir / "results.json", results)
    coverage = {
        "dataset": dataset,
        "groups_requested": list(groups),
        "runs_attempted": run_count,
        "results": len(results),
        "fidelity": "DATA_RECONSTRUCTED_COMPUTE_ADAPTED",
        "official_baselines": "NOT_AVAILABLE_IN_REPOSITORY",
        "catalog": catalog,
    }
    _write_json(output_dir / "coverage.json", coverage)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "name", "status", "seed", "fidelity", "metrics", "error"])
        writer.writeheader()
        for item in results:
            writer.writerow({key: json.dumps(item.get(key), ensure_ascii=False, default=str) if isinstance(item.get(key), (dict, list)) else item.get(key, "") for key in writer.fieldnames})
    return coverage
