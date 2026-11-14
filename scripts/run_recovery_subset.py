"""Recover the missing Paper-2 ablations and persist evaluation outputs to W&B.

This is intentionally explicit about fidelity.  The repository is a compute-adapted
KGDMC reconstruction; quick runs are diagnostics and must not be reported as paper
results.  Full runs use the supplied manifest and 200 training steps.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import wandb
import torch

from kgdmc.inference import run_inference
from kgdmc.corruption import generate_corruptions, read_triples, save_manifest
from kgdmc.experiment_suite import _feature_maps
from kgdmc.training import run_training


MISSING_ABLATIONS = ("w/o S_GS", "w/o S_GP", "w/o C_T")
ROBUSTNESS_STRATEGIES = ("random", "similar", "adversarial")
REPLAY_MAIN_SEEDS = (42, 43, 44, 45, 46)
REPLAY_ABLATIONS = (
    "w/o C_M",
    "w/o L_DA",
    "w/o L_InterDD",
    "w/o L_IntraDD",
    "w/o S_LT",
    "w/o v_e",
    "w/o w",
    "w/o z_ti",
)
HYPERPARAM_SWEEPS = {
    "alpha": (5e-4, 1e-3, 5e-3, 1e-2, 5e-2),
    "beta": (1e-3, 5e-3, 1e-2, 5e-2, 1e-1),
    "neighbor_limit": (30, 60, 90, 120, 150),
    "path_limit": (5, 10, 15, 20, 25),
    "lambda_structure": (0.0, 0.5, 1.0, 1.5, 2.0),
    "delta_path": (0.0, 0.5, 1.0, 1.5, 2.0),
}


def cache_entity_ids(path: Path) -> set[str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return set(payload["entity_ids"])


def make_subset(source: Path, destination: Path, limit: int, allowed_entities: set[str]) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = [
        record
        for record in payload.get("records", [])
        if all(entity in allowed_entities for entity in (*record["original"][0:1], record["original"][2], record["corrupted"][0], record["corrupted"][2]))
    ]
    if not records:
        raise SystemExit(
            "No corruption records are covered by both feature caches. "
            "Use the matching full text/image caches and manifest."
        )
    if limit > 0:
        records = records[:limit]
    subset = {
        "metadata": {
            **payload.get("metadata", {}),
            "subset_of": str(source),
            "subset_records": limit,
            "fidelity": "COMPUTE_ADAPTED_QUICK_DIAGNOSTIC",
        },
        "records": records[:limit],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(subset, indent=2), encoding="utf-8")
    return destination


def flatten_metrics(metrics: dict) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    for percent, values in metrics.items():
        for metric, value in values.items():
            if metric == "k":
                flattened[f"eval/{metric}_{percent}"] = int(value)
            else:
                flattened[f"eval/{metric}_{percent}"] = float(value)
    return flattened


def upload_evaluation(
    *,
    name: str,
    config: dict,
    metrics: dict,
    prediction: Path,
    checkpoint: Path,
) -> None:
    mode = os.getenv("WANDB_MODE", "disabled")
    if mode == "disabled":
        print("W&B disabled; evaluation was saved locally only:", prediction)
        return
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "mmkgc-research"),
        entity=os.getenv("WANDB_ENTITY") or None,
        group=os.getenv("WANDB_RUN_GROUP") or "paper2-recovery-evaluation",
        job_type="evaluation",
        name=name,
        config={**config, "fidelity": "KGDMC_RECONSTRUCTED_COMPUTE_ADAPTED"},
        mode=mode,
        reinit="create_new",
    )
    run.log(flatten_metrics(metrics))
    run.summary.update(flatten_metrics(metrics))
    artifact = wandb.Artifact(f"{name}-outputs", type="evaluation")
    artifact.add_file(str(prediction), name="predictions.jsonl")
    artifact.add_file(str(checkpoint), name="checkpoint.pt")
    run.log_artifact(artifact)
    run.finish()
    print("W&B evaluation run:", run.url)


def execute_one(args: argparse.Namespace, label: str, ablation: str | None, steps: int, manifest: Path, tag: str, **params: object) -> dict:
    safe = label.replace("/", "_").replace(" ", "_")
    root = Path(args.output_dir) / tag / safe
    checkpoint = root / f"{safe}_seed{args.seed}.pt"
    prediction = root / "predictions.jsonl"
    root.mkdir(parents=True, exist_ok=True)

    train = run_training(
        args.text_cache,
        args.image_cache,
        manifest,
        checkpoint,
        steps=steps,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        graph_file=args.graph_file,
        ablation=ablation,
        **params,
        select_best=args.select_best,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    inference = run_inference(
        checkpoint,
        args.text_cache,
        args.image_cache,
        manifest,
        prediction,
        device=args.device,
        graph_file=args.graph_file,
        batch_size=args.batch_size,
    )
    metrics = inference.get("metrics", {})
    upload_evaluation(
        name=f"eval-{tag}-{safe}-seed{args.seed}",
        config={
            "label": label,
            "ablation": ablation,
            "steps": steps,
            "records": inference.get("rows", 0),
            "manifest": str(manifest),
            "seed": args.seed,
            "select_best": args.select_best,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            **params,
        },
        metrics=metrics,
        prediction=prediction,
        checkpoint=checkpoint,
    )
    result = {"label": label, "ablation": ablation, "tag": tag, "train": train, "inference": inference}
    (root / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"label": label, "tag": tag, "steps": steps, "records": inference.get("rows"), "metrics": metrics}, indent=2))
    return result


def make_robustness_manifest(args: argparse.Namespace, strategy: str) -> Path:
    entity_ids, _, image_vectors = _feature_maps(args.text_cache, args.image_cache)
    triples = read_triples(args.graph_file)
    relation_pool = sorted({triple[1] for triple in triples})
    records = generate_corruptions(
        triples,
        fraction=0.05,
        seed=args.seed,
        entity_pool=entity_ids,
        relation_pool=relation_pool,
        strategy=strategy,
        entity_vectors=image_vectors,
    )
    path = args.output_dir / "manifests" / f"triplet_{strategy}_seed{args.seed}.json"
    save_manifest(
        path,
        records,
        {
            "dataset": "fb15k237",
            "strategy": strategy,
            "seed": args.seed,
            "fidelity": "KGDMC_RECONSTRUCTED_COMPUTE_ADAPTED",
        },
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full", "missing", "replay", "hyperparam", "all"), default="quick")
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/paper2_recovery"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick-steps", type=int, default=10)
    parser.add_argument("--quick-records", type=int, default=1000)
    parser.add_argument("--full-steps", type=int, default=200)
    parser.add_argument("--select-best", action="store_true", help="Keep the checkpoint with the lowest training loss.")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop after this many non-improving steps; 0 disables early stopping.")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum loss improvement required to reset patience.")
    args = parser.parse_args()

    for path in (args.text_cache, args.image_cache, args.manifest, args.graph_file):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable. Check the 5060 Ti PyTorch CUDA install before starting a long run.")

    allowed_entities = cache_entity_ids(args.text_cache) & cache_entity_ids(args.image_cache)
    quick_manifest = make_subset(args.manifest, args.output_dir / "manifests" / "quick.json", args.quick_records, allowed_entities)
    results = []
    if args.mode in {"quick", "all"}:
        results.append(execute_one(args, "main_quick", None, args.quick_steps, quick_manifest, "quick"))
    if args.mode in {"full", "all", "missing"}:
        for ablation in MISSING_ABLATIONS:
            results.append(execute_one(args, f"ablation_{ablation}", ablation, args.full_steps, args.manifest, "full"))
    if args.mode in {"missing", "all"}:
        for strategy in ROBUSTNESS_STRATEGIES:
            manifest = make_robustness_manifest(args, strategy)
            results.append(execute_one(args, f"robustness_triplet_{strategy}", None, args.full_steps, manifest, "robustness"))
    if args.mode == "replay":
        # Re-run the previously completed main seeds and ablations on the small,
        # explicitly labelled subset. This produces genuine subset metrics, not
        # fabricated estimates of the lost full-corpus evaluation.
        for seed in REPLAY_MAIN_SEEDS:
            run_args = argparse.Namespace(**vars(args))
            run_args.seed = seed
            results.append(execute_one(run_args, f"main_seed{seed}", None, args.full_steps, args.manifest, "subset_replay"))
        for ablation in REPLAY_ABLATIONS:
            results.append(execute_one(args, f"ablation_{ablation}_seed42", ablation, args.full_steps, args.manifest, "subset_replay"))
    if args.mode == "hyperparam":
        defaults = {"alpha": 0.005, "beta": 0.01, "neighbor_limit": 120, "path_limit": 20, "lambda_structure": 0.5, "delta_path": 1.5}
        for parameter, values in HYPERPARAM_SWEEPS.items():
            for value in values:
                params = dict(defaults)
                params[parameter] = value
                label = f"hyperparam_{parameter}_{value:g}"
                results.append(execute_one(args, label, None, args.full_steps, args.manifest, "subset_hyperparam", **params))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "recovery_summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Saved recovery summary to {args.output_dir / 'recovery_summary.json'}")


if __name__ == "__main__":
    main()
