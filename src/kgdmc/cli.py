from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import click

from .data import DATASETS, fetch_dataset, inspect_dataset, verify_dataset
from .corruption import generate_corruptions, read_triples, save_manifest
from .encoders import encode_images_cached, encode_text_cached
from .readiness import write_readiness
from .training import run_training
from .inference import run_inference
from .experiment_suite import run_experiment_suite

ROOT = Path(__file__).resolve().parents[2]


@click.group()
def main() -> None:
    """KGDMC reproducibility CLI."""


@main.group()
def env() -> None:
    """Inspect the local runtime."""


@env.command("doctor")
@click.option("--device", default="auto")
def doctor(device: str) -> None:
    info = {"python": sys.version, "platform": platform.platform(), "requested_device": device}
    try:
        import torch  # type: ignore
        info.update({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()})
    except (ImportError, AttributeError):
        info["torch"] = "NOT_INSTALLED"
    click.echo(json.dumps(info, indent=2, default=str))


@main.group()
def data() -> None:
    """Download and validate real data."""


@data.command("fetch")
@click.option("--dataset", type=click.Choice(sorted(DATASETS)), required=True)
@click.option("--components", default="graph,text", help="Comma-separated: graph,text")
@click.option("--resume/--no-resume", default=True)
def fetch(dataset: str, components: str, resume: bool) -> None:
    result = fetch_dataset(ROOT, dataset, {x.strip() for x in components.split(",")}, resume)
    click.echo(json.dumps(result, indent=2))


@data.command("verify")
@click.option("--dataset", type=click.Choice(sorted(DATASETS)), required=True)
@click.option("--strict/--no-strict", default=False)
def verify(dataset: str, strict: bool) -> None:
    result = verify_dataset(ROOT, dataset, strict)
    click.echo(json.dumps(result, indent=2))
    if not result["ok"]:
        raise click.exceptions.Exit(1)


@data.command("inspect")
@click.option("--dataset", type=click.Choice(sorted(DATASETS)), required=True)
def inspect(dataset: str) -> None:
    click.echo(json.dumps(inspect_dataset(ROOT, dataset), indent=2))


@data.command("encode-text")
@click.option("--dataset", type=click.Choice(sorted(DATASETS)), required=True)
@click.option("--model", "model_name", default="bert-base-cased")
@click.option("--batch-size", default=4, type=int)
@click.option("--limit", default=None, type=int, help="Optional real-record limit for smoke testing")
@click.option("--device", default="auto")
def encode_text(dataset: str, model_name: str, batch_size: int, limit: int | None, device: str) -> None:
    text_file = ROOT / "datasets" / "raw" / dataset / "entity2text.txt"
    if not text_file.exists():
        raise click.ClickException(f"Missing real text file: {text_file}; run data fetch first")
    result = encode_text_cached(text_file, ROOT / "datasets" / "feature_cache" / dataset, model_name, batch_size, limit, device)
    click.echo(json.dumps(result, indent=2))


@data.command("encode-images")
@click.option("--dataset", type=click.Choice(sorted(DATASETS)), required=True)
@click.option("--images-from", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--image-manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="TSV: entity_id<TAB>image_path")
@click.option("--model", "model_name", default="google/vit-base-patch16-224-in21k")
@click.option("--batch-size", default=4, type=int)
@click.option("--limit", default=None, type=int)
@click.option("--device", default="auto")
def encode_images(dataset: str, images_from: Path, image_manifest: Path, model_name: str, batch_size: int, limit: int | None, device: str) -> None:
    try:
        result = encode_images_cached(images_from, ROOT / "datasets" / "feature_cache" / dataset, model_name, batch_size, limit, device, image_manifest)
    except ValueError as exc:
        raise click.ClickException(f"BLOCKED_DATA: {exc}") from exc
    click.echo(json.dumps(result, indent=2))


@data.command("corrupt")
@click.option("--dataset", type=click.Choice(sorted(DATASETS)), required=True)
@click.option("--split", default="train", type=click.Choice(["train", "valid", "test"]))
@click.option("--triples-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Optional real subgraph triples file")
@click.option("--entity-ids-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Optional allowed entity IDs, one per line")
@click.option("--fraction", default=0.05, type=float)
@click.option("--seed", default=42, type=int)
@click.option("--strategy", type=click.Choice(["random", "similar", "adversarial"]), default="random")
@click.option("--output", type=click.Path(path_type=Path), default=None)
def corrupt(dataset: str, split: str, triples_file: Path | None, entity_ids_file: Path | None, fraction: float, seed: int, strategy: str, output: Path | None) -> None:
    source = triples_file or ROOT / "datasets" / "raw" / dataset / f"{split}.txt"
    if not source.exists():
        raise click.ClickException(f"Missing real graph split: {source}; run data fetch first")
    relation_source = ROOT / "datasets" / "raw" / dataset / f"{split}.txt"
    relation_pool = sorted({triple[1] for triple in read_triples(relation_source)})
    entity_pool = [line.strip() for line in entity_ids_file.read_text().splitlines() if line.strip()] if entity_ids_file else None
    records = generate_corruptions(read_triples(source), fraction, seed, entity_pool=entity_pool, relation_pool=relation_pool, strategy=strategy)
    output_path = output or ROOT / "datasets" / "manifests" / dataset / f"corruptions_{split}_{strategy}_seed{seed}.json"
    result = save_manifest(output_path, records, {"dataset": dataset, "split": split, "input": str(source), "fraction": fraction, "seed": seed, "strategy": strategy, "status": "COMPUTE_ADAPTED_NOISE_GENERATOR" if strategy != "random" else "GENERIC_REAL_GRAPH_CORRUPTION_PAPER_AUDIT_PENDING"})
    click.echo(json.dumps(result, indent=2))


@data.command("readiness")
@click.option("--dataset", type=click.Choice(sorted(DATASETS)), required=True)
def readiness(dataset: str) -> None:
    result = write_readiness(ROOT, dataset)
    click.echo(json.dumps(result, indent=2))


@main.command("status")
@click.option("--verbose", is_flag=True)
def status(verbose: bool) -> None:
    click.echo(json.dumps({"version": "0.1.0", "root": str(ROOT), "verbose": verbose, "full_pipeline": "PARTIAL: DMR/MTC training and inference require approved aligned image cache"}, indent=2))


@main.group()
def experiments() -> None:
    """Experiment lifecycle commands."""


@experiments.command("run")
@click.option("--dataset", required=True)
@click.option("--profile", required=True)
@click.option("--text-cache", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option("--image-cache", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option("--corruption-manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--steps", default=2, type=int)
@click.option("--batch-size", default=128, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--device", default="auto")
@click.option("--resume", "resume_from", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--graph-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--ablation", default=None)
@click.option("--alpha", default=0.005, type=float)
@click.option("--beta", default=0.01, type=float)
@click.option("--lambda-structure", default=0.5, type=float)
@click.option("--delta-path", default=1.5, type=float)
@click.option("--neighbor-limit", default=120, type=int)
@click.option("--path-limit", default=20, type=int)
def run(dataset: str, profile: str, text_cache: Path | None, image_cache: Path | None, corruption_manifest: Path | None, output: Path | None, steps: int, batch_size: int, seed: int, device: str, resume_from: Path | None, graph_file: Path | None, ablation: str | None, alpha: float, beta: float, lambda_structure: float, delta_path: float, neighbor_limit: int, path_limit: int) -> None:
    if not text_cache or not image_cache or not corruption_manifest:
        raise click.ClickException("BLOCKED_DATA: provide text cache, image cache, and corruption manifest; no visual placeholder is allowed")
    try:
        result = run_training(text_cache, image_cache, corruption_manifest, output or ROOT / "runs" / f"{dataset}_{profile}_seed{seed}.pt", steps=steps, seed=seed, device=device, resume_from=resume_from, batch_size=batch_size, graph_file=graph_file, ablation=ablation, alpha=alpha, beta=beta, lambda_structure=lambda_structure, delta_path=delta_path, neighbor_limit=neighbor_limit, path_limit=path_limit)
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2))


@experiments.command("infer")
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--text-cache", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--image-cache", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--corruption-manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--device", default="cpu")
@click.option("--batch-size", default=128, type=int)
@click.option("--graph-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
def infer(checkpoint: Path, text_cache: Path, image_cache: Path, corruption_manifest: Path, output: Path, device: str, batch_size: int, graph_file: Path | None) -> None:
    try:
        result = run_inference(checkpoint, text_cache, image_cache, corruption_manifest, output, device, graph_file, batch_size)
    except (ValueError, RuntimeError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2))


@experiments.command("suite")
@click.option("--dataset", type=click.Choice(sorted(DATASETS)), required=True)
@click.option("--text-cache", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--image-cache", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--graph-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False)
@click.option("--groups", default="main,ablation,hyperparam,robustness,baselines", help="Comma-separated: main,ablation,hyperparam,robustness,baselines")
@click.option("--main-seeds", default="42,43,44,45,46", help="Five paper-style seeds, comma-separated")
@click.option("--secondary-seed", default=42, type=int)
@click.option("--steps", default=30, type=int)
@click.option("--batch-size", default=128, type=int)
@click.option("--device", default="auto")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--max-runs", type=int, default=None, help="Optional bounded run count for a measured pilot")
def suite(dataset: str, text_cache: Path, image_cache: Path, graph_file: Path | None, groups: str, main_seeds: str, secondary_seed: int, steps: int, batch_size: int, device: str, output_dir: Path | None, max_runs: int | None) -> None:
    graph_file = graph_file or ROOT / "datasets" / "raw" / dataset / "train.txt"
    if not graph_file.exists():
        raise click.ClickException(f"Missing graph file: {graph_file}")
    try:
        result = run_experiment_suite(
            ROOT,
            dataset,
            text_cache,
            image_cache,
            graph_file,
            groups=tuple(item.strip() for item in groups.split(",") if item.strip()),
            main_seeds=tuple(int(item.strip()) for item in main_seeds.split(",") if item.strip()),
            secondary_seed=secondary_seed,
            steps=steps,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            max_runs=max_runs,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

