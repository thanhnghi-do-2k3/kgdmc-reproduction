from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from .corruption import Corruption
from .evaluation import evaluate_rankings
from .models import DMR, MTC
from .wandb_logging import finish as finish_wandb
from .wandb_logging import log as log_wandb
from .wandb_logging import start_run


def _load_cache(path: Path) -> tuple[list[str], Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload["entity_ids"]), payload["embeddings"].float()


def _load_corruptions(path: Path) -> list[Corruption]:
    payload = json.loads(path.read_text())
    return [Corruption(**record) for record in payload["records"]]


def _graph_contexts(
    items: list[Corruption],
    entity_index: dict[str, int],
    entity_vectors: Tensor,
    graph_triples: Iterable[tuple[str, str, str]] | None = None,
    neighbor_limit: int = 120,
    path_limit: int = 20,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Build deterministic one-hop and two-hop contexts from positive graph triples.

    Full runs should pass the complete training graph.  The corruption records remain
    the fallback for the small legacy smoke path.
    """
    adjacency: dict[str, list[str]] = {entity_id: [] for entity_id in entity_index}
    edges = graph_triples if graph_triples is not None else (item.original for item in items)
    for h, _, t in edges:
        if h in adjacency and t in adjacency:
            adjacency[h].append(t)
            adjacency[t].append(h)
    neighbor: dict[str, Tensor] = {}
    path: dict[str, Tensor] = {}
    fallback = entity_vectors.mean(0)
    for entity_id, neighbors in adjacency.items():
        valid = sorted({n for n in neighbors if n in entity_index})[:neighbor_limit]
        neighbor[entity_id] = entity_vectors[[entity_index[n] for n in valid]].mean(0) if valid else fallback
        two_hop = []
        for first in valid:
            two_hop.extend(n for n in adjacency.get(first, []) if n != entity_id and n in entity_index)
        path_nodes = sorted(set(two_hop))[:path_limit]
        path[entity_id] = entity_vectors[[entity_index[n] for n in path_nodes]].mean(0) if path_nodes else neighbor[entity_id]
    return neighbor, path


def run_training(
    text_cache: Path,
    image_cache: Path,
    corruption_manifest: Path,
    output: Path,
    steps: int = 2,
    seed: int = 42,
    device: str = "cpu",
    resume_from: Path | None = None,
    batch_size: int = 128,
    graph_file: Path | None = None,
    ablation: str | None = None,
    alpha: float = 0.005,
    beta: float = 0.01,
    lambda_structure: float = 0.5,
    delta_path: float = 1.5,
    neighbor_limit: int = 120,
    path_limit: int = 20,
    select_best: bool = False,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
) -> dict:
    """Run a compact real-feature DMR→MTC training loop and persist a checkpoint.

    The training order and exact paper coefficients remain audit-pending. This function
    deliberately refuses to substitute missing visual features with zeros or random data.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if early_stop_patience < 0:
        raise ValueError("early_stop_patience must be non-negative")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    text_ids, text = _load_cache(text_cache)
    image_ids, visual = _load_cache(image_cache)
    image_by_id = {entity_id: i for i, entity_id in enumerate(image_ids)}
    common = [entity_id for entity_id in text_ids if entity_id in image_by_id]
    if not common:
        raise ValueError("No overlapping entity IDs between text and image feature caches")
    text_idx = {entity_id: i for i, entity_id in enumerate(text_ids)}
    text_batch = text[[text_idx[x] for x in common]].to(device)
    visual_batch = visual[[image_by_id[x] for x in common]].to(device)
    dmr = DMR(input_dim=text_batch.shape[1], alpha=alpha, beta=beta, ablation=ablation).to(device)
    mtc = MTC(dim=256, relation_dim=256, lambda_structure=lambda_structure, delta_path=delta_path, ablation=ablation).to(device)
    optimizer = torch.optim.Adam(list(dmr.parameters()) + list(mtc.parameters()), lr=1e-4)
    completed_steps = 0
    if resume_from is not None:
        state = torch.load(resume_from, map_location="cpu", weights_only=False)
        saved_config = state.get("config", {})
        if ablation is None:
            ablation = saved_config.get("ablation")
        alpha = float(saved_config.get("alpha", alpha))
        beta = float(saved_config.get("beta", beta))
        lambda_structure = float(saved_config.get("lambda_structure", lambda_structure))
        delta_path = float(saved_config.get("delta_path", delta_path))
        neighbor_limit = int(saved_config.get("neighbor_limit", neighbor_limit))
        path_limit = int(saved_config.get("path_limit", path_limit))
        dmr.alpha = alpha
        dmr.beta = beta
        dmr.ablation = ablation
        mtc.lambda_structure = lambda_structure
        mtc.delta_path = delta_path
        mtc.ablation = ablation
        dmr.load_state_dict(state["dmr"])
        mtc.load_state_dict(state["mtc"])
        optimizer.load_state_dict(state["optimizer"])
        completed_steps = int(state.get("completed_steps", state.get("steps", 0)))
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
    entity_index = {entity_id: i for i, entity_id in enumerate(common)}
    relation_ids: dict[str, int] = {}
    corruptions = _load_corruptions(corruption_manifest)
    used = [item for item in corruptions if item.original[0] in entity_index and item.original[2] in entity_index and item.corrupted[0] in entity_index and item.corrupted[2] in entity_index]
    if not used:
        raise ValueError("Corruption manifest has no records fully covered by the text/image cache intersection")
    for item in used:
        relation_ids.setdefault(item.original[1], len(relation_ids))
        relation_ids.setdefault(item.corrupted[1], len(relation_ids))
    graph_triples = None
    if graph_file is not None:
        graph_triples = []
        for line in graph_file.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) == 3:
                graph_triples.append((fields[0], fields[1], fields[2]))
        if not graph_triples:
            raise ValueError(f"Graph file contains no triples: {graph_file}")
    experiment_config = {
        "seed": seed,
        "steps": steps,
        "device": device,
        "batch_size": batch_size,
        "records": len(used),
        "entities": len(common),
        "graph_file": str(graph_file) if graph_file else None,
        "ablation": ablation,
        "alpha": alpha,
        "beta": beta,
        "lambda_structure": lambda_structure,
        "delta_path": delta_path,
        "neighbor_limit": neighbor_limit,
        "path_limit": path_limit,
        "select_best": select_best,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
    }
    wandb_run = start_run(
        name=f"{output.stem}-seed{seed}",
        config=experiment_config,
    )
    last = None
    best_loss = float("inf")
    best_step = 0
    best_dmr_state = None
    best_mtc_state = None
    no_improvement = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        dmr_result = dmr(text_batch, visual_batch)
        entity_visual = dmr_result.raw_visual if ablation == "w/o w" else dmr_result.visual
        entity_visual = dmr_result.text if ablation == "w/o v_e" else entity_visual
        neighbor_context, path_context = _graph_contexts(used, entity_index, entity_visual, graph_triples, neighbor_limit, path_limit)
        ranking_terms = []
        for start in range(0, len(used), batch_size):
            batch = used[start : start + batch_size]
            h_idx = torch.tensor([entity_index[item.original[0]] for item in batch], device=device)
            t_idx = torch.tensor([entity_index[item.original[2]] for item in batch], device=device)
            ch_idx = torch.tensor([entity_index[item.corrupted[0]] for item in batch], device=device)
            ct_idx = torch.tensor([entity_index[item.corrupted[2]] for item in batch], device=device)
            rel = torch.tensor([relation_ids[item.original[1]] for item in batch], device=device)
            crel = torch.tensor([relation_ids[item.corrupted[1]] for item in batch], device=device)
            heads = entity_visual[h_idx]
            tails = entity_visual[t_idx]
            cheads = entity_visual[ch_idx]
            ctails = entity_visual[ct_idx]
            n_heads = torch.stack([neighbor_context[item.original[0]] for item in batch]).to(device)
            p_heads = torch.stack([path_context[item.original[0]] for item in batch]).to(device)
            n_cheads = torch.stack([neighbor_context.get(item.corrupted[0], neighbor_context[item.original[0]]) for item in batch]).to(device)
            p_cheads = torch.stack([path_context.get(item.corrupted[0], path_context[item.original[0]]) for item in batch]).to(device)
            positive_output = mtc(heads, rel, tails, n_heads, p_heads)
            negative_output = mtc(cheads, crel, ctails, n_cheads, p_cheads)
            positive = positive_output["total"]
            negative = negative_output["total"]
            # Paper Eq. 11 treats the clean/original triplet as the positive
            # reference and the generated corruption as the negative/error. Since
            # higher S_T is the suspicious direction, the clean score must be lower.
            confidence = None if ablation == "w/o C_T" else torch.sigmoid(-positive.detach())
            ranking_terms.append(mtc.ranking_loss(positive, negative, confidence=confidence))
        ranking = torch.stack(ranking_terms).mean()
        loss = dmr_result.loss + ranking
        loss.backward()
        optimizer.step()
        completed_steps += 1
        last = {"loss": float(loss.detach().cpu()), "dmr_loss": float(dmr_result.loss.detach().cpu()), "ranking_loss": float(ranking.detach().cpu()), "records": len(used), "entities": len(common), "batch_size": batch_size, "device": device, "completed_steps": completed_steps}
        if select_best and last["loss"] < best_loss - early_stop_min_delta:
            best_loss = last["loss"]
            best_step = completed_steps
            best_dmr_state = {key: value.detach().cpu().clone() for key, value in dmr.state_dict().items()}
            best_mtc_state = {key: value.detach().cpu().clone() for key, value in mtc.state_dict().items()}
            no_improvement = 0
        elif select_best and early_stop_patience > 0:
            no_improvement += 1
        log_wandb(
            wandb_run,
            {"train/loss": last["loss"], "train/dmr_loss": last["dmr_loss"], "train/ranking_loss": last["ranking_loss"]},
            step,
        )
        if select_best and early_stop_patience > 0 and no_improvement >= early_stop_patience:
            break
    if select_best and best_dmr_state is not None and best_mtc_state is not None:
        dmr.load_state_dict(best_dmr_state)
        mtc.load_state_dict(best_mtc_state)
        selected_step = best_step
    else:
        selected_step = completed_steps
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dmr": dmr.state_dict(), "mtc": mtc.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed, "steps": steps, "completed_steps": completed_steps, "selected_step": selected_step, "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(), "last": last, "entity_ids": common, "relation_ids": relation_ids, "config": experiment_config}, output)
    result = {**(last or {}), "checkpoint": str(output), "config": experiment_config, "selected_step": selected_step, "best_loss": best_loss if select_best else None, "early_stopped": completed_steps < steps, "status": "TRAINED_REAL_FEATURES_COMPUTE_ADAPTED"}
    finish_wandb(wandb_run, last)
    return result

