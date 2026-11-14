from __future__ import annotations

import json
from pathlib import Path

import torch

from .evaluation import evaluate_rankings
from .models import DMR, MTC
from .training import _graph_contexts, _load_cache, _load_corruptions


def run_inference(checkpoint: Path, text_cache: Path, image_cache: Path, corruption_manifest: Path, output: Path, device: str = "cpu", graph_file: Path | None = None, batch_size: int = 128) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    text_ids, text = _load_cache(text_cache)
    image_ids, visual = _load_cache(image_cache)
    image_by_id = {entity_id: i for i, entity_id in enumerate(image_ids)}
    common = [entity_id for entity_id in text_ids if entity_id in image_by_id]
    if not common:
        raise ValueError("No overlapping entity IDs between text and image feature caches")
    text_idx = {entity_id: i for i, entity_id in enumerate(text_ids)}
    text_batch = text[[text_idx[x] for x in common]].to(device)
    visual_batch = visual[[image_by_id[x] for x in common]].to(device)
    dmr = DMR(
        input_dim=text_batch.shape[1],
        alpha=float(config.get("alpha", 0.005)),
        beta=float(config.get("beta", 0.01)),
        ablation=config.get("ablation"),
    ).to(device)
    mtc = MTC(
        dim=256,
        relation_dim=256,
        lambda_structure=float(config.get("lambda_structure", 0.5)),
        delta_path=float(config.get("delta_path", 1.5)),
        ablation=config.get("ablation"),
    ).to(device)
    dmr.load_state_dict(state["dmr"])
    mtc.load_state_dict(state["mtc"])
    dmr.eval(); mtc.eval()
    entity_index = {entity_id: i for i, entity_id in enumerate(common)}
    relation_ids = state["relation_ids"]
    records = _load_corruptions(corruption_manifest)
    graph_triples = None
    if graph_file is not None:
        graph_triples = []
        for line in graph_file.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) == 3:
                graph_triples.append((fields[0], fields[1], fields[2]))
        if not graph_triples:
            raise ValueError(f"Graph file contains no triples: {graph_file}")
    rows = []
    scores, labels = [], []
    with torch.inference_mode():
        result = dmr(text_batch, visual_batch)
        covered = [item for item in records if all(x in entity_index for x in (item.original[0], item.original[2], item.corrupted[0], item.corrupted[2]))]
        entity_visual = result.raw_visual if config.get("ablation") == "w/o w" else result.visual
        entity_visual = result.text if config.get("ablation") == "w/o v_e" else entity_visual
        neighbor_context, path_context = _graph_contexts(
            covered,
            entity_index,
            entity_visual,
            graph_triples,
            int(config.get("neighbor_limit", 120)),
            int(config.get("path_limit", 20)),
        )
        for start in range(0, len(covered), batch_size):
            batch = covered[start : start + batch_size]
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
            positive_scores = mtc(heads, rel, tails, n_heads, p_heads)["total"]
            corrupted_scores = mtc(cheads, crel, ctails, n_cheads, p_cheads)["total"]
            for item, positive_score, corrupted_score in zip(batch, positive_scores.tolist(), corrupted_scores.tolist(), strict=True):
                rows.append({"original": item.original, "corrupted": item.corrupted, "original_score": positive_score, "corrupted_score": corrupted_score, "error_type": item.error_type})
                scores.extend([positive_score, corrupted_score])
                # High S_T is the paper's suspicious/error direction.
                labels.extend([0, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))
    return {"rows": len(rows), "predictions": str(output), "metrics": evaluate_rankings(scores, labels) if scores else {}, "status": "INFERENCE_REAL_FEATURES_COMPUTE_ADAPTED"}

