from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Corruption:
    source_split: str
    row_index: int
    original: tuple[str, str, str]
    corrupted: tuple[str, str, str]
    error_type: str
    label: int


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"Expected 3 tab-separated fields in {path}: {line[:80]!r}")
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def generate_corruptions(
    triples: list[tuple[str, str, str]],
    fraction: float = 0.05,
    seed: int = 42,
    entity_pool: list[str] | None = None,
    relation_pool: list[str] | None = None,
    strategy: str = "random",
    entity_vectors: Mapping[str, Sequence[float]] | None = None,
    candidate_limit: int = 256,
) -> list[Corruption]:
    """Generate real-data negative triples without self-replacement or true-triple collisions.

    ``random`` is the default paper-compatible noise family. ``similar`` and
    ``adversarial`` are deterministic, compute-adapted candidates used by the paper's
    robustness section when an explicit author-side candidate generator is unavailable:
    they prefer cosine-near entities, with ``adversarial`` additionally restricting to
    entities observed in the same relation position where possible.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if strategy not in {"random", "similar", "adversarial"}:
        raise ValueError("strategy must be random, similar, or adversarial")
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    rng = random.Random(seed)
    entities = sorted(entity_pool or {x for t in triples for x in (t[0], t[2])})
    relations = sorted(relation_pool or {t[1] for t in triples})
    truth = set(triples)
    relation_heads: dict[str, set[str]] = {}
    relation_tails: dict[str, set[str]] = {}
    for h, r, t in triples:
        relation_heads.setdefault(r, set()).add(h)
        relation_tails.setdefault(r, set()).add(t)

    normalized_vectors: dict[str, torch.Tensor] = {}
    if entity_vectors:
        normalized_vectors = {
            entity_id: F.normalize(vector.reshape(1, -1).float(), dim=1).squeeze(0)
            for entity_id, vector in entity_vectors.items()
            if isinstance(vector, torch.Tensor)
        }

    def cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item())
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        return dot / max(1e-12, left_norm * right_norm)

    def ordered_candidates(pool: list[str], reference: str, compatible: set[str] | None = None) -> list[str]:
        candidates = [candidate for candidate in pool if candidate != reference]
        if strategy == "adversarial" and compatible:
            compatible_candidates = [candidate for candidate in candidates if candidate in compatible]
            if compatible_candidates:
                candidates = compatible_candidates
        if strategy == "random" or not entity_vectors or reference not in entity_vectors:
            rng.shuffle(candidates)
            return candidates
        if len(candidates) > candidate_limit:
            candidates = rng.sample(candidates, candidate_limit)
        reference_vector = normalized_vectors.get(reference)
        if reference_vector is None:
            return candidates
        candidate_vectors = torch.stack([normalized_vectors.get(candidate, reference_vector) for candidate in candidates])
        order = torch.argsort(candidate_vectors @ reference_vector, descending=True).tolist()
        return [candidates[index] for index in order]
    n = max(1, round(len(triples) * fraction))
    selected = list(range(len(triples)))
    rng.shuffle(selected)
    selected = selected[:n]
    out: list[Corruption] = []
    for row_index in selected:
        h, r, t = triples[row_index]
        for error_type in ("head", "relation", "tail"):
            candidate = None
            pool = entities if error_type != "relation" else relations
            compatible = None
            if error_type == "head":
                compatible = relation_heads.get(r)
            elif error_type == "tail":
                compatible = relation_tails.get(r)
            for replacement in ordered_candidates(pool, h if error_type == "head" else t, compatible):
                if error_type == "head":
                    proposed = (replacement, r, t)
                elif error_type == "relation":
                    proposed = (h, replacement, t)
                else:
                    proposed = (h, r, replacement)
                if proposed != (h, r, t) and proposed not in truth:
                    candidate = proposed
                    break
            if candidate is None and strategy != "random" and error_type != "relation":
                # Similar/adversarial candidate subsampling can miss the only valid
                # replacement in a dense small slice. Fall back to a complete valid
                # pool so the manifest remains executable and collision-free.
                fallback = [candidate for candidate in pool if candidate != (h if error_type == "head" else t)]
                rng.shuffle(fallback)
                for replacement in fallback:
                    proposed = (replacement, r, t) if error_type == "head" else (h, r, replacement)
                    if proposed not in truth:
                        candidate = proposed
                        break
            if candidate is None:
                # A one-relation graph cannot produce a valid relation corruption.
                # Keep the real head/tail corruptions instead of inventing a relation.
                if error_type == "relation":
                    continue
                raise RuntimeError(f"No valid {error_type} candidate for row {row_index}")
            label = f"{strategy}_{error_type}" if strategy != "random" else error_type
            out.append(Corruption("train", row_index, (h, r, t), candidate, label, 1))
    return out


def save_manifest(path: Path, corruptions: list[Corruption], metadata: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "records": [asdict(item) for item in corruptions]}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode()
    path.write_bytes(encoded)
    return {"path": str(path), "records": len(corruptions), "sha256": hashlib.sha256(encoded).hexdigest()}

