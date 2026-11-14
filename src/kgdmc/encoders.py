from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_entity_text(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                rows.append((parts[0], parts[1].strip()))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def encode_text_cached(
    text_file: Path,
    cache_root: Path,
    model_name: str = "bert-base-cased",
    batch_size: int = 4,
    limit: int | None = None,
    device: str = "auto",
) -> dict:
    """Encode real entity descriptions and reuse a cache keyed by input hash/settings."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    rows = load_entity_text(text_file, limit)
    data_hash = _file_sha256(text_file)
    key = hashlib.sha256(json.dumps({"model": model_name, "data": data_hash, "limit": limit, "batch": batch_size}, sort_keys=True).encode()).hexdigest()[:16]
    output = cache_root / model_name.replace("/", "--") / key
    output.mkdir(parents=True, exist_ok=True)
    tensor_file = output / "embeddings.pt"
    metadata_file = output / "metadata.json"
    if tensor_file.exists() and metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        metadata["cache_hit"] = True
        return metadata

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embeddings = []
    ids = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer([text for _, text in batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            embeddings.append(pooled.cpu())
            ids.extend(entity_id for entity_id, _ in batch)
    matrix = torch.cat(embeddings) if embeddings else torch.empty((0, model.config.hidden_size))
    torch.save({"entity_ids": ids, "embeddings": matrix}, tensor_file)
    metadata = {"cache_hit": False, "model": model_name, "device": device, "rows": len(rows), "hidden_size": matrix.shape[1], "data_sha256": data_hash, "cache": str(tensor_file)}
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def encode_images_cached(
    image_root: Path,
    cache_root: Path,
    model_name: str = "google/vit-base-patch16-224-in21k",
    batch_size: int = 4,
    limit: int | None = None,
    device: str = "auto",
    image_manifest: Path | None = None,
) -> dict:
    """Encode only locally supplied, decodable images; never downloads images."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    if image_manifest is None:
        raise ValueError("An explicit TSV image manifest is required: entity_id<TAB>image_path")
    rows: list[tuple[str, Path]] = []
    with image_manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError(f"Invalid image manifest row {line_number}; expected entity_id<TAB>image_path")
            entity_id, raw_path = parts[0].strip(), parts[1].strip()
            path = Path(raw_path)
            if not path.is_absolute():
                path = image_manifest.parent / path
            if path.parent != image_root and image_root not in path.parents:
                raise ValueError(f"Image path outside --images-from at row {line_number}: {path}")
            rows.append((entity_id, path))
    paths = [path for _, path in rows]
    if limit is not None:
        rows = rows[:limit]
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No approved local images found in {image_root}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Image manifest references missing files: {missing[:3]}")
    # Some large web archives contain technically decodable placeholder images
    # such as 1x1 JPEGs. PIL accepts them, but ViT's processor cannot turn
    # their degenerate arrays into a valid image batch. Keep only images with
    # a meaningful spatial extent; the manifest remains the audit source and
    # the skipped rows are reported in metadata.
    valid_rows: list[tuple[str, Path]] = []
    skipped_degenerate = 0
    skipped_unreadable = 0
    for entity_id, path in rows:
        try:
            with Image.open(path) as image:
                image.verify()
                if image.width < 2 or image.height < 2:
                    skipped_degenerate += 1
                    continue
            valid_rows.append((entity_id, path))
        except Exception:
            skipped_unreadable += 1
    rows = valid_rows
    paths = [path for _, path in rows]
    if not paths:
        raise ValueError("No usable images remain after removing degenerate/unreadable files")
    index_hash = hashlib.sha256("\n".join(f"{p}:{p.stat().st_size}" for p in paths).encode()).hexdigest()
    key = hashlib.sha256(json.dumps({"model": model_name, "images": index_hash, "batch": batch_size}, sort_keys=True).encode()).hexdigest()[:16]
    output = cache_root / model_name.replace("/", "--") / key
    output.mkdir(parents=True, exist_ok=True)
    tensor_file = output / "embeddings.pt"
    metadata_file = output / "metadata.json"
    if tensor_file.exists() and metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        metadata["cache_hit"] = True
        return metadata
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    entity_order: list[str] = []
    entity_sums: dict[str, torch.Tensor] = {}
    entity_counts: dict[str, int] = {}
    stored_paths: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            encoded = processor(images=images, return_tensors="pt").to(device)
            output_state = model(**encoded).last_hidden_state[:, 0]
            for (entity_id, path), vector in zip(rows[start : start + batch_size], output_state.cpu(), strict=True):
                if entity_id not in entity_sums:
                    entity_order.append(entity_id)
                    entity_sums[entity_id] = torch.zeros_like(vector)
                    entity_counts[entity_id] = 0
                entity_sums[entity_id] += vector
                entity_counts[entity_id] += 1
                stored_paths.append(str(path))
    entity_ids = entity_order
    entity_matrix = torch.stack([entity_sums[entity_id] / entity_counts[entity_id] for entity_id in entity_ids])
    torch.save({"entity_ids": entity_ids, "paths": stored_paths, "embeddings": entity_matrix}, tensor_file)
    metadata = {"cache_hit": False, "model": model_name, "device": device, "rows": len(entity_ids), "source_images": len(paths), "skipped_degenerate": skipped_degenerate, "skipped_unreadable": skipped_unreadable, "images_per_entity": max(entity_counts.values()), "hidden_size": entity_matrix.shape[1], "index_sha256": index_hash, "manifest": str(image_manifest), "cache": str(tensor_file)}
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata

