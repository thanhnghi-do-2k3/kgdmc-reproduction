from __future__ import annotations

import json
from pathlib import Path

from .data import verify_dataset


def write_readiness(root: Path, dataset: str) -> dict:
    result = verify_dataset(root, dataset, strict=True)
    image_root = root / "datasets" / "modality" / dataset
    image_count = sum(1 for p in image_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}) if image_root.exists() else 0
    corruption = sorted((root / "datasets" / "manifests" / dataset).glob("corruptions_*.json"))
    result.update({
        "real_image_files": image_count,
        "corruption_manifests": [str(p) for p in corruption],
        "image_status": "DATA_RECONSTRUCTED: candidate MMKB-derived images with explicit local entity manifest" if image_count else "BLOCKED_DATA: no approved entity-linked images",
        "ready_for_end_to_end_smoke": bool(result["ok"] and image_count > 0 and corruption),
        "ready_for_real_subset_eval": bool(result["ok"] and image_count > 0 and corruption),
        "ready_for_full_paper_eval": False,
        "blockers": [] if image_count > 0 else ["BLOCKED_DATA: approved entity-linked images are missing"],
    })
    report = root / "reports" / "data_readiness.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    result["report"] = str(report)
    return result

