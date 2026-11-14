#!/usr/bin/env python3
"""Import and validate the full FB15K image archive.

This deliberately accepts a locally acquired archive only.  It does not
download images, invent missing images, or silently fall back to the subset.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image


ENTITY_RE = re.compile(r"(?:^|/)(m\.[^/]+|m/[^/]+)(?:/|$)")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def canonical_entity(member: str) -> str | None:
    parts = [p for p in member.replace("\\", "/").split("/") if p]
    for index, part in enumerate(parts):
        if part.startswith("m.") and len(part) > 2:
            return "/m/" + part[2:]
        if part == "m" and index + 1 < len(parts) and parts[index + 1]:
            return "/m/" + parts[index + 1]
    return None


def safe_member(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def graph_entities(root: Path) -> set[str]:
    result: set[str] = set()
    for split in ("train.txt", "valid.txt", "test.txt"):
        for line in (root / split).read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) == 3:
                result.update((fields[0], fields[2]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/raw/fb15k237"))
    parser.add_argument("--output", type=Path, default=Path("datasets/modality/fb15k237/full"))
    parser.add_argument("--images-per-entity", type=int, default=10)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    if not args.archive.is_file():
        raise SystemExit(f"missing archive: {args.archive}")
    expected = graph_entities(args.dataset_root)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []
    entity_counts: dict[str, int] = {}
    rejected = 0

    with zipfile.ZipFile(args.archive) as archive:
        members = [m for m in archive.infolist() if not m.is_dir() and Path(m.filename).suffix.lower() in IMAGE_EXTS]
        for member in members:
            if not safe_member(member.filename):
                rejected += 1
                continue
            entity = canonical_entity(member.filename)
            if entity is None or entity not in expected:
                continue
            entity_dir = args.output / entity[3:]
            entity_dir.mkdir(parents=True, exist_ok=True)
            index = entity_counts.get(entity, 0)
            target = entity_dir / f"{index:02d}{Path(member.filename).suffix.lower()}"
            with archive.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            try:
                with Image.open(target) as image:
                    image.verify()
            except Exception:
                target.unlink(missing_ok=True)
                rejected += 1
                continue
            rows.append((entity, str(target.relative_to(args.output.parent))))
            entity_counts[entity] = index + 1

    rows.sort()
    per_entity: dict[str, int] = {}
    for entity, _ in rows:
        per_entity[entity] = per_entity.get(entity, 0) + 1
    covered = {entity for entity, count in per_entity.items() if count >= args.images_per_entity}
    no_images = sorted(expected - set(per_entity))
    below_required = sorted(set(per_entity) - covered)
    missing = sorted(expected - covered)
    manifest = args.output.parent / "images_full.tsv"
    manifest.write_text("".join(f"{entity}\t{path}\n" for entity, path in rows), encoding="utf-8")
    report = {
        "archive": str(args.archive),
        "archive_bytes": args.archive.stat().st_size,
        "graph_entities": len(expected),
        "manifest_rows": len(rows),
        "entities_with_required_images": len(covered),
        "required_images_per_entity": args.images_per_entity,
        "entities_without_images": len(no_images),
        "entities_below_required_images": len(below_required),
        "entities_not_ready": len(missing),
        "rejected_members": rejected,
        "status": "READY_FULL_IMAGE_CORPUS" if not missing else "INCOMPLETE_IMAGE_CORPUS",
    }
    (args.output.parent / "full_import_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if missing and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
