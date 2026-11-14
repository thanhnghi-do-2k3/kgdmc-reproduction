#!/usr/bin/env python3
"""Build a portable FB15K-237 image subset from the MMKB URL archive."""

from __future__ import annotations

import argparse
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_entities(root: Path) -> set[str]:
    entities: set[str] = set()
    for name in ("train.txt", "valid.txt", "test.txt"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                entities.update((parts[0], parts[2]))
    return entities


def read_candidates(url_dir: Path, entities: set[str], per_entity: int) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {entity: [] for entity in entities}
    for provider in ("google", "bing", "yahoo"):
        path = url_dir / f"URLS_{provider}.txt"
        for line in path.open(encoding="utf-8", errors="ignore"):
            try:
                url, raw_id = line.rstrip("\n").split("\t", 1)
                raw_entity = raw_id.rsplit("/", 1)[0]
                entity = "/m/" + raw_entity[2:] if raw_entity.startswith("m.") else raw_entity
            except ValueError:
                continue
            if entity not in entities or len(result[entity]) >= per_entity:
                continue
            url = url.replace(r"\u003d", "=")
            result[entity].append(url)
    return {entity: urls for entity, urls in result.items() if urls}


def fetch_one(entity: str, urls: list[str], out_dir: Path, images_per_entity: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for index, url in enumerate(urls[:images_per_entity]):
        target_name = f"{entity[3:]}_{index}.jpg" if images_per_entity > 1 else f"{entity[3:]}.jpg"
        target = out_dir / target_name
        if target.exists():
            try:
                Image.open(target).verify()
                rows.append((entity, str(target.relative_to(out_dir.parent))))
                continue
            except Exception:
                target.unlink(missing_ok=True)
        temp = out_dir / f".{entity[3:]}_{index}.download"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=20) as response:
                temp.write_bytes(response.read())
            with Image.open(temp) as image:
                image.convert("RGB").save(target, format="JPEG", quality=92)
            temp.unlink(missing_ok=True)
            rows.append((entity, str(target.relative_to(out_dir.parent))))
        except Exception:
            temp.unlink(missing_ok=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/raw/fb15k237"))
    parser.add_argument("--url-dir", type=Path, default=Path("datasets/downloads/mmkb_urls/image-graph_urls"))
    parser.add_argument("--output", type=Path, default=Path("datasets/subsets/fb15k237_1000_seed42"))
    parser.add_argument("--entities", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--images-per-entity", type=int, default=1)
    parser.add_argument("--finalize-existing", action="store_true", help="Use already downloaded valid JPGs without contacting source URLs")
    args = parser.parse_args()

    all_entities = sorted(read_entities(args.dataset_root))
    candidates = read_candidates(args.url_dir, set(all_entities), per_entity=25)
    rng = random.Random(args.seed)
    ordered = sorted(candidates)
    rng.shuffle(ordered)
    selected = ordered[: args.entities]
    if len(selected) < args.entities:
        raise SystemExit(f"Only {len(selected)} entities have candidate URLs; requested {args.entities}")

    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[tuple[str, str]] = []
    if args.finalize_existing:
        for path in sorted(image_dir.glob("*.jpg")):
            entity = f"/m/{re.sub(r'_\d+$', '', path.stem)}"
            if entity in candidates:
                try:
                    Image.open(path).verify()
                    manifest_rows.append((entity, str(path.relative_to(args.output))))
                except Exception:
                    pass
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_one, entity, candidates[entity], image_dir, args.images_per_entity) for entity in selected]
            for future in as_completed(futures):
                manifest_rows.extend(future.result())

    manifest_rows.sort()
    expected_images = args.entities * args.images_per_entity
    if len(manifest_rows) < expected_images and not args.finalize_existing:
        raise SystemExit(f"Downloaded {len(manifest_rows)}/{expected_images} valid images; increase URL candidates or retry")
    (args.output / "images.tsv").write_text("\n".join(f"{entity}\timages/{path.split('images/', 1)[-1]}" for entity, path in manifest_rows) + "\n", encoding="utf-8")
    (args.output / "entities.txt").write_text("\n".join(entity for entity, _ in manifest_rows) + "\n", encoding="utf-8")

    selected_set = {entity for entity, _ in manifest_rows}
    for split in ("train", "valid", "test"):
        source = args.dataset_root / f"{split}.txt"
        rows = [line for line in source.read_text(encoding="utf-8").splitlines() if len(line.split("\t")) == 3 and line.split("\t")[0] in selected_set and line.split("\t")[2] in selected_set]
        (args.output / f"{split}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print({"requested_entities": args.entities, "downloaded_images": len(manifest_rows), "downloaded_entities": len({entity for entity, _ in manifest_rows}), "images_per_entity": args.images_per_entity, "output": str(args.output), "seed": args.seed})


if __name__ == "__main__":
    main()
