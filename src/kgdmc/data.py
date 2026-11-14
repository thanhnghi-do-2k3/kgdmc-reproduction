from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

DATASETS = {
    "fb15k237": {
        "files": {
            "train.txt": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/train.txt",
            "valid.txt": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/valid.txt",
            "test.txt": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/test.txt",
            "entity2text.txt": "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/FB15k-237/entity2text.txt",
        },
        "expected": {"train.txt": 272115, "valid.txt": 17535, "test.txt": 20466},
        "source_note": "KG mirror and KG-BERT text; not asserted to be KGDMC-author assets",
    },
    "wn18rr": {
        "files": {
            "train.txt": "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/WN18RR/train.txt",
            "valid.txt": "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/WN18RR/valid.txt",
            "test.txt": "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/WN18RR/test.txt",
        },
        "expected": {"train.txt": 86835, "valid.txt": 3034, "test.txt": 3134},
        "source_note": "KG-BERT mirror; verify against canonical WN18RR source",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, destination: Path, resume: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if resume and partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={existing}-"} if existing else {})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("ab" if existing else "wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(destination)


def fetch_dataset(root: Path, dataset: str, components: set[str], resume: bool) -> dict:
    spec = DATASETS[dataset]
    raw = root / "datasets" / "raw" / dataset
    manifest = root / "datasets" / "manifests" / dataset / "source_manifest.json"
    records = []
    for name, url in spec["files"].items():
        kind = "text" if name == "entity2text.txt" else "graph"
        if kind not in components:
            continue
        dest = raw / name
        _download(url, dest, resume)
        records.append({"file": name, "url": url, "sha256": _sha256(dest), "bytes": dest.stat().st_size})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"dataset": dataset, "source_note": spec["source_note"], "records": records}, indent=2) + "\n")
    return {"dataset": dataset, "downloaded": records, "manifest": str(manifest)}


def _count(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def verify_dataset(root: Path, dataset: str, strict: bool) -> dict:
    spec = DATASETS[dataset]
    raw = root / "datasets" / "raw" / dataset
    files = {}
    ok = True
    for name, expected in spec["expected"].items():
        path = raw / name
        actual = _count(path) if path.exists() else None
        files[name] = {"expected": expected, "actual": actual, "ok": actual == expected}
        ok = ok and actual == expected
    text = raw / "entity2text.txt"
    files["entity2text.txt"] = {"exists": text.exists(), "rows": _count(text) if text.exists() else 0}
    if strict and not text.exists():
        ok = False
    image_root = root / "datasets" / "modality" / dataset / "images"
    image_count = sum(1 for p in image_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}) if image_root.exists() else 0
    image_status = "DATA_RECONSTRUCTED: local candidate images present; KGDMC equivalence unverified" if image_count else "BLOCKED_DATA: no approved KGDMC image corpus configured"
    return {"dataset": dataset, "ok": ok, "strict": strict, "files": files, "image_status": image_status}


def inspect_dataset(root: Path, dataset: str) -> dict:
    raw = root / "datasets" / "raw" / dataset
    result = {"dataset": dataset, "files": []}
    for path in sorted(raw.glob("*")):
        result["files"].append({"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return result

