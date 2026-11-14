import json

from kgdmc.readiness import write_readiness


def test_readiness_blocks_without_images(tmp_path):
    (tmp_path / "datasets/raw/fb15k237").mkdir(parents=True)
    (tmp_path / "datasets/manifests/fb15k237").mkdir(parents=True)
    for name, count in (("train.txt", 1), ("valid.txt", 1), ("test.txt", 1)):
        (tmp_path / "datasets/raw/fb15k237" / name).write_text("a\tr\tb\n")
    (tmp_path / "datasets/raw/fb15k237/entity2text.txt").write_text("a\tA\n")
    result = write_readiness(tmp_path, "fb15k237")
    assert result["ready_for_end_to_end_smoke"] is False
    assert "BLOCKED_DATA" in result["blockers"][0]
    assert json.loads((tmp_path / "reports/data_readiness.json").read_text())["real_image_files"] == 0

