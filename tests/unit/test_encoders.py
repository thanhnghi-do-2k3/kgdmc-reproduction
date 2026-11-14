from pathlib import Path

import pytest

from kgdmc.encoders import encode_images_cached


def test_image_encoder_requires_explicit_entity_manifest(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    with pytest.raises(ValueError, match="explicit TSV image manifest"):
        encode_images_cached(image_root, tmp_path / "cache")

