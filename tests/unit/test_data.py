from pathlib import Path

from kgdmc.data import _sha256


def test_sha256(tmp_path: Path) -> None:
    p = tmp_path / "x"
    p.write_bytes(b"kgdmc")
    assert len(_sha256(p)) == 64

