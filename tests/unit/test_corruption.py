from pathlib import Path

from kgdmc.corruption import generate_corruptions, read_triples


def test_corruption_uses_real_triples_and_avoids_truth_collisions(tmp_path: Path) -> None:
    path = tmp_path / "train.txt"
    path.write_text("a\tr\tb\na\tr2\tc\nd\tr\tb\n")
    triples = read_triples(path)
    records = generate_corruptions(triples, fraction=1.0, seed=7)
    truth = set(triples)
    assert len(records) == 9
    assert all(item.corrupted not in truth for item in records)
    assert {item.error_type for item in records} == {"head", "relation", "tail"}

