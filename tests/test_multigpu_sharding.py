from pathlib import Path

import pytest

from matcha.utils.multigpu import parse_gpus, shard_ligand_files


def test_parse_gpus_valid():
    assert parse_gpus("2,3,4") == [2, 3, 4]
    assert parse_gpus("0") == [0]
    assert parse_gpus("1, 5") == [1, 5]


@pytest.mark.parametrize("raw", ["", "1,,2", "-1,2", "1,1", "x,2"])
def test_parse_gpus_invalid(raw):
    with pytest.raises(ValueError):
        parse_gpus(raw)


def test_shard_ligand_files_round_robin_is_deterministic():
    files = [
        Path("z.mol2"),
        Path("b.mol2"),
        Path("a.mol2"),
        Path("c.sdf"),
        Path("d.sdf"),
    ]
    shards = shard_ligand_files(files, n_shards=3)
    assert [[p.name for p in chunk] for chunk in shards] == [
        ["a.mol2", "d.sdf"],
        ["b.mol2", "z.mol2"],
        ["c.sdf"],
    ]


def test_shard_ligand_files_more_shards_than_files():
    files = [Path("a.sdf"), Path("b.mol2")]
    shards = shard_ligand_files(files, n_shards=3)
    assert len(shards) == 3
    assert [[p.name for p in chunk] for chunk in shards] == [["a.sdf"], ["b.mol2"], []]
