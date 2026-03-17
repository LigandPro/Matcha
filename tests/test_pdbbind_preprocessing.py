import numpy as np

from matcha.dataset.pdbbind import PDBBind


def _make_dataset(tmp_path, dataset_type="any_conf"):
    dataset = PDBBind.__new__(PDBBind)
    dataset.data_dir = str(tmp_path)
    dataset.dataset_type = dataset_type
    dataset.aa_mapping = {"A": 1, "C": 2}
    return dataset


def _write_receptor(tmp_path, complex_name, content):
    complex_dir = tmp_path / complex_name
    complex_dir.mkdir(parents=True, exist_ok=True)
    receptor_path = complex_dir / f"{complex_name}_protein.pdb"
    receptor_path.write_text(content, encoding="utf-8")


def test_load_embeddings_preserves_superlig_complex_name(monkeypatch, tmp_path):
    dataset = _make_dataset(tmp_path)

    monkeypatch.setattr(
        "matcha.dataset.pdbbind.torch.load",
        lambda path, weights_only=False: {"foo_superlig_chain_0": np.array([[1.0, 2.0]])},
    )
    monkeypatch.setattr(
        "matcha.dataset.pdbbind.load",
        lambda path: {"foo_superlig_chain_0": "AC"},
    )

    lm_embeddings, sequence_chains, tokenized_sequence_chains = dataset._load_embeddings(
        "embeddings.pt",
        "sequences.pkl",
        ["foo_superlig"],
    )

    assert len(lm_embeddings[0]) == 1
    np.testing.assert_array_equal(sequence_chains[0][0], np.array(["A", "C"]))
    np.testing.assert_array_equal(tokenized_sequence_chains[0][0], np.array([[1], [2]]))


def test_complexes_share_single_receptor_rejects_different_receptors(tmp_path):
    dataset = _make_dataset(tmp_path)
    _write_receptor(tmp_path, "first", "ATOM      1  N   ALA A   1      10.000  10.000  10.000\n")
    _write_receptor(tmp_path, "second", "ATOM      1  N   GLY A   1      20.000  20.000  20.000\n")

    assert dataset._complexes_share_single_receptor(["first", "second"]) is False


def test_complexes_share_single_receptor_accepts_identical_receptors(tmp_path):
    dataset = _make_dataset(tmp_path)
    protein = "ATOM      1  N   ALA A   1      10.000  10.000  10.000\n"
    _write_receptor(tmp_path, "first", protein)
    _write_receptor(tmp_path, "second", protein)

    assert dataset._complexes_share_single_receptor(["first", "second"]) is True
