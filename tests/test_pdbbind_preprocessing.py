import numpy as np

from matcha.dataset.pdbbind import PDBBind
from matcha.dataset.complex_dataclasses import Ligand
from matcha.utils.preprocessing import extract_receptor_structure_prody


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


class _FakeCA:
    def getChids(self):
        return np.array(["A", "A", "A"])

    def getSegnames(self):
        return np.array(["", "", ""])


class _FakeSelection:
    def __init__(self, chid="A", segname=""):
        self._chid = chid
        self._segname = segname

    def getSegnames(self):
        return np.array([self._segname])

    def getChids(self):
        return np.array([self._chid])


class _FakeRec:
    def __init__(self):
        self.ca = _FakeCA()

    def select(self, query):
        if query in {"resindex 10", "resindex 20"}:
            return _FakeSelection()
        raise AssertionError(f"unexpected query: {query}")


def test_extract_receptor_structure_aligns_chain_ids_to_unique_residues(monkeypatch):
    coords = np.zeros((2, 14, 3), dtype=np.float32)
    atom_names = np.full((2, 14), "C", dtype=object)
    seq = np.array(["A", "A"])
    resindices = np.array([10, 20])
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tokenized = np.array([[11], [12]])

    monkeypatch.setattr(
        "matcha.utils.preprocessing.get_coords",
        lambda rec: (coords, atom_names, seq, resindices),
    )

    result = extract_receptor_structure_prody(
        _FakeRec(),
        None,
        {"AA": (embeddings, tokenized)},
    )

    c_alpha_coords, lm_embeddings, sequences, chain_lengths, full_coords, full_atom_names, full_atom_residue_ids = result

    assert c_alpha_coords.shape == (2, 3)
    np.testing.assert_array_equal(lm_embeddings, embeddings)
    np.testing.assert_array_equal(sequences, tokenized)
    assert chain_lengths == [(2, 0)]
    assert full_coords.shape == (28, 3)
    assert full_atom_names.shape == (28,)
    np.testing.assert_array_equal(full_atom_residue_ids, np.repeat(np.arange(2), 14))


def test_extract_receptor_structure_returns_full_arity_when_no_valid_chain(monkeypatch):
    coords = np.zeros((1, 14, 3), dtype=np.float32)
    atom_names = np.full((1, 14), "C", dtype=object)
    seq = np.array(["A"])
    resindices = np.array([10])
    embeddings = np.zeros((1, 2), dtype=np.float32)
    tokenized = np.zeros((1, 1), dtype=np.int64)

    monkeypatch.setattr(
        "matcha.utils.preprocessing.get_coords",
        lambda rec: (coords, atom_names, seq, resindices),
    )

    class _FakeConformer:
        def GetPositions(self):
            return np.array([[100.0, 100.0, 100.0]], dtype=np.float32)

    class _FakeLigand:
        def GetConformer(self):
            return _FakeConformer()

    result = extract_receptor_structure_prody(
        _FakeRec(),
        _FakeLigand(),
        {"A": (embeddings, tokenized)},
    )

    assert result == (None, None, None, None, None, None, None)


def test_extract_receptor_structure_skips_missing_chain_embeddings(monkeypatch):
    coords = np.zeros((1, 14, 3), dtype=np.float32)
    atom_names = np.full((1, 14), "C", dtype=object)
    seq = np.array(["A"])
    resindices = np.array([10])

    monkeypatch.setattr(
        "matcha.utils.preprocessing.get_coords",
        lambda rec: (coords, atom_names, seq, resindices),
    )

    result = extract_receptor_structure_prody(_FakeRec(), None, {})

    assert result == (None, None, None, None, None, None, None)


def test_get_complex_falls_back_to_ligand_specific_receptor_extraction(monkeypatch, tmp_path):
    dataset = PDBBind.__new__(PDBBind)
    dataset.data_dir = str(tmp_path)
    dataset.dataset_type = "any_conf"
    dataset.use_all_chains = True
    dataset.is_train_dataset = False
    dataset.add_all_atom_pos = False
    dataset.min_lig_size = 1
    dataset.n_confs_to_use = 20

    ligand_mol = object()
    receptor_calls = []

    monkeypatch.setattr("matcha.dataset.pdbbind.parse_receptor", lambda *args, **kwargs: object())
    monkeypatch.setattr("matcha.dataset.pdbbind.read_molecule", lambda *args, **kwargs: ligand_mol)
    monkeypatch.setattr("matcha.dataset.pdbbind.split_molecule", lambda mol, min_lig_size: [mol])
    monkeypatch.setattr("matcha.dataset.pdbbind.generate_conformer_mols", lambda mol, num_conformers: [mol])

    def fake_extract(rec_model, lig, sequences_to_embeddings):
        receptor_calls.append(lig)
        if lig is None:
            raise KeyError("shared-template-sequence")
        return (
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int64),
            [(1, 0.0)],
            None,
            None,
            None,
        )

    monkeypatch.setattr("matcha.dataset.pdbbind.extract_receptor_structure_prody", fake_extract)

    def fake_get_ligand(lig, protein_center, parse_rotbonds=True):
        ligand = Ligand()
        ligand.pos = np.zeros((1, 3), dtype=np.float32)
        ligand.x = np.zeros((1, 1), dtype=np.float32)
        ligand.final_tr = np.zeros((1, 3), dtype=np.float32)
        ligand.orig_mol = lig
        ligand.bond_periods = None
        return ligand

    monkeypatch.setattr("matcha.dataset.pdbbind.get_ligand_without_randomization", fake_get_ligand)

    complexes = dataset._get_complex(["foo"], {"SEQ": (np.zeros((1, 2), dtype=np.float32), np.zeros((1, 1), dtype=np.int64))})

    assert len(complexes) == 1
    assert receptor_calls[0] is None
    assert receptor_calls[1] is ligand_mol
