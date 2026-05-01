import pytest
from rdkit import Chem

from matcha.utils.preprocessing import generate_conformer_mols_batch


def test_generate_conformer_mols_batch_rdkit_shapes():
    mols = [
        Chem.MolFromSmiles("CCO"),
        Chem.MolFromSmiles("c1ccccc1"),
    ]
    out = generate_conformer_mols_batch(mols, confs_per_mol=2, backend="rdkit", chunk_size=2)
    assert len(out) == 2
    assert all(len(x) == 2 for x in out)
    assert all(m.GetNumConformers() == 1 for x in out for m in x)


def test_generate_conformer_mols_batch_invalid_backend():
    mols = [
        Chem.MolFromSmiles("CCO"),
        Chem.MolFromSmiles("c1ccccc1"),
    ]
    with pytest.raises(ValueError, match="Unknown conformer backend"):
        generate_conformer_mols_batch(mols, confs_per_mol=2, backend="nvmolkit", chunk_size=2)
