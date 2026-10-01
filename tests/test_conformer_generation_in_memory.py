from rdkit import Chem

from matcha.utils.preprocessing import generate_conformer_mols


def test_generate_conformer_mols_returns_single_conformer_molecules():
    mol = Chem.MolFromSmiles("CCO")
    conformers = generate_conformer_mols(mol, num_conformers=3, backend="rdkit")

    assert len(conformers) == 3
    assert all(m.GetNumConformers() == 1 for m in conformers)
    assert all(m.HasProp("ID") for m in conformers)
