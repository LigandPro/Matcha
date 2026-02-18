import numpy as np

from matcha.dataset.complex_dataclasses import Complex, Ligand, Protein
from matcha.dataset.pdbbind import PDBBind


def test_reset_predicted_ligand_transforms_shares_protein(tmp_path):
    dataset = PDBBind.__new__(PDBBind)
    dataset.use_predicted_tr_only = False

    protein = Protein(full_protein_center=np.zeros(3, dtype=np.float32))
    ligand = Ligand(pos=np.zeros((2, 3), dtype=np.float32))
    dataset.complexes = [Complex(name="c1", ligand=ligand, protein=protein, original_augm_rot=None)]

    predicted = {
        "c1": [
            {
                "tr_pred_init": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "full_protein_center": np.zeros(3, dtype=np.float32),
                "transformed_orig": np.zeros((2, 3), dtype=np.float32),
            },
            {
                "tr_pred_init": np.array([4.0, 5.0, 6.0], dtype=np.float32),
                "full_protein_center": np.zeros(3, dtype=np.float32),
                "transformed_orig": np.zeros((2, 3), dtype=np.float32),
            },
        ]
    }

    path = tmp_path / "pred.npy"
    np.save(path, [predicted])

    dataset.reset_predicted_ligand_transforms(str(path), n_preds_to_use=2)

    assert len(dataset.complexes) == 2
    assert dataset.complexes[0].protein is dataset.complexes[1].protein

