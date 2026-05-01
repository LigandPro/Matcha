from types import SimpleNamespace

import numpy as np

from matcha.dataset.pdbbind import PDBBind


def _make_complex(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        ligand=SimpleNamespace(pred_tr=None, predicted_pos=None),
        protein=SimpleNamespace(full_protein_center=np.zeros((1, 3), dtype=np.float32)),
        original_augm_rot=np.eye(3, dtype=np.float32),
    )


def _make_prediction(value: float) -> dict:
    return {
        "tr_pred_init": np.array([[value, value + 1, value + 2]], dtype=np.float32),
        "full_protein_center": np.zeros((1, 3), dtype=np.float32),
        "transformed_orig": np.array([[value, value, value]], dtype=np.float32),
    }


def test_set_predicted_ligand_transforms_skips_empty_prediction_lists(tmp_path):
    dataset = PDBBind.__new__(PDBBind)
    dataset.complexes = [
        _make_complex("ligand_a"),
        _make_complex("ligand_b"),
        _make_complex("ligand_c"),
    ]
    dataset.use_predicted_tr_only = True

    predictions = {
        "ligand_a": [_make_prediction(1.0), _make_prediction(2.0)],
        "ligand_b": [],
        "ligand_c": [_make_prediction(3.0)],
    }
    predictions_path = tmp_path / "stage2_any_conf.npy"
    np.save(predictions_path, [predictions], allow_pickle=True)

    dataset._set_predicted_ligand_transforms(str(predictions_path), n_preds_to_use=2)

    assert [complex.name for complex in dataset.complexes] == [
        "ligand_a",
        "ligand_a",
        "ligand_c",
    ]
    pred_tr_values = [complex.ligand.pred_tr.flatten().tolist() for complex in dataset.complexes]
    assert pred_tr_values == [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ]
