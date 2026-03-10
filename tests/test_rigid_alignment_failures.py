from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_find_rigid_alignment_rejects_non_finite_input():
    from matcha.utils.transforms import RigidAlignmentError, find_rigid_alignment

    pos_a = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    pos_b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    with pytest.raises(RigidAlignmentError, match="non-finite numpy coordinates"):
        find_rigid_alignment(pos_a, pos_b)


def test_find_rigid_alignment_svd_fails_fast(monkeypatch):
    from matcha.utils import transforms

    calls = {"n": 0}

    def flaky_svd(*args, **kwargs):
        calls["n"] += 1
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "svd", flaky_svd)

    pos_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    pos_b = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)

    with pytest.raises(transforms.RigidAlignmentError, match="SVD failed for numpy input"):
        transforms.find_rigid_alignment(pos_a, pos_b)

    assert calls["n"] == 1


def test_set_ligand_data_from_preds_falls_back_to_stage2(monkeypatch):
    from matcha.dataset.complex_dataclasses import Ligand
    from matcha.dataset import pdbbind
    from matcha.utils.transforms import RigidAlignmentError

    ligand = Ligand(
        orig_pos=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        predicted_pos=np.array([[0.2, 0.1, 0.0], [1.2, 0.1, 0.0]], dtype=np.float32),
        rotatable_bonds=np.empty((0, 2), dtype=np.int32),
        bond_periods=np.empty((0,), dtype=np.float32),
        pred_tr=np.zeros((1, 3), dtype=np.float32),
        rmsd=torch.zeros(1),
    )
    ligand.mask_rotate_before_fixing = np.empty((0, 0), dtype=np.int32)
    ligand.rotatable_bonds_ext = SimpleNamespace(
        start=np.empty((0,), dtype=np.int32),
        end=np.empty((0,), dtype=np.int32),
        neighbor_of_start=np.empty((0,), dtype=np.int32),
        neighbor_of_end=np.empty((0,), dtype=np.int32),
        bond_periods=np.empty((0,), dtype=np.float32),
    )

    monkeypatch.setattr(
        pdbbind,
        "find_rigid_alignment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RigidAlignmentError("boom")),
    )

    fallback = {"called": False}

    def fake_randomize_ligand_with_preds(ligand_obj, with_preds, tr_mean=0.0, tr_std=5.0):
        fallback["called"] = True
        fallback["with_preds"] = with_preds
        ligand_obj.init_tr = ligand_obj.pred_tr.reshape(1, 3)
        ligand_obj.final_rot = np.eye(3, dtype=np.float32)[None, :, :]
        ligand_obj.final_tor = np.empty((0,), dtype=np.float32)
        ligand_obj.pos = ligand_obj.predicted_pos.copy()
        ligand_obj.t = torch.zeros(1)
        ligand_obj.stage_num = torch.tensor([2])

    monkeypatch.setattr(pdbbind, "randomize_ligand_with_preds", fake_randomize_ligand_with_preds)

    assert pdbbind.set_ligand_data_from_preds(ligand, "bad_ligand") is False
    assert fallback == {"called": True, "with_preds": True}
    assert torch.equal(ligand.stage_num, torch.tensor([2]))


def test_run_evaluation_skips_only_failed_alignment_sample(monkeypatch):
    from matcha.dataset.complex_dataclasses import BondsBatch, ComplexBatch, LigandBatch, ProteinBatch
    from matcha.utils import inference
    from matcha.utils.transforms import RigidAlignmentError

    pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    empty_bonds = BondsBatch(
        start=torch.zeros((2, 0), dtype=torch.long),
        end=torch.zeros((2, 0), dtype=torch.long),
        neighbor_of_start=torch.zeros((2, 0), dtype=torch.long),
        neighbor_of_end=torch.zeros((2, 0), dtype=torch.long),
        bond_periods=torch.zeros((2, 0), dtype=torch.float32),
        is_padded_mask=torch.ones((2, 0), dtype=torch.bool),
    )
    ligand = LigandBatch(
        x=torch.zeros((2, 3, 1), dtype=torch.float32),
        pos=pos.clone(),
        orig_pos=pos.clone(),
        orig_pos_before_augm=pos.clone(),
        random_pos=pos.clone(),
        rotatable_bonds=torch.empty((0, 2), dtype=torch.long),
        mask_rotate=[
            torch.zeros((0, 3), dtype=torch.bool),
            torch.zeros((0, 3), dtype=torch.bool),
        ],
        init_tr=torch.zeros((2, 3), dtype=torch.float32),
        init_rot=torch.eye(3, dtype=torch.float32).repeat(2, 1, 1),
        init_tor=torch.empty((0,), dtype=torch.float32),
        init_tor_ext=torch.zeros((2, 0), dtype=torch.float32),
        final_tr=torch.zeros((2, 3), dtype=torch.float32),
        final_rot=torch.eye(3, dtype=torch.float32).repeat(2, 1, 1),
        final_tor=torch.empty((0,), dtype=torch.float32),
        final_tor_ext=torch.zeros((2, 0), dtype=torch.float32),
        pred_tr=torch.zeros((2, 3), dtype=torch.float32),
        num_atoms=torch.tensor([3, 3], dtype=torch.long),
        tor_ptr=[0, 0, 0],
        orig_mols=[None, None],
        is_padded_mask=torch.zeros((2, 3), dtype=torch.bool),
        true_pos=pos.clone(),
        num_rotatable_bonds=torch.tensor([0, 0], dtype=torch.long),
        num_rotatable_bonds_ext=torch.tensor([0, 0], dtype=torch.long),
        t=torch.zeros((2,), dtype=torch.float32),
        rmsd=torch.zeros((2,), dtype=torch.float32),
        stage_num=torch.zeros((2,), dtype=torch.long),
        bond_periods=torch.empty((0,), dtype=torch.float32),
        rotatable_bonds_ext=empty_bonds,
    )
    protein = ProteinBatch(
        x=torch.zeros((2, 1, 1), dtype=torch.float32),
        pos=torch.zeros((2, 1, 3), dtype=torch.float32),
        seq=torch.zeros((2, 1), dtype=torch.long),
        is_padded_mask=torch.zeros((2, 1), dtype=torch.bool),
        all_atom_pos=torch.zeros((2, 1, 3), dtype=torch.float32),
        all_atom_residue_ids=torch.zeros((2, 1), dtype=torch.long),
        mask_all_atom_residue=torch.zeros((2, 1, 1), dtype=torch.bool),
        full_protein_center=torch.zeros((2, 3), dtype=torch.float32),
        all_atom_names=torch.zeros((2, 1), dtype=torch.long),
    )
    batch = ComplexBatch(
        ligand=ligand,
        protein=protein,
        names=["bad_conf0", "good_conf0"],
        original_augm_rot=torch.eye(3, dtype=torch.float32).repeat(2, 1, 1),
        allbonds_mask=torch.zeros((0, 0), dtype=torch.bool),
        rotbonds_mask=torch.zeros((0, 0), dtype=torch.bool),
    )

    def solver(_model, solver_batch, device, num_steps, **_kwargs):
        return (
            solver_batch,
            torch.zeros((2, 3), dtype=torch.float32),
            torch.eye(3, dtype=torch.float32).repeat(2, 1, 1),
            torch.empty((0,), dtype=torch.float32),
            [],
            [],
        )

    calls = {"n": 0}

    def fake_find_rigid_alignment(_pos_pred, _pos_true):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RigidAlignmentError("SVD did not converge")
        return torch.eye(3, dtype=torch.float32), torch.zeros((3,), dtype=torch.float32)

    monkeypatch.setattr(inference, "find_rigid_alignment", fake_find_rigid_alignment)

    metrics = inference.run_evaluation(
        dataloader=[{"batch": batch}],
        num_steps=1,
        solver=solver,
        model=object(),
        device="cpu",
    )

    assert metrics["bad_conf0"] == []
    assert len(metrics["good_conf0"]) == 1
