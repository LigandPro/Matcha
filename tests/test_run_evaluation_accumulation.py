import copy

import torch

from matcha.dataset.complex_dataclasses import BondsBatch, ComplexBatch, LigandBatch, ProteinBatch
from matcha.utils.inference import run_evaluation


def _make_dummy_batch(name: str = "uid_conf0") -> ComplexBatch:
    ligand = LigandBatch(
        x=torch.zeros(1, 2, 1, dtype=torch.float32),
        pos=torch.zeros(1, 2, 3, dtype=torch.float32),
        rot=torch.eye(3, dtype=torch.float32).unsqueeze(0),
        orig_pos=torch.zeros(1, 2, 3, dtype=torch.float32),
        orig_pos_before_augm=torch.zeros(1, 2, 3, dtype=torch.float32),
        random_pos=torch.zeros(1, 2, 3, dtype=torch.float32),
        rotatable_bonds=torch.zeros(0, 2, dtype=torch.long),
        mask_rotate=[torch.zeros(0, 2, dtype=torch.bool)],
        init_tr=torch.zeros(1, 3, dtype=torch.float32),
        pred_tr=None,
        init_rot=torch.eye(3, dtype=torch.float32).unsqueeze(0),
        init_tor=torch.zeros(0, dtype=torch.float32),
        init_tor_ext=torch.zeros(1, 0, dtype=torch.float32),
        final_tr=torch.zeros(1, 3, dtype=torch.float32),
        final_rot=torch.eye(3, dtype=torch.float32).unsqueeze(0),
        final_tor=torch.zeros(0, dtype=torch.float32),
        final_tor_ext=torch.zeros(1, 0, dtype=torch.float32),
        num_atoms=torch.tensor([2], dtype=torch.long),
        tor_ptr=[0, 0],
        is_padded_mask=torch.zeros(1, 2, dtype=torch.bool),
        true_pos=torch.zeros(1, 2, 3, dtype=torch.float32),
        t=torch.zeros(1, dtype=torch.float32),
        rmsd=torch.zeros(1, dtype=torch.float32),
        stage_num=torch.zeros(1, dtype=torch.long),
        num_rotatable_bonds=torch.zeros(1, dtype=torch.long),
        bond_periods=torch.zeros(0, dtype=torch.float32),
        rotatable_bonds_ext=BondsBatch(),
        num_rotatable_bonds_ext=torch.zeros(1, dtype=torch.long),
    )
    protein = ProteinBatch(
        x=torch.zeros(1, 1, 1, dtype=torch.float32),
        pos=torch.zeros(1, 1, 3, dtype=torch.float32),
        seq=torch.zeros(1, 1, dtype=torch.long),
        is_padded_mask=torch.zeros(1, 1, dtype=torch.bool),
        all_atom_pos=torch.zeros(1, 1, 3, dtype=torch.float32),
        all_atom_residue_ids=torch.zeros(1, 1, dtype=torch.long),
        mask_all_atom_residue=torch.zeros(1, 1, dtype=torch.bool),
        full_protein_center=torch.zeros(1, 3, dtype=torch.float32),
        all_atom_names=None,
    )
    return ComplexBatch(
        ligand=ligand,
        protein=protein,
        names=[name],
        original_augm_rot=torch.eye(3, dtype=torch.float32).unsqueeze(0),
        allbonds_mask=torch.zeros(1, 1, dtype=torch.bool),
        rotbonds_mask=torch.zeros(1, 1, dtype=torch.bool),
    )


def _solver(_model, batch, device, num_steps=1, **kwargs):
    optimized = copy.deepcopy(batch).to(device)
    tr_agg = None
    r_agg = None
    tor_agg = None
    return optimized, tr_agg, r_agg, tor_agg, [], []


def test_run_evaluation_uses_append_not_concat():
    batch = _make_dummy_batch(name="uid_conf0")
    dataloader = [{"batch": batch}, {"batch": batch}]
    metrics = run_evaluation(
        dataloader,
        num_steps=1,
        solver=_solver,
        model=None,
        device="cpu",
        compute_torsion_angles_pred=False,
        solver_kwargs={},
    )
    assert "uid_conf0" in metrics
    assert len(metrics["uid_conf0"]) == 2
