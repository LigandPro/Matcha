import torch

from matcha.dataset.complex_dataclasses import BondsBatch, ComplexBatch, LigandBatch, ProteinBatch
from matcha.utils.inference import euler


def _make_dummy_batch(batch_size: int = 1, num_atoms: int = 2) -> ComplexBatch:
    ligand = LigandBatch(
        x=torch.zeros(batch_size, num_atoms, 1, dtype=torch.float32),
        pos=torch.zeros(batch_size, num_atoms, 3, dtype=torch.float32),
        rot=torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1),
        orig_pos=torch.zeros(batch_size, num_atoms, 3, dtype=torch.float32),
        orig_pos_before_augm=torch.zeros(batch_size, num_atoms, 3, dtype=torch.float32),
        random_pos=torch.zeros(batch_size, num_atoms, 3, dtype=torch.float32),
        rotatable_bonds=torch.zeros(0, 2, dtype=torch.long),
        mask_rotate=[torch.zeros(0, num_atoms, dtype=torch.bool) for _ in range(batch_size)],
        init_tr=torch.zeros(batch_size, 3, dtype=torch.float32),
        pred_tr=None,
        init_rot=torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1),
        init_tor=torch.zeros(0, dtype=torch.float32),
        init_tor_ext=torch.zeros(batch_size, 0, dtype=torch.float32),
        final_tr=torch.zeros(batch_size, 3, dtype=torch.float32),
        final_rot=torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1),
        final_tor=torch.zeros(0, dtype=torch.float32),
        final_tor_ext=torch.zeros(batch_size, 0, dtype=torch.float32),
        num_atoms=torch.full((batch_size,), num_atoms, dtype=torch.long),
        tor_ptr=[0] * (batch_size + 1),
        is_padded_mask=torch.zeros(batch_size, num_atoms, dtype=torch.bool),
        true_pos=torch.zeros(batch_size, num_atoms, 3, dtype=torch.float32),
        t=torch.zeros(batch_size, dtype=torch.float32),
        rmsd=torch.zeros(batch_size, dtype=torch.float32),
        stage_num=torch.zeros(batch_size, dtype=torch.long),
        num_rotatable_bonds=torch.zeros(batch_size, dtype=torch.long),
        bond_periods=torch.zeros(0, dtype=torch.float32),
        rotatable_bonds_ext=BondsBatch(),
        num_rotatable_bonds_ext=torch.zeros(batch_size, dtype=torch.long),
    )
    protein = ProteinBatch(
        x=torch.zeros(batch_size, 1, 1, dtype=torch.float32),
        pos=torch.zeros(batch_size, 1, 3, dtype=torch.float32),
        seq=torch.zeros(batch_size, 1, dtype=torch.long),
        is_padded_mask=torch.zeros(batch_size, 1, dtype=torch.bool),
        all_atom_pos=torch.zeros(batch_size, 1, 3, dtype=torch.float32),
        all_atom_residue_ids=torch.zeros(batch_size, 1, dtype=torch.long),
        mask_all_atom_residue=torch.zeros(batch_size, 1, dtype=torch.bool),
        full_protein_center=torch.zeros(batch_size, 3, dtype=torch.float32),
        all_atom_names=None,
    )
    return ComplexBatch(
        ligand=ligand,
        protein=protein,
        names=[f"uid_{i}_conf0" for i in range(batch_size)],
        original_augm_rot=torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1),
        allbonds_mask=torch.zeros(batch_size, 1, dtype=torch.bool),
        rotbonds_mask=torch.zeros(batch_size, 1, dtype=torch.bool),
    )


class _DummyModel:
    def forward_step(self, batch, predict_torsion=True):
        bsz = batch.ligand.pos.shape[0]
        device = batch.ligand.pos.device
        dtr = torch.zeros(bsz, 3, device=device, dtype=batch.ligand.pos.dtype)
        drot = None
        dtor = torch.zeros(0, device=device, dtype=batch.ligand.pos.dtype)
        return dtr, drot, dtor, None


def test_euler_no_history_does_not_cpu_sync():
    batch = _make_dummy_batch()
    model = _DummyModel()
    optimized, *_rest, pos_hist, trajectory = euler(
        model,
        batch,
        device="cpu",
        num_steps=2,
        record_history=False,
        record_trajectory=False,
    )
    assert isinstance(optimized, ComplexBatch)
    assert pos_hist == []
    assert trajectory == []
