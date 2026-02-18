import torch

from matcha.dataset.complex_dataclasses import ComplexBatch


def test_clone_structure_is_independent_for_to_mutations():
    batch = ComplexBatch()
    batch.ligand.pos = torch.randn(1, 2, 3, dtype=torch.float32)
    batch.ligand.x = torch.randn(1, 2, 4, dtype=torch.float32)
    batch.ligand.rotatable_bonds_ext.bond_type = torch.randn(1, 1, dtype=torch.float32)

    clone = batch.clone_structure()

    assert clone is not batch
    assert clone.ligand is not batch.ligand
    assert clone.protein is not batch.protein
    assert clone.ligand.rotatable_bonds_ext is not batch.ligand.rotatable_bonds_ext

    assert clone.ligand.pos.data_ptr() == batch.ligand.pos.data_ptr()
    assert clone.ligand.rotatable_bonds_ext.bond_type.data_ptr() == batch.ligand.rotatable_bonds_ext.bond_type.data_ptr()

    clone.to(dtype=torch.float16)

    assert batch.ligand.pos.dtype == torch.float32
    assert batch.ligand.rotatable_bonds_ext.bond_type.dtype == torch.float32
    assert clone.ligand.pos.dtype == torch.float16
    assert clone.ligand.rotatable_bonds_ext.bond_type.dtype == torch.float16
    assert clone.ligand.pos.data_ptr() != batch.ligand.pos.data_ptr()
    assert clone.ligand.rotatable_bonds_ext.bond_type.data_ptr() != batch.ligand.rotatable_bonds_ext.bond_type.data_ptr()

