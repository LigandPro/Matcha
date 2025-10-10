import torch
import numpy as np
from typing import List

from matcha.dataset.complex_dataclasses import Ligand, ComplexBatch
from matcha.dataset.pdbbind import PDBBind, apply_random_rotation_inplace


def init_ligand_for_scoring(ligand: Ligand):
    """
    Initialize fields without randomization the position.

    Parameters:
    ----------
    ligand : Ligand
        The input ligand to be randomized.

    Returns:
    -------
    None
    """
    pos = np.copy(ligand.pos)

    tr = pos.mean(axis=0).reshape(1, 3)

    ligand.init_tr = tr.reshape(1, 3)
    ligand.t = torch.ones(1, dtype=torch.float32)
    ligand.pos = np.copy(pos)
    return ligand


class PDBBindForScoringInferenceMixin:
    def __init__(self, **kwargs):
        # Pop arguments specific to this class
        predicted_complex_positions_path = kwargs.pop(
            'predicted_complex_positions_path', None)

        # Call parent class initialization with remaining kwargs
        super().__init__(**kwargs)
        self.predicted_complex_positions = np.load(
            predicted_complex_positions_path, allow_pickle=True)[0]
        self.predicted_complex_positions = [(name, f'{name}_{i}', sample['transformed_orig'], sample.get('symm_rmsd', 0))
                                            for name, preds_list in self.predicted_complex_positions.items()
                                            for i, sample in enumerate(preds_list)]
        self.name2index = {complex.name: idx for idx,
                           complex in enumerate(self.complexes)}

    def __len__(self):
        return len(self.predicted_complex_positions)

    def __getitem__(self, idx):
        uid, uid_full, lig_pos, rmsd = self.predicted_complex_positions[idx]
        try:
            complex = self.__get_nonrand_item__(self.name2index[uid])
        except Exception as e:
            complex = self.__get_nonrand_item__(
                self.name2index[uid.split('_conf')[0]])
        complex.ligand.pos = np.copy(lig_pos[:len(complex.ligand.pos)])

        # 1. Rotate complex
        apply_random_rotation_inplace(complex)

        # 4. Compute ligand gt values
        complex.set_ground_truth_values()

        # save rmsd in complex
        init_ligand_for_scoring(complex.ligand)
        complex.original_augm_rot = np.eye(3, dtype=np.float32)[None, :, :]
        complex.ligand.rmsd = torch.tensor([rmsd]).float()
        complex.name = uid_full
        return complex


class PDBBindForScoringInference(PDBBindForScoringInferenceMixin, PDBBind):
    pass


def dummy_ranking_collate_fn(batch: List[ComplexBatch]) -> ComplexBatch:
    return batch[0]
