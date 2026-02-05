import torch
from dataclasses import dataclass, field
import numpy as np
from rdkit import Chem
from typing import List, Optional


@dataclass
class Bonds:
    """
    A data class to represent the properties of bonds in a molecule.

    Attributes:
    bond_type (np.ndarray): Categorical feature representing the type of bond (float constants are from rdkit.Chem.BondType).
    is_aromatic (np.ndarray): Categorical feature indicating if the bond is aromatic (True/False).
    start (Optional[np.ndarray]): Indices of the starting atoms of the bonds.
    end (Optional[np.ndarray]): Indices of the ending atoms of the bonds.
    neighbor_of_start (Optional[np.ndarray]): Indices of the randomly sampled neighbor atoms of the starting atoms of the bonds.
    neighbor_of_end (Optional[np.ndarray]): Indices of the randomly sampled neighbor atoms of the ending atoms of the bonds.
    all_neighbors_of_start (Optional[np.ndarray]): Indices of all neighbors of the starting atoms of the bonds.
    all_neighbors_of_end (Optional[np.ndarray]): Indices of all neighbors of the ending atoms of the bonds.
    length (Optional[np.ndarray]): Lengths of the bonds.
    mask_rotate (Optional[np.ndarray]): Boolean mask indicating which bonds can rotate.
    is_conjugated (Optional[np.ndarray]): Categorical feature indicating if the bond is conjugated (True/False).
    is_in_ring (Optional[np.ndarray]): Categorical feature indicating if the bond is part of a ring (True/False).
    is_rotatable (Optional[np.ndarray]): Categorical feature indicating if the bond is rotatable (True/False).
    bond_periods (Optional[np.ndarray]): Rotational periods of the bonds.
    angles (Optional[np.ndarray]): Angles of the rotatable bonds.
    angle_histograms (Optional[np.ndarray]): Histograms of angle distributions of the rotatable bonds.
    """
    bond_type: np.ndarray  # Categorical feature
    is_aromatic: np.ndarray  # Categorical feature (True/False)
    start: Optional[np.ndarray] = None
    end: Optional[np.ndarray] = None
    neighbor_of_start: Optional[np.ndarray] = None
    neighbor_of_end: Optional[np.ndarray] = None
    all_neighbors_of_start: Optional[List[np.ndarray]] = None
    all_neighbors_of_end: Optional[List[np.ndarray]] = None
    length: Optional[np.ndarray] = None
    mask_rotate: Optional[np.ndarray] = None
    is_conjugated: Optional[np.ndarray] = None
    is_in_ring: Optional[np.ndarray] = None
    is_rotatable: Optional[np.ndarray] = None
    bond_periods: Optional[np.ndarray] = None
    angles: Optional[np.ndarray] = None
    angle_histograms: Optional[np.ndarray] = None


@dataclass
class BondsBatch:
    """
    A batch of Bonds data with padding.

    Attributes:
    ----------
    bond_type : torch.Tensor
        Padded tensor of bond types, shape: (batch_size, max_num_bonds).
    start : Optional[torch.Tensor]
        Padded tensor of indices of the starting atoms of the bonds, shape: (batch_size, max_num_bonds).
    end : Optional[torch.Tensor]
        Padded tensor of indices of the ending atoms of the bonds, shape: (batch_size, max_num_bonds).
    neighbor_of_start : Optional[torch.Tensor]
        Padded tensor of indices of the neighbor atoms of the starting atoms of the bonds, shape: (batch_size, max_num_bonds).
    neighbor_of_end : Optional[torch.Tensor]
        Padded tensor of indices of the neighbor atoms of the ending atoms of the bonds, shape: (batch_size, max_num_bonds).
    length : Optional[torch.Tensor]
        Padded tensor of bond lengths, shape: (batch_size, max_num_bonds).
    mask_rotate : Optional[torch.Tensor]
        Padded boolean mask indicating which bonds can rotate, shape: (batch_size, max_num_bonds).
    is_conjugated : Optional[torch.Tensor]
        Padded tensor indicating if bonds are conjugated, shape: (batch_size, max_num_bonds).
    is_in_ring : Optional[torch.Tensor]
        Padded tensor indicating if bonds are part of a ring, shape: (batch_size, max_num_bonds).
    is_rotatable : Optional[torch.Tensor]
        Padded tensor indicating if bonds are rotatable, shape: (batch_size, max_num_bonds).
    num_rotatable_bonds : Optional[torch.Tensor]
        Number of rotatable bonds per sample in a batch, shape: (batch_size, ).
    bond_periods : Optional[torch.Tensor]
        Padded tensor of rotational periods of the bonds, shape: (batch_size, max_num_bonds).
    angles : Optional[torch.Tensor]
        Padded tensor of angles of the rotatable bonds, shape: (batch_size, max_num_bonds).
    """
    bond_type: torch.Tensor = torch.empty((0, 0))
    start: Optional[torch.Tensor] = torch.empty((0, 0))
    end: Optional[torch.Tensor] = torch.empty((0, 0))
    neighbor_of_start: Optional[torch.Tensor] = torch.empty((0, 0))
    neighbor_of_end: Optional[torch.Tensor] = torch.empty((0, 0))
    length: Optional[torch.Tensor] = torch.empty((0, 0))
    mask_rotate: Optional[torch.Tensor] = torch.empty((0, 0))
    is_conjugated: Optional[torch.Tensor] = torch.empty((0, 0))
    is_in_ring: Optional[torch.Tensor] = torch.empty((0, 0))
    is_rotatable: Optional[torch.Tensor] = torch.empty((0, 0))
    num_rotatable_bonds: Optional[torch.Tensor] = torch.empty((0,))
    is_padded_mask: Optional[torch.Tensor] = torch.empty((0,))
    is_aromatic: Optional[torch.Tensor] = torch.empty((0,))
    bond_periods: Optional[torch.Tensor] = torch.empty((0,))
    angles: Optional[torch.Tensor] = torch.empty((0,))
    angle_histograms: Optional[torch.Tensor] = torch.empty((0,))

    def to(self, *args, **kwargs):
        """
        Transfer all tensors in the BondsBatch to the specified device.
        """
        self.bond_type = self.bond_type.to(*args, **kwargs)
        if self.start is not None:
            self.start = self.start.to(*args, **kwargs)
        if self.end is not None:
            self.end = self.end.to(*args, **kwargs)
        if self.neighbor_of_start is not None:
            self.neighbor_of_start = self.neighbor_of_start.to(*args, **kwargs)
        if self.neighbor_of_end is not None:
            self.neighbor_of_end = self.neighbor_of_end.to(*args, **kwargs)
        if self.length is not None:
            self.length = self.length.to(*args, **kwargs)
        if self.mask_rotate is not None:
            self.mask_rotate = self.mask_rotate.to(*args, **kwargs)
        if self.is_conjugated is not None:
            self.is_conjugated = self.is_conjugated.to(*args, **kwargs)
        if self.is_in_ring is not None:
            self.is_in_ring = self.is_in_ring.to(*args, **kwargs)
        if self.is_aromatic is not None:
            self.is_aromatic = self.is_aromatic.to(*args, **kwargs)
        if self.is_rotatable is not None:
            self.is_rotatable = self.is_rotatable.to(*args, **kwargs)
        if self.is_padded_mask is not None:
            self.is_padded_mask = self.is_padded_mask.to(*args, **kwargs)
        if self.bond_periods is not None:
            self.bond_periods = self.bond_periods.to(*args, **kwargs)
        if self.angles is not None:
            self.angles = self.angles.to(*args, **kwargs)
        if self.angle_histograms is not None:
            self.angle_histograms = self.angle_histograms.to(*args, **kwargs)
        return self


@dataclass
class LigandBatch:
    """
    A batch of ligand data.

    Attributes:
    ----------
    x : torch.Tensor
        Feature matrix of the ligand, shape: (batch_size, max_seq_len, feature_dim).
    pos : torch.Tensor
        Position matrix of the ligand, shape: (batch_size, max_seq_len, 3).
    rot : torch.Tensor
        Current rotation matrices for the batch, shape: (batch_size, 3, 3).
    true_pos : torch.Tensor
        Original (true) position matrix of the ligand before complex augmentations, shape: (batch_size, max_seq_len, 3).
    orig_pos : torch.Tensor
        Original (true maybe with torsions) position matrix of the ligand after complex augmentations, shape: (batch_size, max_seq_len, 3).
    orig_pos_before_augm : torch.Tensor
        Original (true maybe with torsions) position matrix of the ligand before complex augmentations, shape: (batch_size, max_seq_len, 3).
    rotatable_bonds : torch.Tensor
        Rotatable bonds in the batch, shape: (number of all rotatable bonds in a batch, 4).
    bond_periods : torch.Tensor
        Bond periods for the batch, shape: (number of all rotatable bonds in a batch, ).
    mask_rotate : Optional[List[torch.Tensor]]
        List of tensors indicating which atoms to rotate for each bond, 
        shape: (num_rotatable_bonds, num_atoms).
    init_tr : torch.Tensor
        Initial translation vectors for the batch, shape: (batch_size, 3).
    pred_tr : torch.Tensor
        Predicted translation vectors for the batch, shape: (batch_size, 3).
    init_rot : torch.Tensor
        Initial rotation matrices for the batch, shape: (batch_size, 3, 3).
    init_tor : torch.Tensor
        Initial torsion angles for the batch, shape: (batch_size,).
    final_tr : torch.Tensor
        Ground-truth translation vectors for the batch, shape: (batch_size, 3).
    final_rot : torch.Tensor
        Ground-truth rotation matrices for the batch, shape: (batch_size, 3, 3).
    final_tor : torch.Tensor
        Ground-truth torsion angles for the batch, shape: (batch_size,).
    num_atoms : List[int]
        Number of atoms in each sample.
    tor_ptr : List[int]
        Indices for each molecule's torsion angles in the tor tensor.
    num_rotatable_bonds : torch.Tensor 
        Number of rotatable bonds per ligand in a batch, shape: (batch_size, ).
    rmsd: torch.Tensor
        RMSD of the ligand to the original ligand position, shape: (batch_size, ).
    t: torch.Tensor
        Ligand time, shape: (batch_size, ).
    """
    x: torch.Tensor = torch.empty(0)
    pos: torch.Tensor = torch.empty(0)
    rot: torch.Tensor = torch.empty(0)
    orig_pos: torch.Tensor = torch.empty(0)
    orig_pos_before_augm: torch.Tensor = torch.empty(0)
    random_pos: torch.Tensor = torch.empty(0)
    rotatable_bonds: torch.Tensor = torch.empty(0)
    mask_rotate: Optional[List[torch.Tensor]] = None
    init_tr: torch.Tensor = torch.empty(0)
    init_rot: torch.Tensor = torch.empty(0)
    init_tor: torch.Tensor = torch.empty(0)
    init_tor_ext: torch.Tensor = torch.empty(0)
    final_tr: torch.Tensor = torch.empty(0)
    final_rot: torch.Tensor = torch.empty(0)
    final_tor: torch.Tensor = torch.empty(0)
    final_tor_ext: torch.Tensor = torch.empty(0)
    pred_tr: torch.Tensor = torch.empty(0)
    num_atoms: torch.Tensor = torch.empty(0)
    tor_ptr: List[int] = None
    orig_mols: List[Chem.Mol] = None
    is_padded_mask: torch.Tensor = torch.empty(0)
    true_pos: torch.Tensor = torch.empty(0)
    t: torch.Tensor = torch.empty(0)
    rmsd: torch.Tensor = torch.empty(0)
    stage_num: torch.Tensor = torch.empty(0)
    num_rotatable_bonds: torch.Tensor = torch.empty(0)
    bond_periods: torch.Tensor = torch.empty(0)

    rotatable_bonds_ext: BondsBatch = field(default_factory=BondsBatch)
    num_rotatable_bonds_ext: torch.Tensor = torch.empty((0,))


@dataclass
class ProteinBatch:
    """
    A batch of protein data.

    Attributes:
    ----------
    x : torch.Tensor
        Feature matrix of the protein, shape: (batch_size, max_seq_len, feature_dim).
    pos : torch.Tensor
        Position matrix of the protein (pocket atoms), shape: (batch_size, max_seq_len, 3).
    all_atom_pos : torch.Tensor
        Position matrix of all protein atoms, shape: (batch_size, num_protein_atoms, 3).
    all_atom_residue_ids : torch.Tensor
        Residue ids of all protein atoms, shape: (batch_size, num_protein_atoms).
    seq : torch.Tensor
        Encoded aa tokens, shape: (batch_size, max_seq_len).
    full_protein_center : torch.Tensor
        Center of all protein atoms (shift for the initial pdb coordinates for both protein and ligand), shape: (batch_size, 3).
    """
    x: torch.Tensor = torch.empty(0)
    pos: torch.Tensor = torch.empty(0)
    seq: torch.Tensor = torch.empty(0)
    is_padded_mask: torch.Tensor = torch.empty(0)
    all_atom_pos: torch.Tensor = torch.empty(0)
    all_atom_residue_ids: torch.Tensor = torch.empty(0)
    mask_all_atom_residue: torch.Tensor = torch.empty(0)
    full_protein_center: torch.Tensor = torch.empty(0)
    all_atom_names: np.ndarray = None


@dataclass
class ComplexBatch:
    """
    A batch of complex data, containing ligand and protein batches.

    Attributes:
    ----------
    ligand : LigandBatch
        Batch of ligand data.
    protein : ProteinBatch
        Batch of protein data.
    """
    ligand: LigandBatch = field(default_factory=LigandBatch)
    protein: ProteinBatch = field(default_factory=ProteinBatch)
    names: List[str] = None
    original_augm_rot: torch.Tensor = torch.empty(0)
    allbonds_mask: torch.Tensor = torch.empty((0, 0), dtype=torch.bool)
    rotbonds_mask: torch.Tensor = torch.empty((0, 0), dtype=torch.bool)

    def __repr__(self):
        ligand_repr = (
            f"LigandBatch(\n"
            f"  x: shape={self.ligand.x.shape}, dtype={self.ligand.x.dtype}\n"
            f"  pos: shape={self.ligand.pos.shape}, dtype={self.ligand.pos.dtype}\n"
            f"  rot: shape={self.ligand.rot.shape}, dtype={self.ligand.rot.dtype}\n"
            f"  t: shape={self.ligand.t.shape}, dtype={self.ligand.t.dtype}\n"
            f"  rmsd: shape={self.ligand.rmsd.shape}, dtype={self.ligand.rmsd.dtype}\n"
            f"  rotatable_bonds: shape={self.ligand.rotatable_bonds.shape}, dtype={self.ligand.rotatable_bonds.dtype}\n"
            f"  init_tr: shape={self.ligand.init_tr.shape}, dtype={self.ligand.init_tr.dtype}\n"
            f"  pred_tr: shape={self.ligand.pred_tr.shape}, dtype={self.ligand.pred_tr.dtype}\n"
            f"  init_rot: shape={self.ligand.init_rot.shape}, dtype={self.ligand.init_rot.dtype}\n"
            f"  init_tor: shape={self.ligand.init_tor.shape}, dtype={self.ligand.init_tor.dtype}\n"
            f"  final_tr: shape={self.ligand.final_tr.shape}, dtype={self.ligand.final_tr.dtype}\n"
            f"  final_rot: shape={self.ligand.final_rot.shape}, dtype={self.ligand.final_rot.dtype}\n"
            f"  final_tor: shape={self.ligand.final_tor.shape}, dtype={self.ligand.final_tor.dtype}\n"
        )

        if self.ligand.mask_rotate is not None:
            ligand_repr += (
                f"  mask_rotate: len={len(self.ligand.mask_rotate)}),\n"
            )

        protein_repr = (
            f"ProteinBatch(\n"
            f"  x: shape={self.protein.x.shape}, dtype={self.protein.x.dtype}\n"
            f"  pos: shape={self.protein.pos.shape}, dtype={self.protein.pos.dtype}\n"
        )

        augm_repr = (
            f"Complex augmentations(\n"
            f"  original_augm_rot: shape={self.original_augm_rot.shape}, dtype={self.original_augm_rot.dtype}),\n"
        )

        return (
            f"ComplexBatch(\n  ligand={ligand_repr}\n  protein={protein_repr}\n"
            f" names={len(self.names) if self.names else 0}\n"
            f" augms={augm_repr})"
        )

    def __len__(self):
        return self.ligand.pos.shape[0]

    def to(self, *args, **kwargs):
        """
        Transfer all tensors in the batch to the specified device and handle additional arguments.

        Parameters:
        ----------
        *args : list
            Positional arguments to pass to the `to()` function.
        **kwargs : dict
            Keyword arguments to pass to the `to()` function.
        """
        # Transfer ligand tensors to device
        self.ligand.x = self.ligand.x.to(*args, **kwargs)
        self.ligand.pos = self.ligand.pos.to(*args, **kwargs)
        if self.ligand.true_pos is not None:
            self.ligand.true_pos = self.ligand.true_pos.to(*args, **kwargs)
        self.ligand.orig_pos = self.ligand.orig_pos.to(*args, **kwargs)
        self.ligand.orig_pos_before_augm = self.ligand.orig_pos_before_augm.to(
            *args, **kwargs)
        self.ligand.random_pos = self.ligand.random_pos.to(*args, **kwargs)
        self.ligand.rotatable_bonds = self.ligand.rotatable_bonds.to(
            *args, **kwargs)
        self.ligand.init_tr = self.ligand.init_tr.to(*args, **kwargs)
        if self.ligand.pred_tr is not None:
            self.ligand.pred_tr = self.ligand.pred_tr.to(*args, **kwargs)
        self.ligand.init_rot = self.ligand.init_rot.to(*args, **kwargs)
        self.ligand.init_tor = self.ligand.init_tor.to(*args, **kwargs)
        self.ligand.init_tor_ext = self.ligand.init_tor_ext.to(*args, **kwargs)
        self.ligand.final_tr = self.ligand.final_tr.to(*args, **kwargs)
        self.ligand.final_rot = self.ligand.final_rot.to(*args, **kwargs)
        self.ligand.final_tor = self.ligand.final_tor.to(*args, **kwargs)
        self.ligand.final_tor_ext = self.ligand.final_tor_ext.to(
            *args, **kwargs)
        self.ligand.num_rotatable_bonds = self.ligand.num_rotatable_bonds.to(
            *args, **kwargs)
        self.ligand.num_atoms = self.ligand.num_atoms.to(*args, **kwargs)
        self.ligand.t = self.ligand.t.to(*args, **kwargs)
        self.ligand.bond_periods = self.ligand.bond_periods.to(*args, **kwargs)
        self.ligand.rmsd = self.ligand.rmsd.to(*args, **kwargs)
        self.ligand.stage_num = self.ligand.stage_num.to(*args, **kwargs)
        if self.ligand.mask_rotate is not None:
            self.ligand.mask_rotate = [
                mr.to(*args, **kwargs) for mr in self.ligand.mask_rotate]

        self.ligand.rotatable_bonds_ext = self.ligand.rotatable_bonds_ext.to(
            *args, **kwargs)
        self.ligand.num_rotatable_bonds_ext = self.ligand.num_rotatable_bonds_ext.to(
            *args, **kwargs)

        # Transfer protein tensors to device
        self.protein.x = self.protein.x.to(*args, **kwargs)
        self.protein.pos = self.protein.pos.to(*args, **kwargs)
        self.protein.seq = self.protein.seq.to(*args, **kwargs)
        if self.protein.all_atom_names is not None:
            self.protein.all_atom_names = self.protein.all_atom_names.to(*args, **kwargs)
        if self.protein.all_atom_pos is not None:
            self.protein.all_atom_pos = self.protein.all_atom_pos.to(*args, **kwargs)
        if self.protein.mask_all_atom_residue is not None:
            self.protein.mask_all_atom_residue = self.protein.mask_all_atom_residue.to(*args, **kwargs)
        # if self.protein.all_atom_residue_ids is not None:
            # self.protein.all_atom_residue_ids = self.protein.all_atom_residue_ids.to(*args, **kwargs)
        self.protein.all_atom_pos = self.protein.all_atom_pos.to(*args, **kwargs)
        self.protein.mask_all_atom_residue = self.protein.mask_all_atom_residue.to(*args, **kwargs)

        # Transfer additional mask tensors to device
        self.ligand.is_padded_mask = self.ligand.is_padded_mask.to(
            *args, **kwargs)
        self.protein.is_padded_mask = self.protein.is_padded_mask.to(
            *args, **kwargs)

        self.original_augm_rot = self.original_augm_rot.to(*args, **kwargs)

        self.allbonds_mask = self.allbonds_mask.to(*args, **kwargs)
        self.rotbonds_mask = self.rotbonds_mask.to(*args, **kwargs)

        return self


@dataclass
class Ligand:
    """
    Ligand data structure.

    Attributes:
    ----------
    x : np.ndarray
        Feature matrix of the ligand, shape: (num_atoms, feature_dim).
    pos : np.ndarray
        Position matrix of the ligand, shape: (num_atoms, 3).
    rot : np.ndarray
        Current rotation matrices for the ligand, shape: (3, 3).
    orig_pos : np.ndarray
        Original position matrix of the ligand after pocket augmentations and rotations, shape: (num_atoms, 3).
    orig_pos_before_augm : np.ndarray
        Original position matrix of the ligand before pocket augmentations and rotations, shape: (num_atoms, 3).
    mask_rotate : np.ndarray
        Mask indicating which atoms to rotate for each bond, shape: (num_rotatable_bonds, num_atoms).
    rotatable_bonds : np.ndarray
        Rotatable bonds in the ligand, shape: (num_rotatable_bonds, 2).
    init_tr : np.ndarray
        Initial translation vectors for the ligand, shape: (3).
    pred_tr: np.ndarray
        Predicted translation vectors for the ligand from previous model, shape: (3).
    init_rot : np.ndarray
        Initial rotation matrices for the ligand, shape: (3, 3).
    init_tor : np.ndarray
        Initial torsion angles for the ligand, shape: (num_rotatable_bonds, ).
    final_tr : np.ndarray
        Ground-truth translation vectors for the ligand, shape: (3).
    final_rot : np.ndarray
        Ground-truth rotation matrices for the ligand, shape: (3, 3).
    final_tor : np.ndarray
        Ground-truth torsion angles for the ligand, shape: (num_rotatable_bonds, ).
    orig_mol : Chem.Mol
        Original RDKit molecule object.
    t : float
        Optional float value.
    """
    x: np.ndarray = None
    pos: np.ndarray = None
    rot: np.ndarray = None
    orig_pos: np.ndarray = None
    orig_pos_before_augm: np.ndarray = None
    predicted_pos: np.ndarray = None
    mask_rotate: np.ndarray = None
    rotatable_bonds: np.ndarray = None
    bond_periods: np.ndarray = None
    init_tr: np.ndarray = None
    pred_tr: np.ndarray = None
    init_rot: np.ndarray = None
    init_tor: np.ndarray = None
    final_tr: np.ndarray = None
    final_rot: np.ndarray = None
    final_tor: np.ndarray = None
    orig_mol: Chem.Mol = None
    t: float = None
    rmsd: float = None
    stage_num: int = None
    true_pos: np.ndarray = None

    rotatable_bonds_ext: Bonds = None  # Updated to store rotatable bonds

    def __repr__(self):
        return (f'Ligand(\n'
                f'  x: {self._format_shape(self.x)},\n'
                f'  pos: {self._format_shape(self.pos)},\n'
                f'  rot: {self._format_shape(self.rot)},\n'
                f'  orig_pos: {self._format_shape(self.orig_pos)},\n'
                f'  orig_pos_before_augm: {self._format_shape(self.orig_pos_before_augm)},\n'
                f'  mask_rotate: {self._format_shape(self.mask_rotate)},\n'
                f'  rotatable_bonds: {self._format_shape(self.rotatable_bonds)},\n'
                f'  bond_periods: {self._format_shape(self.bond_periods)},\n'
                f'  init_tr: {self._format_shape(self.init_tr)},\n'
                f'  pred_tr: {self._format_shape(self.pred_tr)},\n'
                f'  init_rot: {self._format_shape(self.init_rot)},\n'
                f'  init_tor: {self._format_shape(self.init_tor)},\n'
                f'  final_tr: {self._format_shape(self.final_tr)},\n'
                f'  final_rot: {self._format_shape(self.final_rot)},\n'
                f'  final_tor: {self._format_shape(self.final_tor)},\n'
                f'  orig_mol: {self._format_shape(self.orig_mol)},\n'
                f'  t: {self.t},\n'
                f'  rmsd: {self.rmsd},\n'
                f'  true_pos: {self._format_shape(self.true_pos)}\n'
                f')')

    def _format_shape(self, obj):
        if obj is None:
            return "None"
        if isinstance(obj, np.ndarray):
            return f"np.ndarray{obj.shape}"
        if isinstance(obj, torch.Tensor):
            return f"torch.Size({list(obj.shape)})"
        return str(type(obj))

    def set_ground_truth_values(self):
        self.orig_pos = np.copy(self.pos)
        self.final_tr = self.pos.mean(0).astype(np.float32).reshape(1, 3)

    def sample_first_neighbor_of_rotatable_bonds(self):
        self.rotatable_bonds_ext.neighbor_of_start = np.array([self.rotatable_bonds_ext.all_neighbors_of_start[i][0]
                                                               for i in range(len(self.rotatable_bonds))])
        self.rotatable_bonds_ext.neighbor_of_end = np.array([self.rotatable_bonds_ext.all_neighbors_of_end[i][0]
                                                             for i in range(len(self.rotatable_bonds))])

    def randomly_sample_neighbors_of_rotatable_bonds(self):
        self.rotatable_bonds_ext.neighbor_of_start = np.array([np.random.choice(self.rotatable_bonds_ext.all_neighbors_of_start[i])
                                                               for i in range(len(self.rotatable_bonds))])
        self.rotatable_bonds_ext.neighbor_of_end = np.array([np.random.choice(self.rotatable_bonds_ext.all_neighbors_of_end[i])
                                                             for i in range(len(self.rotatable_bonds))])

    def randomly_mask_atoms(self, mask_ratio):

        if mask_ratio < 1e-6:
            return

        def mask_bonds(bonds, kept_atom_indices, index_mapping, mask=None, is_rotatable_bonds=False):
            # Only keep rotatable bonds where both atoms are kept
            if mask is None:
                mask = np.isin(bonds.start, kept_atom_indices) & \
                    np.isin(bonds.end, kept_atom_indices)

            bonds.start = bonds.start[mask]
            bonds.end = bonds.end[mask]
            bonds.bond_type = bonds.bond_type[mask]
            if bonds.all_neighbors_of_start is not None:
                bonds.all_neighbors_of_start = [neighbors for to_keep, neighbors in zip(
                    mask, bonds.all_neighbors_of_start) if to_keep]
                bonds.all_neighbors_of_end = [neighbors for to_keep, neighbors in zip(
                    mask, bonds.all_neighbors_of_end) if to_keep]
            if bonds.angles is not None:
                bonds.angles = bonds.angles[mask]

            if bonds.bond_periods is not None:
                bonds.bond_periods = bonds.bond_periods[mask]
            bonds.length = bonds.length[mask]
            bonds.is_conjugated = bonds.is_conjugated[mask]
            bonds.is_in_ring = bonds.is_in_ring[mask]
            bonds.is_aromatic = bonds.is_aromatic[mask]
            if bonds.is_rotatable is not None:
                bonds.is_rotatable = bonds.is_rotatable[mask]

            if is_rotatable_bonds:
                bonds.mask_rotate = bonds.mask_rotate[mask][:,
                                                            kept_atom_indices]

            # Directly map old indices to new ones using the mapping array
            bonds.start = index_mapping[bonds.start]
            bonds.end = index_mapping[bonds.end]
            if bonds.all_neighbors_of_start is not None:
                bonds.all_neighbors_of_start = [
                    index_mapping[neighbors] for neighbors in bonds.all_neighbors_of_start]
                bonds.all_neighbors_of_end = [
                    index_mapping[neighbors] for neighbors in bonds.all_neighbors_of_end]

        while True:
            keep_mask = np.random.choice([True, False], size=len(
                self.pos), p=[1 - mask_ratio, mask_ratio])
            if keep_mask.sum() > 0:
                break

        if len(self.rotatable_bonds) > 0:
            kept_atom_indices = np.where(keep_mask)[0]
            masked_atom_indices = np.where(~keep_mask)[0]

            if len(self.rotatable_bonds) > 0:
                # Only keep rotatable bonds where both atoms are kept
                rot_mask = np.isin(self.rotatable_bonds,
                                   kept_atom_indices).all(axis=1)

                for i in range(len(self.rotatable_bonds)):
                    if rot_mask[i]:
                        if np.isin(self.rotatable_bonds_ext.all_neighbors_of_end[i], masked_atom_indices).all():
                            # unmask one random neighbor of end
                            keep_mask[np.random.choice(
                                self.rotatable_bonds_ext.all_neighbors_of_end[i])] = True
                        if np.isin(self.rotatable_bonds_ext.all_neighbors_of_start[i], masked_atom_indices).all():
                            # unmask one random neighbor of start
                            keep_mask[np.random.choice(
                                self.rotatable_bonds_ext.all_neighbors_of_start[i])] = True

            # Recalculate kept and masked atom indices
            kept_atom_indices = np.where(keep_mask)[0]
            masked_atom_indices = np.where(~keep_mask)[0]

            # Create a mapping array where index = old idx, value = new idx
            # Initialize with -1 to identify atoms that were removed
            index_mapping = np.full(len(keep_mask), -1)
            index_mapping[kept_atom_indices] = np.arange(
                len(kept_atom_indices))

            if len(self.rotatable_bonds) > 0:
                def filter_neighbors(neighbor_list, masked_atom_indices, kept_bond_mask):
                    return [neighbors[~np.isin(neighbors, masked_atom_indices)] if bond_is_unmasked else neighbors
                            for bond_is_unmasked, neighbors in zip(kept_bond_mask, neighbor_list)]

                # recompute all_neighbors after masking
                self.rotatable_bonds_ext.all_neighbors_of_start = filter_neighbors(self.rotatable_bonds_ext.all_neighbors_of_start,
                                                                                   masked_atom_indices, rot_mask)
                self.rotatable_bonds_ext.all_neighbors_of_end = filter_neighbors(self.rotatable_bonds_ext.all_neighbors_of_end,
                                                                                 masked_atom_indices, rot_mask)

        if len(self.rotatable_bonds) > 0:
            # Only keep rotatable bonds where both atoms are kept
            self.rotatable_bonds = self.rotatable_bonds[rot_mask]

            # Directly map old indices to new ones using the mapping array
            self.rotatable_bonds = index_mapping[self.rotatable_bonds]
            self.mask_rotate = self.mask_rotate[rot_mask][:, kept_atom_indices]
            self.init_tor = self.init_tor[rot_mask]
            self.bond_periods = self.bond_periods[rot_mask]

            mask_bonds(self.rotatable_bonds_ext, kept_atom_indices,
                       index_mapping, mask=rot_mask, is_rotatable_bonds=True)

        self.x = self.x[keep_mask]
        self.pos = self.pos[keep_mask]
        self.orig_pos = self.orig_pos[keep_mask]
        self.orig_pos_before_augm = self.orig_pos_before_augm[keep_mask]
        if self.true_pos is not None:
            self.true_pos = self.true_pos[keep_mask]
        if self.predicted_pos is not None:
            self.predicted_pos = self.predicted_pos[keep_mask]


@dataclass
class Protein:
    """
    Protein data structure.

    Attributes:
    ----------
    x : np.ndarray
        Feature matrix of the protein, shape: (num_residues, feature_dim).
    pos : np.ndarray
        Position matrix of the protein, shape: (num_residues, 3).
    all_atom_pos : np.ndarray
        Position matrix of all protein atoms, shape: (num_protein_atoms, 3).
    all_atom_residue_ids : np.ndarray
        Residue ids of all protein atoms, shape: (num_protein_atoms, ).
    all_atom_names : np.ndarray
        Names of all protein atoms, shape: (num_protein_atoms, ).
    seq : np.ndarray
        Amino acid sequence of a protein, shape: (num_residues).
    name : str
        PDB id of a protein.
    """
    x: np.ndarray = None
    pos: np.ndarray = None
    all_atom_pos: np.ndarray = None
    all_atom_residue_ids: np.ndarray = None
    all_atom_names: np.ndarray = None
    seq: np.ndarray = None
    name: str = None
    full_protein_center: np.ndarray = None
    chain_lengths: List[int] = None

    def __repr__(self):
        return (f'Protein(\n'
                f'  name: {self.name},\n'
                f'  x: {self._format_shape(self.x)},\n'
                f'  pos: {self._format_shape(self.pos)},\n'
                f'  all_atom_pos: {self._format_shape(self.all_atom_pos)},\n'
                f'  seq: {self._format_shape(self.seq)},\n'
                f'  full_protein_center: {self._format_shape(self.full_protein_center)},\n'
                f'  chain_lengths: {self._format_shape(self.chain_lengths)},\n'
                f')')

    def _format_shape(self, obj):
        if obj is None:
            return "None"
        if isinstance(obj, np.ndarray):
            return f"np.ndarray{obj.shape}"
        if isinstance(obj, torch.Tensor):
            return f"torch.Size({list(obj.shape)})"
        return str(type(obj))

    def randomly_mask_residues(self, mask_ratio):
        if mask_ratio < 1e-6:
            return
        while True:
            keep_mask = np.random.choice([True, False], size=len(
                self.seq), p=[1 - mask_ratio, mask_ratio])
            if keep_mask.sum() > 0:
                break

        self.x = self.x[keep_mask]
        self.pos = self.pos[keep_mask]
        self.seq = self.seq[keep_mask]

        if self.all_atom_pos is not None:
            kept_residues = np.where(keep_mask)[0]
            kept_mask_all_atom = np.isin(
                self.all_atom_residue_ids, kept_residues)
            self.all_atom_pos = self.all_atom_pos[kept_mask_all_atom]
            self.all_atom_residue_ids = self.all_atom_residue_ids[kept_mask_all_atom]
            self.all_atom_names = self.all_atom_names[kept_mask_all_atom]


@dataclass
class Complex:
    """
    Complex data structure containing a ligand and a protein.

    Attributes:
    ----------
    name : str
        Name of the complex.
    ligand : Ligand
        Ligand object.
    protein : Protein
        Protein object.
    original_augm_rot: np.ndarray, shape (3, 3)
        Rotation applied to the whole complex in dataset getitem method.
    """
    name: str = ''
    ligand: Ligand = None
    protein: Protein = None
    original_augm_rot: np.ndarray = None

    def __repr__(self):
        return (f'Complex(\n'
                f'  name: {self.name},\n'
                f'  ligand: {repr(self.ligand)},\n'
                f'  protein: {repr(self.protein)},\n'
                f'  original_augm_rot: {self.ligand._format_shape(self.original_augm_rot)},\n'
                f')')

    def shift_to_protein_center(self):
        protein_center = self.protein.pos.mean(axis=0).reshape(1, 3)
        self.protein.pos -= protein_center
        self.ligand.pos -= protein_center
        if self.ligand.pred_tr is not None:
            self.ligand.pred_tr -= protein_center
        if self.ligand.predicted_pos is not None:
            self.ligand.predicted_pos -= protein_center
        if self.ligand.orig_pos_before_augm is not None:
            self.ligand.orig_pos_before_augm -= protein_center
        if self.ligand.true_pos is not None:
            self.ligand.true_pos -= protein_center
        self.protein.full_protein_center += protein_center

    def randomly_mask_complex(self, ligand_mask_ratio, protein_mask_ratio):
        self.ligand.randomly_mask_atoms(ligand_mask_ratio)
        self.protein.randomly_mask_residues(protein_mask_ratio)

    def set_ground_truth_values(self):
        self.ligand.set_ground_truth_values()
