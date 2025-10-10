import torch
from dataclasses import dataclass, field
import numpy as np
from rdkit import Chem
from typing import List, Optional


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
    init_tor : torch.Tensor
        Initial torsion angles for the batch, shape: (batch_size,).
    final_tr : torch.Tensor
        Ground-truth translation vectors for the batch, shape: (batch_size, 3).
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
    init_tor: torch.Tensor = torch.empty(0)
    final_tr: torch.Tensor = torch.empty(0)
    pred_tr: torch.Tensor = torch.empty(0)
    num_atoms: torch.Tensor = torch.empty(0)
    tor_ptr: List[int] = None
    orig_mols: List[Chem.Mol] = None
    is_padded_mask: torch.Tensor = torch.empty(0)
    t: torch.Tensor = torch.empty(0)
    rmsd: torch.Tensor = torch.empty(0)
    num_rotatable_bonds: torch.Tensor = torch.empty(0)
    bond_periods: torch.Tensor = torch.empty(0)


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
            f"  init_tor: shape={self.ligand.init_tor.shape}, dtype={self.ligand.init_tor.dtype}\n"
            f"  final_tr: shape={self.ligand.final_tr.shape}, dtype={self.ligand.final_tr.dtype}\n"
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
        self.ligand.orig_pos = self.ligand.orig_pos.to(*args, **kwargs)
        self.ligand.orig_pos_before_augm = self.ligand.orig_pos_before_augm.to(
            *args, **kwargs)
        self.ligand.random_pos = self.ligand.random_pos.to(*args, **kwargs)
        self.ligand.rotatable_bonds = self.ligand.rotatable_bonds.to(
            *args, **kwargs)
        self.ligand.init_tr = self.ligand.init_tr.to(*args, **kwargs)
        if self.ligand.pred_tr is not None:
            self.ligand.pred_tr = self.ligand.pred_tr.to(*args, **kwargs)
        self.ligand.init_tor = self.ligand.init_tor.to(*args, **kwargs)
        self.ligand.final_tr = self.ligand.final_tr.to(*args, **kwargs)
        self.ligand.num_rotatable_bonds = self.ligand.num_rotatable_bonds.to(
            *args, **kwargs)
        self.ligand.num_atoms = self.ligand.num_atoms.to(*args, **kwargs)
        self.ligand.t = self.ligand.t.to(*args, **kwargs)
        self.ligand.bond_periods = self.ligand.bond_periods.to(*args, **kwargs)
        self.ligand.rmsd = self.ligand.rmsd.to(*args, **kwargs)
        if self.ligand.mask_rotate is not None:
            self.ligand.mask_rotate = [
                mr.to(*args, **kwargs) for mr in self.ligand.mask_rotate]

        # Transfer protein tensors to device
        self.protein.x = self.protein.x.to(*args, **kwargs)
        self.protein.pos = self.protein.pos.to(*args, **kwargs)
        self.protein.seq = self.protein.seq.to(*args, **kwargs)

        # Transfer additional mask tensors to device
        self.ligand.is_padded_mask = self.ligand.is_padded_mask.to(
            *args, **kwargs)
        self.protein.is_padded_mask = self.protein.is_padded_mask.to(
            *args, **kwargs)

        self.original_augm_rot = self.original_augm_rot.to(*args, **kwargs)
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
    init_tor : np.ndarray
        Initial torsion angles for the ligand, shape: (num_rotatable_bonds, ).
    final_tr : np.ndarray
        Ground-truth translation vectors for the ligand, shape: (3).
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
    init_tor: np.ndarray = None
    final_tr: np.ndarray = None
    orig_mol: Chem.Mol = None
    t: float = None
    rmsd: float = None

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
                f'  init_tor: {self._format_shape(self.init_tor)},\n'
                f'  final_tr: {self._format_shape(self.final_tr)},\n'
                f'  orig_mol: {self._format_shape(self.orig_mol)},\n'
                f'  t: {self.t},\n'
                f'  rmsd: {self.rmsd},\n'
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
    seq: np.ndarray = None
    name: str = None
    full_protein_center: np.ndarray = None
    chain_lengths: List[int] = None
    all_atom_names: np.ndarray = None

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

    def set_ground_truth_values(self):
        self.ligand.set_ground_truth_values()
