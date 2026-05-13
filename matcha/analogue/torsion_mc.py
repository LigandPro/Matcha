from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from .constrained_embed import align_mol_to_template_core, core_rmsd, mol_positions
from .mcs import MCSMapping

_ROTATABLE_SMARTS = "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]"


def _ensure_ring_info(mol: Chem.Mol) -> None:
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        pass


@dataclass
class TorsionMCResult:
    poses: list[Chem.Mol]
    warnings: list[str] = field(default_factory=list)


def _rotatable_bonds(mol: Chem.Mol, fixed_atoms: set[int]) -> list[tuple[int, int]]:
    _ensure_ring_info(mol)
    patt = Chem.MolFromSmarts(_ROTATABLE_SMARTS)
    if patt is None:
        return []
    _ensure_ring_info(patt)
    bonds: list[tuple[int, int]] = []
    for a, b in mol.GetSubstructMatches(patt):
        bond = mol.GetBondBetweenAtoms(int(a), int(b))
        if bond is None:
            continue
        # Avoid rotating bonds fully inside the anchored MCS core.
        if int(a) in fixed_atoms and int(b) in fixed_atoms:
            continue
        bonds.append((int(a), int(b)))
    return bonds


def _dihedral_atoms_for_bond(mol: Chem.Mol, a: int, b: int) -> tuple[int, int, int, int] | None:
    a_nei = [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() != b]
    b_nei = [n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != a]
    if not a_nei or not b_nei:
        return None
    return int(a_nei[0]), int(a), int(b), int(b_nei[0])


def _energy(mol: Chem.Mol) -> float:
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
            if ff is not None:
                return float(ff.CalcEnergy())
    except Exception:
        pass
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
        if ff is not None:
            return float(ff.CalcEnergy())
    except Exception:
        pass
    return 0.0


def _internal_clash_count(mol: Chem.Mol, threshold: float = 1.15) -> int:
    if mol.GetNumConformers() == 0:
        return 999999
    pos = mol_positions(mol)
    clashes = 0
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            # Skip directly bonded and 1-3 bonded pairs.
            if mol.GetBondBetweenAtoms(i, j) is not None:
                continue
            path = Chem.rdmolops.GetShortestPath(mol, i, j)
            if path and len(path) <= 3:
                continue
            if float(np.linalg.norm(pos[i] - pos[j])) < threshold:
                clashes += 1
    return int(clashes)


def _score(mol: Chem.Mol, template: Chem.Mol, mapping: MCSMapping, core_weight: float = 25.0) -> float:
    return float(_energy(mol) + core_weight * core_rmsd(mol, template, mapping) + 50.0 * _internal_clash_count(mol))


def torsional_mc_refine(
    template: Chem.Mol,
    seed_poses: Iterable[Chem.Mol],
    mapping: MCSMapping,
    *,
    n_steps: int = 250,
    keep: int = 64,
    temperature: float = 300.0,
    max_delta_deg: float = 45.0,
    random_seed: int = 777,
) -> TorsionMCResult:
    """Small constrained torsional Monte Carlo for analogue seed diversification.

    This is intentionally lightweight and deterministic under ``random_seed``. It
    perturbs rotatable bonds outside the MCS core, re-aligns the MCS after each
    accepted proposal, and keeps the lowest-strain / lowest-core-RMSD poses.
    """

    rng = random.Random(int(random_seed))
    warnings: list[str] = []
    fixed_atoms = set(mapping.ligand_atom_indices)
    candidates: list[tuple[float, Chem.Mol]] = []
    beta = 1.0 / max(0.00198720425864083 * float(temperature), 1e-6)

    for seed_idx, seed in enumerate(seed_poses):
        if seed.GetNumConformers() == 0:
            continue
        current = align_mol_to_template_core(seed, template, mapping)
        bonds = _rotatable_bonds(current, fixed_atoms)
        if not bonds or int(n_steps) <= 0:
            candidates.append((_score(current, template, mapping), copy.deepcopy(current)))
            continue

        current_score = _score(current, template, mapping)
        best = copy.deepcopy(current)
        best_score = current_score
        candidates.append((current_score, copy.deepcopy(current)))

        for _ in range(int(n_steps)):
            bond = rng.choice(bonds)
            atoms = _dihedral_atoms_for_bond(current, *bond)
            if atoms is None:
                continue
            proposal = copy.deepcopy(current)
            try:
                cur_angle = rdMolTransforms.GetDihedralDeg(proposal.GetConformer(), *atoms)
                delta = rng.uniform(-float(max_delta_deg), float(max_delta_deg))
                rdMolTransforms.SetDihedralDeg(proposal.GetConformer(), *atoms, float(cur_angle + delta))
            except Exception:
                continue
            proposal = align_mol_to_template_core(proposal, template, mapping)
            proposal_score = _score(proposal, template, mapping)
            accept = proposal_score <= current_score
            if not accept:
                prob = math.exp(-min(700.0, beta * (proposal_score - current_score)))
                accept = rng.random() < prob
            if accept:
                current = proposal
                current_score = proposal_score
                candidates.append((current_score, copy.deepcopy(current)))
                if current_score < best_score:
                    best = copy.deepcopy(current)
                    best_score = current_score
        candidates.append((best_score, best))

    candidates.sort(key=lambda item: item[0])
    poses: list[Chem.Mol] = []
    seen: list[np.ndarray] = []
    for score, mol in candidates:
        pos = mol_positions(mol)
        duplicate = False
        for old in seen:
            if old.shape == pos.shape and float(np.sqrt(np.mean(np.sum((old - pos) ** 2, axis=1)))) < 0.10:
                duplicate = True
                break
        if duplicate:
            continue
        mol.SetProp("analogue_torsion_mc_score", f"{float(score):.6f}")
        poses.append(mol)
        seen.append(pos)
        if len(poses) >= int(keep):
            break

    if not poses:
        warnings.append("torsion_mc_generated_no_poses")
    return TorsionMCResult(poses=poses, warnings=warnings)
