from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from .mcs import MCSMapping


@dataclass
class ConformerGenerationResult:
    conformers: list[Chem.Mol]
    warnings: list[str] = field(default_factory=list)
    failures: int = 0


def mol_positions(mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
    return np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=float)


def set_mol_positions(mol: Chem.Mol, positions: np.ndarray, conf_id: int = 0) -> None:
    conf = mol.GetConformer(conf_id)
    for idx, xyz in enumerate(np.asarray(positions, dtype=float)):
        conf.SetAtomPosition(int(idx), Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def kabsch_align_positions(
    moving_positions: np.ndarray,
    moving_anchor_indices: Sequence[int],
    reference_positions: np.ndarray,
    reference_anchor_indices: Sequence[int],
) -> np.ndarray:
    """Rigidly align ``moving_positions`` to reference anchors using Kabsch."""

    moving_positions = np.asarray(moving_positions, dtype=float)
    reference_positions = np.asarray(reference_positions, dtype=float)
    moving_anchor = moving_positions[list(moving_anchor_indices)]
    reference_anchor = reference_positions[list(reference_anchor_indices)]
    if len(moving_anchor) == 0:
        return moving_positions

    cm = moving_anchor.mean(axis=0)
    cr = reference_anchor.mean(axis=0)
    a = moving_anchor - cm
    b = reference_anchor - cr
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    return (moving_positions - cm) @ r + cr


def align_mol_to_template_core(mol: Chem.Mol, template: Chem.Mol, mapping: MCSMapping) -> Chem.Mol:
    """Return a copy of ``mol`` rigidly aligned by the MCS core."""

    aligned = copy.deepcopy(mol)
    if not mapping.ok or aligned.GetNumConformers() == 0 or template.GetNumConformers() == 0:
        return aligned
    new_pos = kabsch_align_positions(
        mol_positions(aligned),
        mapping.ligand_atom_indices,
        mol_positions(template),
        mapping.template_atom_indices,
    )
    set_mol_positions(aligned, new_pos)
    return aligned


def core_rmsd(mol: Chem.Mol, template: Chem.Mol, mapping: MCSMapping) -> float:
    if not mapping.ok or mol.GetNumConformers() == 0 or template.GetNumConformers() == 0:
        return float("inf")
    pos = mol_positions(mol)[mapping.ligand_atom_indices]
    ref = mol_positions(template)[mapping.template_atom_indices]
    if len(pos) == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.sum((pos - ref) ** 2, axis=1))))


def _make_coord_map(template: Chem.Mol, mapping: MCSMapping) -> dict[int, Point3D]:
    coord_map: dict[int, Point3D] = {}
    conf = template.GetConformer()
    for tmpl_idx, lig_idx in mapping.template_to_ligand:
        p = conf.GetAtomPosition(int(tmpl_idx))
        coord_map[int(lig_idx)] = Point3D(float(p.x), float(p.y), float(p.z))
    return coord_map


def _set_coord_map(params, coord_map: dict[int, Point3D]) -> None:
    # RDKit exposes this either as SetCoordMap() or a writable coordMap field
    # depending on release.  Support both without making the package version-fragile.
    if hasattr(params, "SetCoordMap"):
        params.SetCoordMap(coord_map)
    else:  # pragma: no cover - version fallback
        params.coordMap = coord_map


def _set_embed_timeout(params, timeout_seconds: int | None) -> None:
    if timeout_seconds is None or not hasattr(params, "timeout"):
        return
    params.timeout = max(0, int(timeout_seconds))


def _mmff_or_uff_optimize(mol: Chem.Mol, conf_id: int) -> float | None:
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is not None:
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", confId=int(conf_id), maxIters=200)
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(conf_id))
            if ff is not None:
                return float(ff.CalcEnergy())
    except Exception:
        pass
    try:
        AllChem.UFFOptimizeMolecule(mol, confId=int(conf_id), maxIters=200)
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(conf_id))
        if ff is not None:
            return float(ff.CalcEnergy())
    except Exception:
        return None
    return None


def _single_conformer_copy(mol: Chem.Mol, conf_id: int) -> Chem.Mol:
    out = copy.deepcopy(mol)
    conf = mol.GetConformer(int(conf_id))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out


def _deduplicate_by_core_and_whole_rmsd(
    mols: list[Chem.Mol],
    template: Chem.Mol,
    mapping: MCSMapping,
    *,
    max_conformers: int,
    whole_rmsd_cutoff: float = 0.15,
) -> list[Chem.Mol]:
    kept: list[Chem.Mol] = []
    for mol in mols:
        if mol.GetNumConformers() == 0:
            continue
        pos = mol_positions(mol)
        duplicate = False
        for old in kept:
            old_pos = mol_positions(old)
            if pos.shape == old_pos.shape:
                rmsd = float(np.sqrt(np.mean(np.sum((pos - old_pos) ** 2, axis=1))))
                if rmsd < whole_rmsd_cutoff:
                    duplicate = True
                    break
        if duplicate:
            continue
        kept.append(mol)
        if len(kept) >= int(max_conformers):
            break
    kept.sort(key=lambda m: core_rmsd(m, template, mapping))
    return kept[: int(max_conformers)]


def generate_constrained_conformers(
    template: Chem.Mol,
    ligand: Chem.Mol,
    mapping: MCSMapping,
    *,
    n_conformers: int = 64,
    random_seed: int = 777,
    use_random_coords: bool = True,
    optimize: bool = True,
    deduplicate: bool = True,
    embed_timeout_seconds: int | None = 30,
) -> ConformerGenerationResult:
    """Generate analogue conformers whose MCS core is aligned to ``template``.

    The routine uses RDKit ETKDG with an MCS coordinate map when available, then
    rigidly aligns each resulting conformer back to the template core.  This gives
    Matcha stage-3 refinement a strong, FEP-like starting pose instead of a blind
    docking pose.
    """

    warnings: list[str] = []
    if not mapping.ok:
        return ConformerGenerationResult([], ["mcs_not_ok"], failures=1)
    if template.GetNumConformers() == 0:
        return ConformerGenerationResult([], ["template_has_no_conformer"], failures=1)

    work = copy.deepcopy(ligand)
    try:
        work = Chem.AddHs(work, addCoords=True)
    except Exception:
        warnings.append("add_hs_failed")

    work.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useRandomCoords = bool(use_random_coords)
    params.clearConfs = True
    _set_embed_timeout(params, embed_timeout_seconds)
    try:
        _set_coord_map(params, _make_coord_map(template, mapping))
    except Exception as exc:  # pragma: no cover - RDKit-version dependent
        warnings.append(f"coord_map_failed:{type(exc).__name__}")

    conf_ids: list[int] = []
    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(work, numConfs=int(n_conformers), params=params))
    except Exception as exc:
        warnings.append(f"constrained_embed_failed:{type(exc).__name__}")

    if not conf_ids:
        # Fallback: unconstrained ETKDG followed by explicit MCS alignment.  This
        # often rescues cases where coordMap overconstrains macrocycles/linkers.
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = int(random_seed)
            params.useRandomCoords = True
            _set_embed_timeout(params, embed_timeout_seconds)
            conf_ids = list(AllChem.EmbedMultipleConfs(work, numConfs=max(1, int(n_conformers)), params=params))
            warnings.append("used_unconstrained_embed_fallback")
        except Exception as exc:  # pragma: no cover - RDKit-version dependent
            return ConformerGenerationResult([], warnings + [f"embed_fallback_failed:{type(exc).__name__}"], failures=1)

    out: list[Chem.Mol] = []
    for conf_id in conf_ids:
        if int(conf_id) < 0:
            continue
        if optimize:
            _mmff_or_uff_optimize(work, int(conf_id))
        mol_conf = _single_conformer_copy(work, int(conf_id))
        try:
            mol_conf = Chem.RemoveHs(mol_conf, sanitize=False)
        except Exception:
            pass
        mol_conf = align_mol_to_template_core(mol_conf, template, mapping)
        mol_conf.SetProp("analogue_core_rmsd", f"{core_rmsd(mol_conf, template, mapping):.6f}")
        out.append(mol_conf)

    if deduplicate:
        out = _deduplicate_by_core_and_whole_rmsd(
            out,
            template,
            mapping,
            max_conformers=max(1, int(n_conformers)),
        )

    return ConformerGenerationResult(out, warnings, failures=max(0, int(n_conformers) - len(out)))
