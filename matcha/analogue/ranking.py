from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

from .constrained_embed import core_rmsd, mol_positions
from .mcs import MCSMapping
from .standardize import formal_charge


@dataclass
class PoseQC:
    ligand_id: str
    pose_index: int
    core_rmsd: float
    whole_centroid_distance: float
    internal_clash_count: int
    strain_energy: float | None
    mcs_atoms: int
    mcs_fraction_ligand: float
    mcs_fraction_template: float
    rank_score: float
    status: str
    fep_ready: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ligand_id": self.ligand_id,
            "pose_index": int(self.pose_index),
            "core_rmsd": float(self.core_rmsd),
            "whole_centroid_distance": float(self.whole_centroid_distance),
            "internal_clash_count": int(self.internal_clash_count),
            "strain_energy": None if self.strain_energy is None else float(self.strain_energy),
            "mcs_atoms": int(self.mcs_atoms),
            "mcs_fraction_ligand": float(self.mcs_fraction_ligand),
            "mcs_fraction_template": float(self.mcs_fraction_template),
            "rank_score": float(self.rank_score),
            "status": self.status,
            "fep_ready": bool(self.fep_ready),
            "warnings": list(self.warnings),
        }


def internal_clash_count(mol: Chem.Mol, *, threshold: float = 1.15) -> int:
    if mol is None or mol.GetNumConformers() == 0:
        return 999999
    pos = mol_positions(mol)
    clashes = 0
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            if mol.GetBondBetweenAtoms(i, j) is not None:
                continue
            path = Chem.rdmolops.GetShortestPath(mol, i, j)
            if path and len(path) <= 3:
                continue
            if float(np.linalg.norm(pos[i] - pos[j])) < threshold:
                clashes += 1
    return int(clashes)


def centroid_distance(mol: Chem.Mol, template: Chem.Mol) -> float:
    if mol.GetNumConformers() == 0 or template.GetNumConformers() == 0:
        return float("inf")
    return float(np.linalg.norm(mol_positions(mol).mean(axis=0) - mol_positions(template).mean(axis=0)))


def estimate_mmff_or_uff_energy(mol: Chem.Mol) -> float | None:
    from rdkit.Chem import AllChem

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
        return None
    return None


def evaluate_pose(
    ligand_id: str,
    pose_index: int,
    mol: Chem.Mol,
    template: Chem.Mol,
    mapping: MCSMapping,
    *,
    core_rmsd_cutoff: float = 1.0,
    min_mcs_fraction: float = 0.35,
    max_internal_clashes: int = 0,
) -> PoseQC:
    warnings: list[str] = []
    c_rmsd = core_rmsd(mol, template, mapping)
    cent = centroid_distance(mol, template)
    clashes = internal_clash_count(mol)
    energy = estimate_mmff_or_uff_energy(mol)
    energy_term = 0.0 if energy is None else min(max(float(energy), -1000.0), 1000.0) * 0.01
    rank_score = (
        25.0 * min(c_rmsd, 20.0)
        + 4.0 * min(cent, 20.0)
        + 50.0 * float(clashes)
        + energy_term
        - 2.0 * float(mapping.quality_score)
    )

    status = "FEP_READY"
    if not mapping.ok:
        status = "FAILED_MAPPING"
        warnings.append("mapping_not_ok")
    if c_rmsd > float(core_rmsd_cutoff):
        status = "NEEDS_REVIEW" if status == "FEP_READY" else status
        warnings.append(f"core_rmsd_high:{c_rmsd:.3f}>{float(core_rmsd_cutoff):.3f}")
    if clashes > int(max_internal_clashes):
        status = "NEEDS_REVIEW" if status == "FEP_READY" else status
        warnings.append(f"internal_clashes:{clashes}>{int(max_internal_clashes)}")
    if mapping.fraction_ligand < float(min_mcs_fraction):
        status = "NEEDS_REVIEW" if status == "FEP_READY" else status
        warnings.append(f"mcs_fraction_ligand_low:{mapping.fraction_ligand:.3f}")
    if formal_charge(mol) != formal_charge(template):
        warnings.append(f"formal_charge_delta:{formal_charge(mol) - formal_charge(template)}")

    return PoseQC(
        ligand_id=ligand_id,
        pose_index=int(pose_index),
        core_rmsd=float(c_rmsd),
        whole_centroid_distance=float(cent),
        internal_clash_count=int(clashes),
        strain_energy=energy,
        mcs_atoms=int(mapping.num_atoms),
        mcs_fraction_ligand=float(mapping.fraction_ligand),
        mcs_fraction_template=float(mapping.fraction_template),
        rank_score=float(rank_score),
        status=status,
        fep_ready=status == "FEP_READY",
        warnings=warnings,
    )


def rank_poses(
    ligand_id: str,
    poses: list[Chem.Mol],
    template: Chem.Mol,
    mapping: MCSMapping,
    *,
    core_rmsd_cutoff: float = 1.0,
    min_mcs_fraction: float = 0.35,
) -> list[tuple[Chem.Mol, PoseQC]]:
    evaluated: list[tuple[Chem.Mol, PoseQC]] = []
    for idx, mol in enumerate(poses):
        qc = evaluate_pose(
            ligand_id,
            idx,
            mol,
            template,
            mapping,
            core_rmsd_cutoff=core_rmsd_cutoff,
            min_mcs_fraction=min_mcs_fraction,
        )
        mol.SetProp("analogue_ligand_id", ligand_id)
        mol.SetProp("analogue_pose_index", str(idx))
        mol.SetProp("analogue_core_rmsd", f"{qc.core_rmsd:.6f}")
        mol.SetProp("analogue_rank_score", f"{qc.rank_score:.6f}")
        mol.SetProp("analogue_status", qc.status)
        evaluated.append((mol, qc))
    evaluated.sort(key=lambda item: (0 if item[1].fep_ready else 1, item[1].rank_score))
    return evaluated
