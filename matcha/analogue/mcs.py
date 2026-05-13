from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS

from .standardize import heavy_atom_count


@dataclass
class MCSMapping:
    """Template↔ligand atom mapping used by analogue docking and RBFE export."""

    template_to_ligand: list[tuple[int, int]]
    smarts: str = ""
    level: str = "failed"
    num_atoms: int = 0
    num_bonds: int = 0
    fraction_template: float = 0.0
    fraction_ligand: float = 0.0
    quality_score: float = 0.0
    warnings: list[str] = field(default_factory=list)
    status: str = "failed"

    @property
    def ligand_to_template(self) -> list[tuple[int, int]]:
        return [(lig_idx, tmpl_idx) for tmpl_idx, lig_idx in self.template_to_ligand]

    @property
    def ligand_atom_indices(self) -> list[int]:
        return [lig_idx for _, lig_idx in self.template_to_ligand]

    @property
    def template_atom_indices(self) -> list[int]:
        return [tmpl_idx for tmpl_idx, _ in self.template_to_ligand]

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.num_atoms > 0

    def to_dict(self) -> dict:
        return {
            "template_to_ligand": [[int(a), int(b)] for a, b in self.template_to_ligand],
            "ligand_to_template": [[int(a), int(b)] for a, b in self.ligand_to_template],
            "smarts": self.smarts,
            "level": self.level,
            "num_atoms": int(self.num_atoms),
            "num_bonds": int(self.num_bonds),
            "fraction_template": float(self.fraction_template),
            "fraction_ligand": float(self.fraction_ligand),
            "quality_score": float(self.quality_score),
            "warnings": list(self.warnings),
            "status": self.status,
        }


def _enum_or_default(enum_cls, name: str, default):
    try:
        return getattr(enum_cls, name)
    except AttributeError:  # pragma: no cover - RDKit-version compatibility
        return default


def _iter_mcs_levels():
    compare_elements = _enum_or_default(rdFMCS.AtomCompare, "CompareElements", rdFMCS.AtomCompare.CompareAny)
    compare_any_atom = rdFMCS.AtomCompare.CompareAny
    compare_order = _enum_or_default(rdFMCS.BondCompare, "CompareOrder", rdFMCS.BondCompare.CompareAny)
    compare_any_bond = rdFMCS.BondCompare.CompareAny

    return [
        {
            "name": "strict",
            "atomCompare": compare_elements,
            "bondCompare": compare_order,
            "ringMatchesRingOnly": True,
            "completeRingsOnly": True,
            "matchChiralTag": True,
            "matchValences": True,
        },
        {
            "name": "stereo_relaxed",
            "atomCompare": compare_elements,
            "bondCompare": compare_order,
            "ringMatchesRingOnly": True,
            "completeRingsOnly": True,
            "matchChiralTag": False,
            "matchValences": True,
        },
        {
            "name": "aromatic_bond_relaxed",
            "atomCompare": compare_elements,
            "bondCompare": compare_any_bond,
            "ringMatchesRingOnly": True,
            "completeRingsOnly": True,
            "matchChiralTag": False,
            "matchValences": False,
        },
        {
            "name": "scaffold_fallback",
            "atomCompare": compare_elements,
            "bondCompare": compare_any_bond,
            "ringMatchesRingOnly": True,
            "completeRingsOnly": False,
            "matchChiralTag": False,
            "matchValences": False,
        },
        {
            "name": "heavy_atom_last_resort",
            "atomCompare": compare_any_atom,
            "bondCompare": compare_any_bond,
            "ringMatchesRingOnly": False,
            "completeRingsOnly": False,
            "matchChiralTag": False,
            "matchValences": False,
        },
    ]


def _find_mcs(template: Chem.Mol, ligand: Chem.Mol, level: dict, timeout: int):
    kwargs = dict(
        timeout=int(timeout),
        atomCompare=level["atomCompare"],
        bondCompare=level["bondCompare"],
        ringMatchesRingOnly=bool(level["ringMatchesRingOnly"]),
        completeRingsOnly=bool(level["completeRingsOnly"]),
        matchChiralTag=bool(level["matchChiralTag"]),
        matchValences=bool(level["matchValences"]),
    )
    try:
        return rdFMCS.FindMCS([template, ligand], **kwargs)
    except TypeError:  # pragma: no cover - older RDKit signatures
        kwargs.pop("matchValences", None)
        return rdFMCS.FindMCS([template, ligand], **kwargs)


def _has_conformer(mol: Chem.Mol) -> bool:
    return mol is not None and mol.GetNumConformers() > 0


def _spread_bonus(template: Chem.Mol, match: Iterable[int]) -> float:
    if not _has_conformer(template):
        return 0.0
    idxs = list(match)
    if len(idxs) < 2:
        return 0.0
    conf = template.GetConformer()
    coords = np.asarray([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in idxs], dtype=float)
    centroid = coords.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1))))
    # A spatially spread MCS anchors the binding mode better than a tiny side phenyl.
    return min(rg / 6.0, 1.0)


def _ring_penalty(template: Chem.Mol, ligand: Chem.Mol, mapping: list[tuple[int, int]]) -> float:
    penalty = 0.0
    for tmpl_idx, lig_idx in mapping:
        if template.GetAtomWithIdx(tmpl_idx).IsInRing() != ligand.GetAtomWithIdx(lig_idx).IsInRing():
            penalty += 0.25
    return penalty


def _chirality_warnings(template: Chem.Mol, ligand: Chem.Mol, mapping: list[tuple[int, int]]) -> list[str]:
    warnings: list[str] = []
    for tmpl_idx, lig_idx in mapping:
        tmpl_tag = template.GetAtomWithIdx(tmpl_idx).GetChiralTag()
        lig_tag = ligand.GetAtomWithIdx(lig_idx).GetChiralTag()
        if (
            tmpl_tag != Chem.ChiralType.CHI_UNSPECIFIED
            and lig_tag != Chem.ChiralType.CHI_UNSPECIFIED
            and tmpl_tag != lig_tag
        ):
            warnings.append(f"mapped_chiral_tag_diff:{tmpl_idx}->{lig_idx}")
    return warnings


def _best_mapping_for_smarts(
    template: Chem.Mol,
    ligand: Chem.Mol,
    smarts: str,
    *,
    max_matches: int = 256,
) -> list[tuple[int, int]] | None:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return None
    tmpl_matches = list(template.GetSubstructMatches(patt, uniquify=False))[:max_matches]
    lig_matches = list(ligand.GetSubstructMatches(patt, uniquify=False))[:max_matches]
    if not tmpl_matches or not lig_matches:
        return None

    best: tuple[float, list[tuple[int, int]]] | None = None
    for tmpl_match in tmpl_matches:
        spread = _spread_bonus(template, tmpl_match)
        for lig_match in lig_matches:
            mapping = list(zip(tmpl_match, lig_match))
            ring_pen = _ring_penalty(template, ligand, mapping)
            score = spread - ring_pen
            if best is None or score > best[0]:
                best = (score, mapping)
    return best[1] if best else None


def _score_mapping(
    template: Chem.Mol,
    ligand: Chem.Mol,
    mapping: list[tuple[int, int]],
    num_bonds: int,
    warnings: list[str],
) -> float:
    ht = max(heavy_atom_count(template), 1)
    hl = max(heavy_atom_count(ligand), 1)
    frac_template = len(mapping) / ht
    frac_ligand = len(mapping) / hl
    mapped_fraction = 0.5 * (frac_template + frac_ligand)
    spread = _spread_bonus(template, [t for t, _ in mapping])
    ring_penalty = _ring_penalty(template, ligand, mapping)
    stereo_penalty = 0.05 * sum(1 for w in warnings if w.startswith("mapped_chiral_tag_diff"))
    bond_bonus = min(float(num_bonds) / max(float(len(mapping)), 1.0), 1.0) * 0.1
    return float(mapped_fraction + 0.25 * spread + bond_bonus - ring_penalty - stereo_penalty)


def find_robust_mcs(
    template: Chem.Mol,
    ligand: Chem.Mol,
    *,
    min_atoms: int = 8,
    min_fraction: float = 0.35,
    timeout: int = 10,
) -> MCSMapping:
    """Find a chemically useful MCS using strict-to-relaxed fallback levels.

    RDKit's MCS can fail on stereochemistry/kekulization details or return a
    mathematically large but chemically poor core.  This routine tries strict
    settings first, relaxes only when needed, and attaches quality metadata used
    later by FEP-readiness gates.
    """

    template_heavy = max(heavy_atom_count(template), 1)
    ligand_heavy = max(heavy_atom_count(ligand), 1)
    attempted: list[str] = []

    for level in _iter_mcs_levels():
        attempted.append(level["name"])
        try:
            result = _find_mcs(template, ligand, level, timeout)
        except Exception as exc:  # pragma: no cover - RDKit-version dependent
            continue
        if result is None or result.canceled or result.numAtoms <= 0 or not result.smartsString:
            continue
        mapping = _best_mapping_for_smarts(template, ligand, result.smartsString)
        if not mapping:
            continue

        warnings = _chirality_warnings(template, ligand, mapping)
        fraction_template = len(mapping) / template_heavy
        fraction_ligand = len(mapping) / ligand_heavy
        quality_score = _score_mapping(template, ligand, mapping, result.numBonds, warnings)
        status = "ok"
        if len(mapping) < int(min_atoms):
            status = "rejected"
            warnings.append(f"mcs_too_small:{len(mapping)}<{int(min_atoms)}")
        if fraction_ligand < float(min_fraction) or fraction_template < float(min_fraction) * 0.5:
            status = "rejected"
            warnings.append(
                "mcs_low_coverage:"
                f"template={fraction_template:.3f},ligand={fraction_ligand:.3f},min={float(min_fraction):.3f}"
            )
        if level["name"] == "heavy_atom_last_resort":
            warnings.append("last_resort_atom_mapping")

        out = MCSMapping(
            template_to_ligand=[(int(t), int(l)) for t, l in mapping],
            smarts=result.smartsString,
            level=level["name"],
            num_atoms=int(len(mapping)),
            num_bonds=int(result.numBonds),
            fraction_template=float(fraction_template),
            fraction_ligand=float(fraction_ligand),
            quality_score=float(quality_score),
            warnings=warnings,
            status=status,
        )
        if out.ok:
            return out

    return MCSMapping(
        template_to_ligand=[],
        warnings=["no_acceptable_mcs", f"attempted:{','.join(attempted)}"],
        status="failed",
    )


def pairwise_mapping_score(mapping: MCSMapping) -> float:
    """Small helper for ligand-network edge ranking."""
    if not mapping.ok:
        return -math.inf
    return float(mapping.quality_score + 0.25 * min(mapping.fraction_ligand, mapping.fraction_template))
