from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

from rdkit import Chem


@dataclass
class StandardizationResult:
    """RDKit molecule plus recoverable chemistry warnings."""

    mol: Chem.Mol | None
    warnings: list[str] = field(default_factory=list)
    failed: bool = False


def _try_sanitize(mol: Chem.Mol, warnings: list[str]) -> None:
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # pragma: no cover - exact RDKit errors vary by version
        warnings.append(f"sanitize_failed:{type(exc).__name__}")


def standardize_mol(
    mol: Chem.Mol | None,
    *,
    remove_hs: bool = True,
    sanitize: bool = True,
    keep_props: Iterable[str] | None = None,
) -> StandardizationResult:
    """Return a defensive copy of ``mol`` normalized for analogue-MCS work.

    The function is intentionally conservative: it keeps atom order stable, does
    not enumerate tautomers/protomers, and records warnings instead of silently
    changing chemistry.  FEP preparation is very sensitive to hidden chemistry
    edits, so this routine only performs operations needed for stable matching.
    """

    warnings: list[str] = []
    if mol is None:
        return StandardizationResult(None, ["mol_is_none"], failed=True)

    out = copy.deepcopy(mol)
    props: dict[str, str] = {}
    if keep_props:
        for prop in keep_props:
            if out.HasProp(prop):
                props[prop] = out.GetProp(prop)

    if sanitize:
        _try_sanitize(out, warnings)

    if remove_hs:
        try:
            out = Chem.RemoveHs(out, sanitize=False)
        except Exception as exc:  # pragma: no cover - RDKit-version dependent
            warnings.append(f"remove_hs_failed:{type(exc).__name__}")

    # RemoveHs(sanitize=False) can leave ring information uninitialized in some
    # RDKit builds.  Initialize the property cache/rings for MCS substructure
    # queries without performing chemistry-changing normalization.
    try:
        out.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(out)
    except Exception as exc:  # pragma: no cover - RDKit-version dependent
        warnings.append(f"ringinfo_init_failed:{type(exc).__name__}")

    for prop, value in props.items():
        out.SetProp(prop, value)

    # Keep user-facing IDs stable when possible.
    if mol.HasProp("_Name") and not out.HasProp("_Name"):
        out.SetProp("_Name", mol.GetProp("_Name"))

    return StandardizationResult(out, warnings, failed=False)


def heavy_atom_count(mol: Chem.Mol | None) -> int:
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def formal_charge(mol: Chem.Mol | None) -> int:
    if mol is None:
        return 0
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
