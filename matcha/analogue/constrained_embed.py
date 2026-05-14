from __future__ import annotations

import copy
import multiprocessing as mp
import queue
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
    requested_conformers: int = 0
    raw_conformers: int = 0
    seed_batches: int = 1


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


def _embed_multiple_confs_once(mol: Chem.Mol, num_conformers: int, params) -> list[int]:
    return list(AllChem.EmbedMultipleConfs(mol, numConfs=int(num_conformers), params=params))


def _embed_multiple_confs_worker(mol: Chem.Mol, num_conformers: int, params, out_queue) -> None:
    try:
        conf_ids = _embed_multiple_confs_once(mol, num_conformers, params)
        out_queue.put(("ok", mol, conf_ids))
    except BaseException as exc:
        out_queue.put(("error", type(exc).__name__, str(exc)))


def _copy_conformers(target: Chem.Mol, source: Chem.Mol) -> None:
    target.RemoveAllConformers()
    for conf in source.GetConformers():
        target.AddConformer(Chem.Conformer(conf), assignId=False)


def _embed_multiple_confs(
    mol: Chem.Mol,
    num_conformers: int,
    params,
    *,
    timeout_seconds: int | float | None,
) -> list[int]:
    if timeout_seconds is None or float(timeout_seconds) <= 0:
        return _embed_multiple_confs_once(mol, num_conformers, params)

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    out_queue = ctx.Queue()
    proc = ctx.Process(
        target=_embed_multiple_confs_worker,
        args=(copy.deepcopy(mol), int(num_conformers), params, out_queue),
    )
    proc.start()
    proc.join(float(timeout_seconds))
    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        if proc.is_alive():  # pragma: no cover - terminate should be enough on supported hosts
            proc.kill()
            proc.join()
        raise TimeoutError(f"rdkit_embed_timeout:{float(timeout_seconds):.1f}s")

    try:
        status, payload, extra = out_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError(f"rdkit_embed_worker_failed:exitcode={proc.exitcode}") from exc
    finally:
        out_queue.close()

    if status == "error":
        raise RuntimeError(f"{payload}:{extra}")
    _copy_conformers(mol, payload)
    return list(extra)


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


def _append_aligned_conformers(
    out: list[Chem.Mol],
    source: Chem.Mol,
    conf_ids: list[int],
    template: Chem.Mol,
    mapping: MCSMapping,
    *,
    optimize: bool,
) -> None:
    for conf_id in conf_ids:
        if int(conf_id) < 0:
            continue
        if optimize:
            _mmff_or_uff_optimize(source, int(conf_id))
        mol_conf = _single_conformer_copy(source, int(conf_id))
        try:
            mol_conf = Chem.RemoveHs(mol_conf, sanitize=False)
        except Exception:
            pass
        mol_conf = align_mol_to_template_core(mol_conf, template, mapping)
        mol_conf.SetProp("analogue_core_rmsd", f"{core_rmsd(mol_conf, template, mapping):.6f}")
        out.append(mol_conf)


def _embed_seed_batches(
    work: Chem.Mol,
    *,
    n_conformers: int,
    seed_batches: int,
    random_seed: int,
    use_random_coords: bool,
    embed_timeout_seconds: int | None,
    coord_map: dict[int, Point3D] | None,
    warnings: list[str],
) -> list[Chem.Mol]:
    embedded: list[Chem.Mol] = []
    batches = max(1, int(seed_batches))
    batch_size = max(1, int(np.ceil(max(1, int(n_conformers)) / batches)))
    for batch_idx in range(batches):
        batch = copy.deepcopy(work)
        batch.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = int(random_seed) + batch_idx * 9973
        params.useRandomCoords = bool(use_random_coords)
        params.clearConfs = True
        _set_embed_timeout(params, embed_timeout_seconds)
        if coord_map is not None:
            try:
                _set_coord_map(params, coord_map)
            except Exception as exc:  # pragma: no cover - RDKit-version dependent
                warnings.append(f"coord_map_failed:{type(exc).__name__}")
                coord_map = None
        try:
            conf_ids = _embed_multiple_confs(
                batch,
                batch_size,
                params,
                timeout_seconds=embed_timeout_seconds,
            )
        except Exception as exc:
            warnings.append(f"embed_batch_{batch_idx}_failed:{type(exc).__name__}")
            continue
        if conf_ids:
            embedded.append(batch)
    return embedded


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
    include_unconstrained_supplement: bool = True,
    seed_batches: int = 4,
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
    coord_map = None
    try:
        coord_map = _make_coord_map(template, mapping)
    except Exception as exc:  # pragma: no cover - RDKit-version dependent
        warnings.append(f"coord_map_failed:{type(exc).__name__}")

    embedded_batches = _embed_seed_batches(
        work,
        n_conformers=int(n_conformers),
        seed_batches=seed_batches,
        random_seed=random_seed,
        use_random_coords=use_random_coords,
        embed_timeout_seconds=embed_timeout_seconds,
        coord_map=coord_map,
        warnings=warnings,
    )

    used_unconstrained_fallback = False
    if not embedded_batches:
        # Fallback: unconstrained ETKDG followed by explicit MCS alignment.  This
        # often rescues cases where coordMap overconstrains macrocycles/linkers.
        embedded_batches = _embed_seed_batches(
            work,
            n_conformers=int(n_conformers),
            seed_batches=seed_batches,
            random_seed=random_seed,
            use_random_coords=True,
            embed_timeout_seconds=embed_timeout_seconds,
            coord_map=None,
            warnings=warnings,
        )
        if not embedded_batches:
            return ConformerGenerationResult(
                [],
                warnings + ["embed_fallback_failed:no_conformers"],
                failures=1,
                requested_conformers=int(n_conformers),
                raw_conformers=0,
                seed_batches=max(1, int(seed_batches)),
            )
        warnings.append("used_unconstrained_embed_fallback")
        used_unconstrained_fallback = True

    out: list[Chem.Mol] = []
    for batch in embedded_batches:
        _append_aligned_conformers(
            out,
            batch,
            [conf.GetId() for conf in batch.GetConformers()],
            template,
            mapping,
            optimize=optimize,
        )

    if include_unconstrained_supplement and not used_unconstrained_fallback:
        supplement_batches = _embed_seed_batches(
            work,
            n_conformers=int(n_conformers),
            seed_batches=seed_batches,
            random_seed=int(random_seed) + 104729,
            use_random_coords=True,
            embed_timeout_seconds=embed_timeout_seconds,
            coord_map=None,
            warnings=warnings,
        )
        if supplement_batches:
            for supplement in supplement_batches:
                _append_aligned_conformers(
                    out,
                    supplement,
                    [conf.GetId() for conf in supplement.GetConformers()],
                    template,
                    mapping,
                    optimize=optimize,
                )
            warnings.append("used_unconstrained_embed_supplement")
        else:
            warnings.append("unconstrained_embed_supplement_failed:no_conformers")

    raw_conformers = len(out)
    if deduplicate:
        out = _deduplicate_by_core_and_whole_rmsd(
            out,
            template,
            mapping,
            max_conformers=max(1, int(n_conformers)),
        )

    return ConformerGenerationResult(
        out,
        warnings,
        failures=max(0, int(n_conformers) - len(out)),
        requested_conformers=int(n_conformers),
        raw_conformers=raw_conformers,
        seed_batches=max(1, int(seed_batches)),
    )
