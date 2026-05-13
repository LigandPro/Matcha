from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from rdkit import Chem

from .constrained_embed import generate_constrained_conformers, mol_positions
from .fep_export import LigandAnalogueExport, write_fep_bundle
from .mcs import MCSMapping, find_robust_mcs
from .ranking import evaluate_pose, load_receptor_heavy_atom_positions, rank_poses
from .standardize import standardize_mol
from .torsion_mc import torsional_mc_refine


@dataclass
class AnalogueWorkflowConfig:
    n_seed_poses: int = 64
    n_final_poses: int = 8
    min_mcs_atoms: int = 8
    min_mcs_fraction: float = 0.35
    mcs_timeout: int = 10
    core_rmsd_cutoff: float = 1.0
    torsion_mc_steps: int = 0
    torsion_mc_keep: int = 64
    random_seed: int = 777
    export_fep_bundle: bool = True
    receptor_aware_ranking: bool = True
    gnina_score_poses: bool = False
    gnina_scorer_path: str | None = None
    gnina_minimize: bool = True
    gnina_score_type: str = "Affinity"
    gnina_cnn_scoring: str = "none"
    gnina_device: int = 0


@dataclass
class AnalogueWorkflowResult:
    output_dir: Path
    seed_transforms_path: Path
    fep_bundle_dir: Path
    selected_molecules: dict[str, Chem.Mol]
    mappings: dict[str, MCSMapping]
    summary: dict
    failures: list[dict] = field(default_factory=list)


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _clone_with_name(mol: Chem.Mol, name: str) -> Chem.Mol:
    out = copy.deepcopy(mol)
    out.SetProp("_Name", name)
    return out


def _seed_transform_for_pose(mol: Chem.Mol, *, full_protein_center: np.ndarray | None = None) -> dict:
    if full_protein_center is None:
        full_protein_center = np.zeros(3, dtype=np.float32)
    coords = mol_positions(mol).astype(np.float32)
    return {
        "tr_pred_init": coords.mean(axis=0).astype(np.float32),
        "full_protein_center": np.asarray(full_protein_center, dtype=np.float32),
        "transformed_orig": coords,
        "seed_source": "analogue_mcs_constrained_embed",
    }


def _build_seed_transforms(
    ranked_by_ligand: Mapping[str, list[tuple[Chem.Mol, object]]],
    *,
    n_seed_poses: int,
) -> dict:
    transforms: dict[str, list[dict]] = {}
    for ligand_id, ranked in ranked_by_ligand.items():
        selected = ranked[: max(1, int(n_seed_poses))]
        for conf_idx, (mol, qc) in enumerate(selected):
            key = f"{ligand_id}_mol0_conf{conf_idx}"
            transform = _seed_transform_for_pose(mol)
            if hasattr(qc, "to_dict"):
                transform.update({
                    "analogue_core_rmsd": float(qc.core_rmsd),
                    "analogue_rank_score": float(qc.rank_score),
                    "analogue_status": qc.status,
                })
            transforms[key] = [transform]
    return transforms


def _extract_score(mol: Chem.Mol, score_type: str, minimized: bool) -> float | None:
    names = [score_type]
    if minimized:
        names.insert(0, f"minimized{score_type}" if score_type != "Affinity" else "minimizedAffinity")
    for name in names:
        if mol.HasProp(name):
            try:
                return float(mol.GetProp(name))
            except (TypeError, ValueError):
                pass
    for name, value in mol.GetPropsAsDict(includePrivate=True, includeComputed=True).items():
        if score_type.lower() in str(name).lower():
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _gnina_rerank_poses(
    *,
    ligand_id: str,
    ranked: list[tuple[Chem.Mol, object]],
    template: Chem.Mol,
    mapping: MCSMapping,
    receptor_path: Path,
    receptor_positions: np.ndarray | None,
    output_dir: Path,
    cfg: AnalogueWorkflowConfig,
) -> list[tuple[Chem.Mol, object]]:
    if not ranked:
        return ranked

    from matcha.scoring.gnina_scorer import GninaScorer

    input_dir = output_dir / "gnina_input"
    output_scored_dir = output_dir / "gnina_scored"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_scored_dir.mkdir(parents=True, exist_ok=True)
    input_sdf = input_dir / f"{ligand_id}.sdf"

    writer = Chem.SDWriter(str(input_sdf))
    writer.SetKekulize(False)
    for pose_idx, (mol, _qc) in enumerate(ranked):
        mol.SetProp("_Name", f"{ligand_id}::{pose_idx}")
        writer.write(mol)
    writer.close()

    scorer = GninaScorer(
        gnina_path=cfg.gnina_scorer_path,
        minimize=cfg.gnina_minimize,
        score_type=cfg.gnina_score_type,
        cnn_scoring=cfg.gnina_cnn_scoring,
    )
    scorer.score_poses(str(receptor_path), str(input_dir), str(output_scored_dir), device=cfg.gnina_device)

    scored_sdf = output_scored_dir / f"{ligand_id}.sdf"
    if not scored_sdf.exists():
        return ranked

    scored: list[tuple[Chem.Mol, object]] = []
    for scored_idx, mol in enumerate(Chem.SDMolSupplier(str(scored_sdf), removeHs=False, sanitize=False)):
        if mol is None:
            continue
        score = _extract_score(mol, cfg.gnina_score_type, cfg.gnina_minimize)
        qc = evaluate_pose(
            ligand_id,
            scored_idx,
            mol,
            template,
            mapping,
            core_rmsd_cutoff=cfg.core_rmsd_cutoff,
            min_mcs_fraction=cfg.min_mcs_fraction,
            receptor_positions=receptor_positions,
        )
        if score is not None:
            mol.SetProp("analogue_gnina_score", f"{score:.6f}")
            qc.warnings.append(f"gnina_{cfg.gnina_score_type}:{score:.6f}")
            qc.rank_score = float(score)
        scored.append((mol, qc))

    if not scored:
        return ranked
    reverse = cfg.gnina_score_type not in {"Affinity", "minimizedAffinity"}
    scored.sort(key=lambda item: item[1].rank_score, reverse=reverse)
    return scored


def run_analogue_workflow(
    *,
    template_mol: Chem.Mol,
    ligands: Iterable[tuple[str, Chem.Mol]],
    output_dir: Path,
    receptor_path: Path | None = None,
    config: AnalogueWorkflowConfig | None = None,
) -> AnalogueWorkflowResult:
    """Generate template-aligned analogue poses and FEP export artifacts.

    This is the main clean entry-point used by the CLI.  It does not require a
    trained Matcha checkpoint: it produces a deterministic seed/FEP bundle on its
    own, and optionally the caller can feed ``seed_transforms_path`` into Matcha
    stage-3 refinement.
    """

    cfg = config or AnalogueWorkflowConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fep_bundle_dir = output_dir / "fep_bundle_seed"

    template_std_res = standardize_mol(template_mol, remove_hs=True, sanitize=True)
    if template_std_res.mol is None or template_std_res.mol.GetNumConformers() == 0:
        raise ValueError("Analogue template must be readable and contain 3D coordinates.")
    template_std = template_std_res.mol
    template_std.SetProp("_Name", template_mol.GetProp("_Name") if template_mol.HasProp("_Name") else "template")
    receptor_positions = (
        load_receptor_heavy_atom_positions(receptor_path)
        if cfg.receptor_aware_ranking and receptor_path is not None
        else None
    )

    exports: dict[str, LigandAnalogueExport] = {}
    selected_molecules: dict[str, Chem.Mol] = {}
    mappings: dict[str, MCSMapping] = {}
    failures: list[dict] = []
    ranked_by_ligand: dict[str, list[tuple[Chem.Mol, object]]] = {}

    for ligand_id, ligand_mol in ligands:
        ligand_std_res = standardize_mol(ligand_mol, remove_hs=True, sanitize=True)
        ligand_warnings = list(ligand_std_res.warnings)
        if ligand_std_res.mol is None:
            mapping = MCSMapping([], status="failed", warnings=["standardization_failed"] + ligand_warnings)
            exports[ligand_id] = LigandAnalogueExport(ligand_id, mapping, [], ligand_warnings, failed=True)
            failures.append({"ligand_id": ligand_id, "reason": "standardization_failed", "warnings": ligand_warnings})
            continue
        ligand_std = ligand_std_res.mol
        ligand_std.SetProp("_Name", ligand_id)

        mapping = find_robust_mcs(
            template_std,
            ligand_std,
            min_atoms=cfg.min_mcs_atoms,
            min_fraction=cfg.min_mcs_fraction,
            timeout=cfg.mcs_timeout,
        )
        mappings[ligand_id] = mapping
        if not mapping.ok:
            warnings = ligand_warnings + list(mapping.warnings)
            exports[ligand_id] = LigandAnalogueExport(ligand_id, mapping, [], warnings, failed=True)
            failures.append({"ligand_id": ligand_id, "reason": "failed_mcs", "warnings": warnings})
            continue

        conformer_result = generate_constrained_conformers(
            template_std,
            ligand_std,
            mapping,
            n_conformers=cfg.n_seed_poses,
            random_seed=cfg.random_seed,
            # Keep the seed generator deterministic and fast.  Strain is still
            # estimated during ranking; expensive QM/MM relaxation can be added
            # downstream for top-N poses.
            optimize=False,
        )
        poses = conformer_result.conformers
        warnings = ligand_warnings + list(mapping.warnings) + conformer_result.warnings
        if cfg.torsion_mc_steps > 0 and poses:
            mc_result = torsional_mc_refine(
                template_std,
                poses,
                mapping,
                n_steps=cfg.torsion_mc_steps,
                keep=max(cfg.torsion_mc_keep, cfg.n_seed_poses),
                random_seed=cfg.random_seed,
            )
            warnings.extend(mc_result.warnings)
            if mc_result.poses:
                poses = mc_result.poses

        if not poses:
            exports[ligand_id] = LigandAnalogueExport(ligand_id, mapping, [], warnings + ["no_seed_poses"], failed=True)
            failures.append({"ligand_id": ligand_id, "reason": "no_seed_poses", "warnings": warnings})
            continue

        ranked = rank_poses(
            ligand_id,
            poses,
            template_std,
            mapping,
            core_rmsd_cutoff=cfg.core_rmsd_cutoff,
            min_mcs_fraction=cfg.min_mcs_fraction,
            receptor_positions=receptor_positions,
        )[: max(1, int(cfg.n_seed_poses))]
        if cfg.gnina_score_poses and receptor_path is not None:
            ranked = _gnina_rerank_poses(
                ligand_id=ligand_id,
                ranked=ranked,
                template=template_std,
                mapping=mapping,
                receptor_path=Path(receptor_path),
                receptor_positions=receptor_positions,
                output_dir=output_dir / "gnina_ranking" / ligand_id,
                cfg=cfg,
            )[: max(1, int(cfg.n_seed_poses))]
        ranked_by_ligand[ligand_id] = ranked
        final_ranked = ranked[: max(1, int(cfg.n_final_poses))]
        exports[ligand_id] = LigandAnalogueExport(ligand_id, mapping, final_ranked, warnings, failed=False)
        if final_ranked:
            selected_molecules[ligand_id] = _clone_with_name(final_ranked[0][0], ligand_id)

    seed_transforms = _build_seed_transforms(ranked_by_ligand, n_seed_poses=cfg.n_seed_poses)
    seed_transforms_path = output_dir / "analogue_seed_transforms.npy"
    np.save(seed_transforms_path, [seed_transforms])

    _write_json(output_dir / "analogue_mappings.json", {k: v.to_dict() for k, v in mappings.items()})
    _write_json(output_dir / "analogue_failures.json", failures)

    if cfg.export_fep_bundle:
        summary = write_fep_bundle(
            output_dir=fep_bundle_dir,
            template_mol=template_std,
            exports=exports,
            receptor_path=receptor_path,
        )
    else:
        summary = {}

    ready = 0
    needs_review = 0
    for export in exports.values():
        if export.best is None:
            continue
        _, qc = export.best
        if qc.status == "FEP_READY":
            ready += 1
        elif qc.status == "NEEDS_REVIEW":
            needs_review += 1
    workflow_summary = {
        "ligands_total": len(exports),
        "ligands_with_seed_poses": len(selected_molecules),
        "fep_ready": ready,
        "needs_review": needs_review,
        "failed": len(failures),
        "seed_transforms_path": str(seed_transforms_path.resolve()),
        "fep_bundle_dir": str(fep_bundle_dir.resolve()),
        **summary,
    }
    _write_json(output_dir / "analogue_summary.json", workflow_summary)

    return AnalogueWorkflowResult(
        output_dir=output_dir,
        seed_transforms_path=seed_transforms_path,
        fep_bundle_dir=fep_bundle_dir,
        selected_molecules=selected_molecules,
        mappings=mappings,
        summary=workflow_summary,
        failures=failures,
    )
