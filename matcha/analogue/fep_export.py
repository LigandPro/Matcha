from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from rdkit import Chem

from .mcs import MCSMapping, find_robust_mcs, pairwise_mapping_score
from .ranking import PoseQC, rank_poses


@dataclass
class LigandAnalogueExport:
    ligand_id: str
    mapping: MCSMapping
    ranked_poses: list[tuple[Chem.Mol, PoseQC]]
    warnings: list[str] = field(default_factory=list)
    failed: bool = False

    @property
    def best(self) -> tuple[Chem.Mol, PoseQC] | None:
        return self.ranked_poses[0] if self.ranked_poses else None


def _write_mol_list(path: Path, mols: list[Chem.Mol]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.SetKekulize(False)
    for mol in mols:
        writer.write(mol)
    writer.close()


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_fep_bundle(
    *,
    output_dir: Path,
    template_mol: Chem.Mol,
    exports: Mapping[str, LigandAnalogueExport],
    receptor_path: Path | None = None,
    template_id: str = "template",
    include_pairwise_edges: bool = True,
) -> dict:
    """Write a generic FEP/RBFE-ready bundle from ranked analogue poses."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complexes_dir = output_dir / "complexes"
    complexes_dir.mkdir(parents=True, exist_ok=True)

    template_out = output_dir / "template.sdf"
    _write_mol_list(template_out, [template_mol])

    aligned_best: list[Chem.Mol] = []
    quality_rows: list[dict] = []
    mappings: dict[str, dict] = {}
    failures: list[dict] = []
    manifest_nodes: list[dict] = []

    if receptor_path is not None:
        receptor_path = Path(receptor_path)
        if receptor_path.exists():
            shutil.copyfile(receptor_path, output_dir / receptor_path.name)

    manifest_nodes.append({"id": template_id, "role": "template", "sdf": str(template_out.relative_to(output_dir))})

    for ligand_id, record in exports.items():
        ligand_dir = complexes_dir / ligand_id
        ligand_dir.mkdir(parents=True, exist_ok=True)
        mappings[ligand_id] = record.mapping.to_dict()

        if record.failed or record.best is None:
            failures.append({
                "ligand_id": ligand_id,
                "status": "failed",
                "warnings": list(record.warnings) + list(record.mapping.warnings),
                "mapping": record.mapping.to_dict(),
            })
            continue

        best_mol, best_qc = record.best
        best_mol.SetProp("_Name", ligand_id)
        best_mol.SetProp("analogue_selected", "true")
        best_mol.SetProp("analogue_fep_ready", str(bool(best_qc.fep_ready)).lower())
        best_mol.SetProp("analogue_core_rmsd", f"{best_qc.core_rmsd:.6f}")
        best_mol.SetProp("analogue_rank_score", f"{best_qc.rank_score:.6f}")
        best_mol.SetProp("analogue_status", best_qc.status)
        aligned_best.append(best_mol)

        all_pose_mols = [mol for mol, _ in record.ranked_poses]
        _write_mol_list(ligand_dir / "poses.sdf", all_pose_mols)
        _write_mol_list(ligand_dir / "best_pose.sdf", [best_mol])
        _write_json(ligand_dir / "quality.json", best_qc.to_dict())
        _write_json(ligand_dir / "atom_mapping.json", record.mapping.to_dict())
        if receptor_path is not None and receptor_path.exists():
            shutil.copyfile(receptor_path, ligand_dir / receptor_path.name)

        row = best_qc.to_dict()
        row["warnings"] = ";".join(best_qc.warnings + record.warnings)
        quality_rows.append(row)
        manifest_nodes.append({
            "id": ligand_id,
            "role": "analogue",
            "sdf": str((ligand_dir / "best_pose.sdf").relative_to(output_dir)),
            "status": best_qc.status,
            "fep_ready": bool(best_qc.fep_ready),
            "core_rmsd": float(best_qc.core_rmsd),
            "mcs_atoms": int(record.mapping.num_atoms),
        })

    _write_mol_list(output_dir / "aligned_series.sdf", aligned_best)
    _write_json(output_dir / "mcs_mappings.json", mappings)
    _write_json(output_dir / "failures.json", failures)

    csv_path = output_dir / "quality_report.csv"
    fieldnames = [
        "ligand_id",
        "pose_index",
        "core_rmsd",
        "whole_centroid_distance",
        "internal_clash_count",
        "receptor_clash_count",
        "receptor_contact_count",
        "strain_energy",
        "mcs_atoms",
        "mcs_fraction_ligand",
        "mcs_fraction_template",
        "rank_score",
        "status",
        "fep_ready",
        "warnings",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in quality_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    rbfe_graph = _build_rbfe_graph(template_id, template_mol, exports, include_pairwise_edges=include_pairwise_edges)
    manifest = {
        "format": "matcha_fep_manifest_v1",
        "template_id": template_id,
        "nodes": manifest_nodes,
        "edges": rbfe_graph["edges"],
        "recommended_engine_inputs": {
            "openfe": "Use aligned_series.sdf plus rbfe_graph.json atom mappings as a starting network.",
            "generic": "Use complexes/*/best_pose.sdf after force-field parameterization and minimization checks.",
        },
    }
    _write_json(output_dir / "fep_manifest.json", manifest)
    _write_json(output_dir / "rbfe_graph.json", rbfe_graph)

    summary = {
        "ligands_total": len(exports),
        "fep_ready": sum(1 for row in quality_rows if row.get("fep_ready")),
        "needs_review": sum(1 for row in quality_rows if row.get("status") == "NEEDS_REVIEW"),
        "failed": len(failures),
        "aligned_series_sdf": str((output_dir / "aligned_series.sdf").resolve()),
        "quality_report_csv": str(csv_path.resolve()),
        "fep_manifest_json": str((output_dir / "fep_manifest.json").resolve()),
        "rbfe_pairwise_edges": bool(include_pairwise_edges),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _build_rbfe_graph(
    template_id: str,
    template_mol: Chem.Mol,
    exports: Mapping[str, LigandAnalogueExport],
    *,
    include_pairwise_edges: bool = True,
) -> dict:
    nodes = [{"id": template_id, "role": "template"}]
    edges: list[dict] = []
    accepted: list[tuple[str, Chem.Mol, MCSMapping]] = []

    for ligand_id, record in exports.items():
        if record.best is None:
            continue
        mol, qc = record.best
        nodes.append({"id": ligand_id, "role": "analogue", "status": qc.status})
        accepted.append((ligand_id, mol, record.mapping))
        edges.append({
            "ligand_a": template_id,
            "ligand_b": ligand_id,
            "kind": "template_star",
            "mapping_score": float(record.mapping.quality_score),
            "mcs_atoms": int(record.mapping.num_atoms),
            "mcs_fraction_ligand": float(record.mapping.fraction_ligand),
            "recommended": bool(record.mapping.ok and qc.status in {"FEP_READY", "NEEDS_REVIEW"}),
            "atom_mapping_template_to_ligand": [[int(a), int(b)] for a, b in record.mapping.template_to_ligand],
        })

    if not include_pairwise_edges:
        return {"format": "matcha_rbfe_graph_v1", "nodes": nodes, "edges": edges}

    # Add a sparse pairwise analogue network for RBFE planning.
    for i, (id_a, mol_a, _) in enumerate(accepted):
        for id_b, mol_b, _ in accepted[i + 1 :]:
            mapping = find_robust_mcs(mol_a, mol_b, min_atoms=6, min_fraction=0.35, timeout=5)
            if not mapping.ok:
                continue
            score = pairwise_mapping_score(mapping)
            edges.append({
                "ligand_a": id_a,
                "ligand_b": id_b,
                "kind": "analogue_pair",
                "mapping_score": float(score),
                "mcs_atoms": int(mapping.num_atoms),
                "mcs_fraction_a": float(mapping.fraction_template),
                "mcs_fraction_b": float(mapping.fraction_ligand),
                "recommended": bool(mapping.fraction_ligand >= 0.5 and mapping.fraction_template >= 0.5),
                "atom_mapping_a_to_b": [[int(a), int(b)] for a, b in mapping.template_to_ligand],
            })

    return {"format": "matcha_rbfe_graph_v1", "nodes": nodes, "edges": edges}


def export_pose_files_as_fep_bundle(
    *,
    output_dir: Path,
    pose_files: Mapping[str, Path],
    template_mol: Chem.Mol,
    mappings: Mapping[str, MCSMapping],
    receptor_path: Path | None = None,
    core_rmsd_cutoff: float = 1.0,
    min_mcs_fraction: float = 0.35,
) -> dict:
    """Build a FEP bundle from already-written Matcha best-pose SDF files."""

    exports: dict[str, LigandAnalogueExport] = {}
    for ligand_id, path in pose_files.items():
        path = Path(path)
        mapping = mappings.get(ligand_id)
        if mapping is None:
            exports[ligand_id] = LigandAnalogueExport(
                ligand_id=ligand_id,
                mapping=MCSMapping([], status="failed", warnings=["mapping_missing_for_refined_pose"]),
                ranked_poses=[],
                failed=True,
            )
            continue
        suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        mols = [mol for mol in suppl if mol is not None]
        if not mols:
            exports[ligand_id] = LigandAnalogueExport(
                ligand_id=ligand_id,
                mapping=mapping,
                ranked_poses=[],
                warnings=[f"pose_file_empty:{path}"],
                failed=True,
            )
            continue
        ranked = rank_poses(
            ligand_id,
            mols,
            template_mol,
            mapping,
            core_rmsd_cutoff=core_rmsd_cutoff,
            min_mcs_fraction=min_mcs_fraction,
        )
        exports[ligand_id] = LigandAnalogueExport(ligand_id, mapping, ranked)
    return write_fep_bundle(
        output_dir=output_dir,
        template_mol=template_mol,
        exports=exports,
        receptor_path=receptor_path,
    )
