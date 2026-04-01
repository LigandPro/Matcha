from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem


PB_BOOL_KEYS = ("not_too_far_away", "no_internal_clash", "no_clashes", "no_volume_clash")


def _select_pb_filters_path(preds_root: Path, dataset_name: str) -> Path:
    minimized = preds_root / dataset_name / "filters_results_minimized.json"
    if minimized.exists():
        return minimized
    return preds_root / dataset_name / "filters_results.json"


def _load_pb_filters(preds_root: Path, dataset_name: str) -> dict[str, Any]:
    path = _select_pb_filters_path(preds_root, dataset_name)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _enrich_metrics_with_pb_filters(metrics: dict[str, Any], pb_filters: dict[str, Any]) -> None:
    for uid_key, uid_data in metrics.items():
        uid_real = uid_key.split("_mol")[0] if "_mol" in uid_key else uid_key
        if uid_real not in pb_filters:
            continue
        fdata = pb_filters[uid_real]
        for index, sample in enumerate(uid_data.get("sample_metrics", [])):
            counts = fdata.get("posebusters_filters_passed_count_fast", [])
            if index >= len(counts):
                continue
            sample["posebusters_filters_passed_count_fast"] = counts[index]
            bool_flags = [fdata.get(key, [])[index] if index < len(fdata.get(key, [])) else False for key in PB_BOOL_KEYS]
            buried = fdata.get("is_buried_fraction", [])
            buried_value = buried[index] if index < len(buried) else None
            sample["posebusters_filters_fast"] = [*bool_flags, buried_value]
            sample["buried_fraction"] = buried_value


def _enrich_metrics_with_gnina_scores(metrics: dict[str, Any], scored_dir: Path) -> None:
    if not scored_dir.exists():
        return
    for uid_key, uid_data in metrics.items():
        uid_real = uid_key.split("_mol")[0] if "_mol" in uid_key else uid_key
        scored_sdf = scored_dir / f"{uid_real}.sdf"
        if not scored_sdf.exists():
            continue
        suppl = Chem.SDMolSupplier(str(scored_sdf), removeHs=False, sanitize=False)
        for index, mol in enumerate(suppl):
            if mol is None or index >= len(uid_data.get("sample_metrics", [])):
                continue
            for prop in ("minimizedAffinity", "Affinity", "minimizedCNNscore", "CNNscore"):
                if not mol.HasProp(prop):
                    continue
                try:
                    uid_data["sample_metrics"][index]["gnina_score"] = float(mol.GetProp(prop))
                    break
                except (TypeError, ValueError):
                    continue


def _rank_samples(sample_metrics: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any]]]:
    has_gnina = any("gnina_score" in sample for sample in sample_metrics)
    ranked_indices = sorted(
        range(len(sample_metrics)),
        key=lambda index: (
            -int(sample_metrics[index].get("posebusters_filters_passed_count_fast", 0)),
            float(sample_metrics[index].get("gnina_score", float("inf"))) if has_gnina else 0.0,
        ),
    )
    return [
        (rank, sample_index, sample_metrics[sample_index])
        for rank, sample_index in enumerate(ranked_indices, start=1)
    ]


def _mol_with_positions(orig_mol, positions: np.ndarray, name: str):
    mol = copy.deepcopy(orig_mol)
    conf = Chem.Conformer(orig_mol.GetNumAtoms())
    for atom_index, (x, y, z) in enumerate(positions):
        conf.SetAtomPosition(atom_index, (float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    mol.SetProp("_Name", name)
    return mol


def _mol_to_sdf_record(mol) -> str:
    buffer = io.StringIO()
    writer = Chem.SDWriter(buffer)
    writer.SetKekulize(False)
    writer.write(mol)
    writer.flush()
    writer.close()
    text = buffer.getvalue().replace("\r\n", "\n")
    return text.removesuffix("$$$$\n").strip()


def _json_ready(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _build_trajectory_frames(orig_mol, sample: dict[str, Any], uid_label: str, pose_rank: int) -> list[dict[str, Any]]:
    frames = []
    for frame_index, frame in enumerate(sample.get("trajectory", [])):
        if isinstance(frame, dict):
            positions = np.asarray(frame.get("positions"))
            step_label = int(frame.get("step", frame_index))
        else:
            positions = np.asarray(frame)
            step_label = frame_index
            frame = {}
        mol = _mol_with_positions(orig_mol, positions, f"{uid_label}_rank{pose_rank}_step{step_label}")
        frames.append(
            {
                "id": frame_index,
                "label": f"Step {step_label}",
                "sdf": _mol_to_sdf_record(mol),
                "step": step_label,
                "time": float(frame["time"]) if frame.get("time") is not None else None,
                "deltaTranslation": _json_ready(frame.get("delta_translation")),
                "deltaRotation": _json_ready(frame.get("delta_rotation")),
                "deltaTorsion": _json_ready(frame.get("delta_torsion")),
                "deltaTranslationNorm": float(frame["delta_translation_norm"]) if frame.get("delta_translation_norm") is not None else None,
                "deltaRotationNorm": float(frame["delta_rotation_norm"]) if frame.get("delta_rotation_norm") is not None else None,
                "deltaTorsionNorm": float(frame["delta_torsion_norm"]) if frame.get("delta_torsion_norm") is not None else None,
                "translation": _json_ready(frame.get("translation")),
                "rotation": _json_ready(frame.get("rotation")),
                "torsion": _json_ready(frame.get("torsion")),
                "translationNorm": float(frame["translation_norm"]) if frame.get("translation_norm") is not None else None,
                "torsionNorm": float(frame["torsion_norm"]) if frame.get("torsion_norm") is not None else None,
            }
        )
    return frames


def _variant_label(sample: dict[str, Any], fallback_rank: int) -> str:
    stage = sample.get("stage")
    stage_sample_index = sample.get("stage_sample_index")
    if stage is None:
        return f"Candidate {fallback_rank}"
    if stage_sample_index is None:
        return f"Stage {int(stage)}"
    return f"Stage {int(stage)} / Sample {int(stage_sample_index)}"


def build_matcha_workspace(
    run_workdir: Path,
    run_name: str,
    receptor_path: Path,
    ligand_path: Path,
    physical_only: bool = False,
) -> dict[str, Any]:
    preds_root = run_workdir / "work" / "runs" / run_name
    dataset_name = "any_conf"
    metrics_path = preds_root / f"{dataset_name}_final_preds_merged.npy"
    metrics = np.load(metrics_path, allow_pickle=True).item()

    pb_filters = _load_pb_filters(preds_root, dataset_name)
    _enrich_metrics_with_pb_filters(metrics, pb_filters)
    _enrich_metrics_with_gnina_scores(metrics, preds_root / dataset_name / "scored_sdf_predictions")

    uid, mdata = next(iter(metrics.items()))
    sample_metrics = mdata["sample_metrics"]
    orig_mol = mdata["orig_mol"]
    best_pb_count = max(int(sample.get("posebusters_filters_passed_count_fast", 0)) for sample in sample_metrics)
    ranked = _rank_samples(sample_metrics)
    ranked_filtered = [
        (rank, sample_index, sample)
        for rank, sample_index, sample in ranked
        if not physical_only or int(sample.get("posebusters_filters_passed_count_fast", 0)) == best_pb_count
    ]
    ranked_to_use = ranked_filtered if ranked_filtered else ranked
    has_gnina = any("gnina_score" in sample for sample in sample_metrics)

    variants = []
    for variant_id, (rank, sample_index, sample) in enumerate(ranked_to_use, start=1):
        mol = _mol_with_positions(orig_mol, np.asarray(sample["pred_pos"]), f"{uid}_rank{rank}")
        primary_key = "gnina_score" if has_gnina else "posebusters_filters_passed_count_fast"
        primary_value = float(sample.get("gnina_score")) if has_gnina else float(sample.get("posebusters_filters_passed_count_fast", 0))
        variants.append(
            {
                "id": variant_id,
                "label": _variant_label(sample, rank),
                "sdf": _mol_to_sdf_record(mol),
                "rank": None,
                "rmsd": float(sample.get("error_estimate_0")) if sample.get("error_estimate_0") is not None else None,
                "score": float(sample.get("gnina_score")) if sample.get("gnina_score") is not None else float(sample.get("posebusters_filters_passed_count_fast", 0)),
                "smiles": None,
                "duplicateCount": None,
                "metricValues": {
                    "posebusters_filters_passed_count_fast": float(sample.get("posebusters_filters_passed_count_fast", 0)),
                    "gnina_score": float(sample.get("gnina_score")) if sample.get("gnina_score") is not None else None,
                    "error_estimate_0": float(sample.get("error_estimate_0")) if sample.get("error_estimate_0") is not None else None,
                    "buried_fraction": float(sample.get("buried_fraction")) if sample.get("buried_fraction") is not None else None,
                    "stage": float(sample.get("stage")) if sample.get("stage") is not None else None,
                    "stage_sample_index": float(sample.get("stage_sample_index")) if sample.get("stage_sample_index") is not None else None,
                },
                "primaryMetricKey": primary_key,
                "primaryMetricValue": primary_value,
                "sampleIndex": sample_index,
                "trajectoryFrames": _build_trajectory_frames(orig_mol, sample, uid, rank),
            }
        )

    return {
        "engine": "matcha",
        "runId": run_name,
        "receptor": {
            "filename": receptor_path.name,
            "format": receptor_path.suffix.lower().lstrip(".") or "pdb",
            "content": receptor_path.read_text(),
        },
        "ligand": {
            "filename": ligand_path.name,
            "format": ligand_path.suffix.lower().lstrip(".") or "sdf",
            "content": ligand_path.read_text(),
        },
        "variants": variants,
    }
