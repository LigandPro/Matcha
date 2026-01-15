"""
Docking worker - runs the docking pipeline with progress reporting.
Supports both single-ligand and batch processing modes.
"""

import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from matcha.tui.protocol import ProgressEvent, LigandStatus

if TYPE_CHECKING:
    from matcha.tui.backend import DockingJob


BANNER = r"""
███╗   ███╗ █████╗ ████████╗ ██████╗██╗  ██╗ █████╗
████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║  ██║██╔══██╗
██╔████╔██║███████║   ██║   ██║     ███████║███████║
██║╚██╔╝██║██╔══██║   ██║   ██║     ██╔══██║██╔══██║
██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗██║  ██║██║  ██║
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝
"""


def find_ligands(ligand_dir: Path, extensions: tuple = (".sdf", ".mol", ".mol2")) -> list[Path]:
    """Find all ligand files in a directory."""
    ligands = []
    for ext in extensions:
        ligands.extend(ligand_dir.glob(f"*{ext}"))
    return sorted(ligands, key=lambda p: p.name.lower())


def _format_runtime(seconds: float) -> str:
    secs = int(round(seconds))
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, rem = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{rem}s")
    return " ".join(parts)


def _split_multi_sdf(sdf_path: Path) -> list[tuple[str, "Chem.Mol"]]:
    from rdkit import Chem

    molecules: list[tuple[str, Chem.Mol]] = []
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    for idx, mol in enumerate(suppl):
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{idx}"
        name = name.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
        if not name:
            name = f"mol_{idx}"
        molecules.append((name, mol))
    return molecules


def _prepare_batch_dataset(
    protein: Path,
    molecules: list[tuple[str, "Chem.Mol"]],
    dataset_dir: Path,
) -> list[str]:
    from rdkit import Chem

    dataset_dir.mkdir(parents=True, exist_ok=True)
    uids: list[str] = []
    for name, mol in molecules:
        uid = name
        sample_dir = dataset_dir / uid
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(protein, sample_dir / f"{uid}_protein.pdb")
        writer = Chem.SDWriter(str(sample_dir / f"{uid}_ligand.sdf"))
        writer.write(mol)
        writer.close()
        uids.append(uid)
    return uids


def _create_batch_pocket_centers(
    pocket_center: tuple[float, float, float],
    molecule_uids: list[str],
    n_samples: int,
    output_path: Path,
) -> None:
    import numpy as np

    pocket_centers: dict[str, list[dict]] = {}
    ligand_center = np.array(pocket_center)
    protein_center = np.zeros(3)
    for uid in molecule_uids:
        for conf_idx in range(n_samples):
            pocket_centers[f"{uid}_mol0_conf{conf_idx}"] = [{
                "tr_pred_init": ligand_center,
                "full_protein_center": protein_center,
            }]
    np.save(output_path, [pocket_centers])


def _autobox_from_ligand(ligand: Path) -> tuple[float, float, float]:
    import numpy as np
    from rdkit import Chem

    mol = Chem.MolFromMolFile(str(ligand), removeHs=False, sanitize=False) if ligand.suffix.lower() in {".mol", ".mol2"} else None
    if mol is None and ligand.suffix.lower() == ".pdb":
        mol = Chem.MolFromPDBFile(str(ligand), removeHs=False, sanitize=False)
    if mol is None:
        suppl = Chem.SDMolSupplier(str(ligand), removeHs=False, sanitize=False)
        mol = suppl[0] if suppl and len(suppl) > 0 else None
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Failed to read ligand for autobox: {ligand}")
    mol = Chem.RemoveAllHs(mol, sanitize=False)
    return tuple(mol.GetConformer().GetPositions().mean(axis=0).tolist())


def _rank_samples(sample_metrics: list[dict]) -> list[tuple[int, dict]]:
    ranked_indices = sorted(
        range(len(sample_metrics)),
        key=lambda i: (
            -int(sample_metrics[i].get("posebusters_filters_passed_count_fast", 0)),
            float(sample_metrics[i].get("error_estimate_0", float("inf"))),
        ),
    )
    return [(rank, sample_metrics[i]) for rank, i in enumerate(ranked_indices, start=1)]


def _save_all_poses_for_uid(
    metrics_data: dict,
    uid: str,
    out_path: Path,
    filter_non_physical: bool = True,
) -> tuple[list[tuple[int, dict]], int, int]:
    import copy
    from rdkit import Chem

    if uid not in metrics_data:
        return [], 0, 0
    sample_data = metrics_data[uid]
    orig_mol = sample_data["orig_mol"]
    ranked = _rank_samples(sample_data["sample_metrics"])
    best_pb_count = max([
        int(s.get("posebusters_filters_passed_count_fast", 0))
        for s in sample_data["sample_metrics"]
    ])

    ranked_filtered = [
        (r, s) for r, s in ranked
        if not filter_non_physical or int(s.get("posebusters_filters_passed_count_fast", 0)) == best_pb_count
    ]
    ranked_to_use = ranked_filtered if ranked_filtered else ranked

    writer = Chem.SDWriter(str(out_path))
    for rank, sample in ranked_to_use:
        mol = copy.deepcopy(orig_mol)
        conf = Chem.Conformer(orig_mol.GetNumAtoms())
        for idx, (x, y, z) in enumerate(sample["pred_pos"]):
            conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        mol.SetProp("_Name", f"{uid}_rank{rank}")
        writer.write(mol)
    writer.close()
    return ranked_to_use, len(ranked_filtered), len(ranked)


def _get_best_sample_idx(errs, pb_counts) -> int:
    best_pb_count = max(pb_counts)
    pb_count_indices = [i for i, v in enumerate(pb_counts) if v == best_pb_count]
    best_score_idx = min(pb_count_indices, key=lambda i: errs[i])
    return best_score_idx


def _write_log_file(
    log_path: Path,
    run_name: str,
    output_dir: Path,
    receptor: str,
    ligand: Optional[Path],
    ligand_dir: Optional[str],
    n_samples: int,
    box_mode: str,
    center: Optional[tuple],
    all_poses: list,
    physical_only: bool,
    runtime: float = 0.0,
) -> None:
    """Write log file similar to CLI output."""
    import numpy as np

    timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    log_lines = [
        "MATCHA DOCKING ENGINE (TUI)",
        "============================================================",
        "",
        "[ RUN INFO ]",
        f"  Start time       : {timestamp}",
        f"  Run name         : {run_name}",
        f"  Output directory : {output_dir}",
        f"  Total runtime    : {runtime:.1f}s",
        "",
    ]

    if box_mode == "manual" and center:
        log_lines.extend([
            "[ DOCKING BOX ]",
            f"  Mode             : manual center",
            f"  Center (Å)       : ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})",
            "",
        ])
    else:
        log_lines.extend([
            "[ DOCKING MODE ]",
            f"  Mode             : {box_mode} docking",
            "",
        ])

    log_lines.extend([
        "[ INPUT FILES ]",
        f"  Receptor         : {receptor}",
    ])

    if ligand:
        log_lines.append(f"  Ligand           : {ligand}")
    elif ligand_dir:
        log_lines.append(f"  Ligand directory : {ligand_dir}")

    log_lines.extend([
        "",
        "[ OUTPUT FILES ]",
        f"  Best pose SDF    : {output_dir / f'{run_name}_best.sdf'}",
        f"  All poses SDF    : {output_dir / f'{run_name}_poses.sdf'}",
        f"  Log file         : {log_path}",
        "",
    ])

    if all_poses:
        errs = [p.get("error_estimate", 0) for p in all_poses]
        pb_counts = [p.get("pb_count", 0) for p in all_poses]

        log_lines.extend([
            "[ SUMMARY ]",
            f"  Samples generated      : {n_samples}",
            f"  Total poses            : {len(all_poses)}",
            f"  error_estimate (Å)     : min={min(errs):.3f}, mean={float(np.mean(errs)):.3f}, max={max(errs):.3f}",
            f"  posebusters checks     : min={min(pb_counts)}/4, max={max(pb_counts)}/4",
            f"  Physical-only filter   : {physical_only}",
            "",
            "[ TOP 10 POSES ]",
            "  rank  error_est(Å)  pb_checks",
            "  ---------------------------------",
        ])

        for i, pose in enumerate(all_poses[:10], 1):
            err = pose.get("error_estimate", 0)
            pb = pose.get("pb_count", 0)
            log_lines.append(f"  {i:4d}  {err:12.3f}  {pb:2d}/4")

        log_lines.extend([
            "",
            "Note: Lower error_estimate values indicate higher confidence.",
            "      Negative values = high confidence, positive = uncertainty.",
        ])
    else:
        log_lines.extend([
            "[ SUMMARY ]",
            "  No poses generated.",
        ])

    log_path.write_text("\n".join(log_lines) + "\n")


def run_docking(job: "DockingJob") -> None:
    """Run docking job with progress reporting."""
    config = job.config
    emit = job.progress_callback
    job_start_time = time.time()
    job_start_dt = datetime.now(timezone.utc)

    try:
        # Extract GPU FIRST and set CUDA_VISIBLE_DEVICES BEFORE any PyTorch imports
        gpu = config.get("gpu")
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # Extract rest of config
        receptor_path = config.get("receptor")
        ligand_path = config.get("ligand")
        ligand_dir = config.get("ligand_dir")
        output_dir = config.get("output_dir", "./results")
        run_name = config.get("run_name", "matcha_tui_run")
        n_samples = config.get("n_samples", 40)
        n_confs = config.get("n_confs")
        checkpoints = config.get("checkpoints")
        physical_only = config.get("physical_only", False)

        # Box settings
        box_mode = config.get("box_mode", "blind")
        center_x = config.get("center_x")
        center_y = config.get("center_y")
        center_z = config.get("center_z")
        autobox_ligand = config.get("autobox_ligand")
        batch_mode = ligand_dir is not None
        if (ligand_path is None) == (ligand_dir is None):
            emit(ProgressEvent(type="error", message="Specify either ligand or ligand_dir"))
            return

        # Import matcha modules (once for all ligands)
        # GPU was already set via CUDA_VISIBLE_DEVICES at the start of this function
        emit(ProgressEvent(type="stage_start", stage="init", name="Initializing"))
        if gpu is not None:
            emit(ProgressEvent(type="info", message=f"Using CUDA device #{gpu}"))

        from omegaconf import OmegaConf
        from matcha.utils.esm_utils import compute_esm_embeddings, compute_sequences
        from matcha.utils.inference_utils import (
            run_inference_pipeline,
            compute_fast_filters,
            save_best_pred_to_sdf,
        )

        emit(ProgressEvent(type="stage_progress", stage="init", progress=50))

        # Setup paths
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        run_workdir = output_path / run_name / "work"
        run_workdir.mkdir(parents=True, exist_ok=True)

        emit(ProgressEvent(type="stage_done", stage="init", elapsed=0))

        if job.is_cancelled():
            emit(ProgressEvent(type="cancelled", message="Job cancelled"))
            return

        # Load checkpoints (once for all ligands)
        emit(ProgressEvent(type="stage_start", stage="checkpoints", name="Loading checkpoints"))
        start_time = time.time()

        if checkpoints:
            checkpoint_path = Path(checkpoints).expanduser().resolve()
        else:
            from huggingface_hub import snapshot_download

            emit(ProgressEvent(
                type="stage_progress",
                stage="checkpoints",
                progress=10,
                message="Downloading from HuggingFace...",
            ))
            checkpoint_path = Path(
                snapshot_download("LigandPro/Matcha", allow_patterns="pipeline/*")
            )

        emit(ProgressEvent(
            type="stage_done", stage="checkpoints", elapsed=time.time() - start_time
        ))

        if job.is_cancelled():
            emit(ProgressEvent(type="cancelled", message="Job cancelled"))
            return

        receptor = Path(receptor_path).expanduser().resolve()
        if not receptor.exists():
            emit(ProgressEvent(type="error", message=f"Receptor file not found: {receptor_path}"))
            return

        # Batch mode: match CLI behavior
        if batch_mode:
            ligand_dir_path = Path(ligand_dir).expanduser().resolve()
            box_center_val = None
            if not ligand_dir_path.exists():
                emit(ProgressEvent(type="error", message=f"Ligand path not found: {ligand_dir}"))
                return

            if ligand_dir_path.is_file():
                if ligand_dir_path.suffix.lower() != ".sdf":
                    emit(ProgressEvent(type="error", message="Batch ligand file must be .sdf"))
                    return
                molecules = _split_multi_sdf(ligand_dir_path)
            elif ligand_dir_path.is_dir():
                molecules = []
                for sdf_file in sorted(ligand_dir_path.glob("*.sdf")):
                    molecules.extend(_split_multi_sdf(sdf_file))
            else:
                emit(ProgressEvent(type="error", message="ligand_dir must be a file or directory"))
                return

            if not molecules:
                emit(ProgressEvent(type="error", message=f"No molecules found in {ligand_dir}"))
                return

            molecule_uids = sorted([name for name, _ in molecules])
            total_ligands = len(molecule_uids)
            ligand_statuses = [
                LigandStatus(name=name, path=str(ligand_dir_path), status="pending")
                for name in molecule_uids
            ]

            emit(ProgressEvent(
                type="batch_start",
                total_ligands=total_ligands,
                message=f"Processing {total_ligands} ligands",
                ligand_statuses=[ls.to_dict() for ls in ligand_statuses],
            ))

            if job.is_cancelled():
                emit(ProgressEvent(type="cancelled", message="Job cancelled"))
                return

            # Prepare dataset for all ligands
            emit(ProgressEvent(type="stage_start", stage="dataset", name="Preparing dataset"))
            dataset_start = time.time()

            dataset_dir = run_workdir / "datasets" / "any_conf"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            _prepare_batch_dataset(receptor, molecules, dataset_dir)

            pocket_centers_filename = None
            if box_mode == "manual":
                if not all(v is not None for v in [center_x, center_y, center_z]):
                    emit(ProgressEvent(type="error", message="Manual box requires center_x/center_y/center_z"))
                    return
                pocket_centers_filename = run_workdir / "stage1_any_conf.npy"
                box_center_val = (center_x, center_y, center_z)
                _create_batch_pocket_centers(
                    (center_x, center_y, center_z),
                    molecule_uids,
                    n_samples,
                    pocket_centers_filename,
                )
            elif box_mode == "autobox" and autobox_ligand:
                try:
                    box_center = _autobox_from_ligand(Path(autobox_ligand).expanduser().resolve())
                except Exception as exc:
                    emit(ProgressEvent(type="error", message=str(exc)))
                    return
                box_center_val = box_center
                pocket_centers_filename = run_workdir / "stage1_any_conf.npy"
                _create_batch_pocket_centers(
                    box_center,
                    molecule_uids,
                    n_samples,
                    pocket_centers_filename,
                )

            emit(ProgressEvent(
                type="stage_done",
                stage="dataset",
                elapsed=time.time() - dataset_start,
            ))

            if job.is_cancelled():
                emit(ProgressEvent(type="cancelled", message="Job cancelled"))
                return

            # Build config (CLI-like)
            conf = OmegaConf.create({
                "seed": 42,
                "tr_std": 5.0,
                "use_time": True,
                "dropout_rate": 0.0,
                "num_kernel_pos_encoder": 128,
                "objective": "ranking",
                "llm_emb_dim": 480,
                "use_all_chains": True,
                "checkpoints_folder": str(checkpoint_path),
                "cache_path": str(run_workdir / "cache"),
                "data_folder": str(run_workdir / "data"),
                "inference_results_folder": str(run_workdir / "runs"),
                "any_data_dir": str(dataset_dir),
                "test_dataset_types": ["any_conf"],
            })

            # ESM embeddings (once for batch)
            emit(ProgressEvent(type="stage_start", stage="esm", name="Computing ESM embeddings"))
            esm_start = time.time()
            compute_sequences(conf)
            emit(ProgressEvent(type="stage_progress", stage="esm", progress=30))
            if job.is_cancelled():
                emit(ProgressEvent(type="cancelled", message="Job cancelled"))
                return
            compute_esm_embeddings(conf)
            emit(ProgressEvent(
                type="stage_done",
                stage="esm",
                elapsed=time.time() - esm_start,
            ))

            if job.is_cancelled():
                emit(ProgressEvent(type="cancelled", message="Job cancelled"))
                return

            # Run inference pipeline for batch
            def inference_progress_callback(event_type, stage, name, elapsed=None, progress=None):
                if event_type == "stage_start":
                    emit(ProgressEvent(type="stage_start", stage=stage, name=name))
                elif event_type == "stage_progress":
                    emit(ProgressEvent(type="stage_progress", stage=stage, progress=progress))
                elif event_type == "stage_done":
                    emit(ProgressEvent(type="stage_done", stage=stage, elapsed=elapsed))

            run_inference_pipeline(
                conf,
                run_name,
                n_samples,
                pocket_centers_filename,
                num_workers=0,
                progress_callback=inference_progress_callback,
            )

            if job.is_cancelled():
                emit(ProgressEvent(type="cancelled", message="Job cancelled"))
                return

            # PoseBusters
            emit(ProgressEvent(
                type="stage_start",
                stage="posebusters",
                name="Physical validation",
            ))
            pose_start = time.time()
            compute_fast_filters(conf, run_name, n_samples)
            emit(ProgressEvent(
                type="stage_done",
                stage="posebusters",
                elapsed=time.time() - pose_start,
            ))

            # Save best predictions
            save_best_pred_to_sdf(conf, run_name)

            preds_root = Path(conf.inference_results_folder) / run_name
            dataset_name = "any_conf"
            fast_metrics_path = preds_root / f"{dataset_name}_final_preds_fast_metrics.npy"
            import numpy as np

            metrics = np.load(fast_metrics_path, allow_pickle=True).item()

            output_run_dir = output_path / run_name
            best_dir = output_run_dir / "best_poses"
            all_dir = output_run_dir / "all_poses"
            logs_dir = output_run_dir / "logs"
            best_dir.mkdir(parents=True, exist_ok=True)
            all_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            sdf_preds_dir = preds_root / dataset_name / "sdf_predictions"
            for idx, mol_uid in enumerate(molecule_uids):
                metrics_key = mol_uid if mol_uid in metrics else f"{mol_uid}_mol0"
                if metrics_key not in metrics:
                    ligand_statuses[idx].status = "failed"
                    ligand_statuses[idx].error_message = "No results"
                    emit(ProgressEvent(
                        type="batch_progress",
                        progress=int(100 * (idx + 1) / total_ligands),
                        ligand_index=idx,
                        total_ligands=total_ligands,
                        ligand_statuses=[ls.to_dict() for ls in ligand_statuses],
                    ))
                    continue

                mdata = metrics[metrics_key]
                errs = np.array([float(s.get("error_estimate_0", float("inf"))) for s in mdata["sample_metrics"]])
                pb_counts = np.array([int(s.get("posebusters_filters_passed_count_fast", 0)) for s in mdata["sample_metrics"]])
                best_idx = _get_best_sample_idx(errs, pb_counts)

                pred_sdf_src = sdf_preds_dir / f"{mol_uid}.sdf"
                best_dest = best_dir / f"{mol_uid}.sdf"
                all_dest = all_dir / f"{mol_uid}_poses.sdf"

                if pred_sdf_src.exists():
                    shutil.copyfile(pred_sdf_src, best_dest)

                ranked_samples, kept_physical, total_samples = _save_all_poses_for_uid(
                    metrics, metrics_key, all_dest, filter_non_physical=physical_only
                )

                ligand_statuses[idx].status = "completed"
                if ranked_samples:
                    ligand_statuses[idx].error_estimate = float(ranked_samples[0][1].get("error_estimate_0", 0))
                    ligand_statuses[idx].pb_count = int(ranked_samples[0][1].get("posebusters_filters_passed_count_fast", 0))

                # Per-ligand log (CLI style)
                ligand_input = (run_workdir / "datasets" / "any_conf" / mol_uid / f"{mol_uid}_ligand.sdf").resolve()
                per_log_path = logs_dir / f"{mol_uid}.log"
                runtime_local = time.time() - job_start_time
                log_lines = [
                    BANNER.rstrip("\n"),
                    "MATCHA DOCKING ENGINE  v0.1.0",
                    "============================================================",
                    "",
                    "",
                    "[ RUN INFO ]",
                    f"  Start time       : {job_start_dt.isoformat()}Z",
                    f"  Command          : batch run {run_name} (ligand {mol_uid})",
                    f"  Workdir          : {output_run_dir.resolve()}",
                    f"  Runtime          : {_format_runtime(runtime_local)}",
                    "",
                    "",
                ]

                if box_mode == "manual":
                    log_lines.extend([
                        "[ AUTODOCKING BOX ]",
                        "  Mode             : manual center",
                        f"  Center (Å)       : ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})",
                        "",
                        "",
                    ])
                elif box_mode == "autobox" and autobox_ligand:
                    log_lines.extend([
                        "[ AUTODOCKING BOX ]",
                        f"  Mode             : autobox from {Path(autobox_ligand).name}",
                        "",
                        "",
                    ])
                    if box_center_val is not None:
                        log_lines.insert(-2, f"  Center (Å)       : ({box_center_val[0]:.3f}, {box_center_val[1]:.3f}, {box_center_val[2]:.3f})")
                else:
                    log_lines.extend([
                        "[ DOCKING MODE ]",
                        "  Mode             : blind docking (entire protein)",
                        "",
                        "",
                    ])

                log_lines.extend([
                    "[ INPUT / OUTPUT FILES ]",
                    f"  Receptor         : {receptor.resolve()}",
                    f"  Ligand           : {ligand_input}",
                    f"  Best pose SDF    : {best_dest.resolve()}",
                    f"  All poses SDF    : {all_dest.resolve()}",
                    f"  Log file         : {per_log_path.resolve()}",
                    "",
                    "",
                    "[ SUMMARY ]",
                    f"  Samples per ligand     : {n_samples}",
                    f"  error_estimate_0 (Å)   : min={min(errs):.3f}, mean={float(np.mean(errs)):.3f}, max={max(errs):.3f}",
                    f"  posebusters_fast checks: min={min(pb_counts)}/4, max={max(pb_counts)}/4",
                    f"  Best sample            : rank={best_idx + 1}, error_estimate_0={errs[best_idx]:.3f}, posebusters_pass_fast={pb_counts[best_idx]}/4",
                    f"  Filtered poses (pb_4/4): kept {kept_physical}/{total_samples}" + ("  [WARNING: none passed, keeping originals]" if physical_only and kept_physical == 0 else ""),
                    "",
                    "",
                    "  Note: error_estimate_0 is a lower-is-better RMSD-like metric (Å).",
                    "        Negative values indicate high model confidence, positive values indicate uncertainty.",
                    "        Lower values = higher confidence in the predicted pose.",
                    "",
                    "",
                    "  PoseBusters checks (4 boolean tests):",
                    "    1. not_too_far_away   : ligand is close to protein (distance check)",
                    "    2. no_internal_clash  : no bad bonds/angles in ligand geometry",
                    "    3. no_clashes         : no inter-molecular clashes (ligand-protein)",
                    "    4. no_volume_clash    : no vdW volume overlaps",
                    "  Additional metric:",
                    "    - buried_fraction     : fraction of ligand buried in protein (shown separately)",
                    "",
                    "[ POSE RANKING ]",
                    "  mode  error_est_0(Å)  pb_4/4  not_far  no_int_clash  no_clash  no_vol_clash  buried_frac",
                    "  --------------------------------------------------------------------------------------------",
                ])
                for mode, sample in ranked_samples:
                    err = float(sample.get("error_estimate_0", float("inf")))
                    pb_count_val = int(sample.get("posebusters_filters_passed_count_fast", 0))
                    pb_flags = sample.get("posebusters_filters_fast", [])
                    not_far = "✓" if len(pb_flags) > 0 and pb_flags[0] else "✗"
                    no_int_clash = "✓" if len(pb_flags) > 1 and pb_flags[1] else "✗"
                    no_clash = "✓" if len(pb_flags) > 2 and pb_flags[2] else "✗"
                    no_vol_clash = "✓" if len(pb_flags) > 3 and pb_flags[3] else "✗"
                    buried_frac = f"{pb_flags[4]:.2f}" if len(pb_flags) > 4 else "n/a"
                    log_lines.append(
                        f"  {mode:<3}     {err:>7.3f}       {pb_count_val}/4      {not_far:^4}     {no_int_clash:^8}     {no_clash:^5}     {no_vol_clash:^8}      {buried_frac:>6}"
                    )

                log_lines.extend([
                    "",
                    "Legend: ✓ = passed, ✗ = failed",
                    "",
                    "",
                ])

                if any(e > 0 for e in errs):
                    log_lines.extend([
                        "WARNING: Some poses have positive error_est_0 values.",
                        "         This indicates reduced model confidence in those predictions.",
                        "         Consider generating more samples or inspecting poses manually.",
                        "",
                        "",
                    ])
                if max(pb_counts) < 4:
                    log_lines.extend([
                        "WARNING: No poses passed all PoseBusters checks (pb_4/4).",
                        "         Inspect poses or regenerate with more samples/box adjustments.",
                        "",
                        "",
                    ])

                log_lines.extend([
                    "[ END ]",
                    f"  Run finished at    : {datetime.now(timezone.utc).isoformat()}Z",
                    f"  Total runtime      : {_format_runtime(runtime_local)}",
                    f"  Workdir preserved  : {output_run_dir.resolve()}",
                    "============================================================",
                ])

                per_log_path.write_text("\n".join(log_lines))

                emit(ProgressEvent(
                    type="batch_progress",
                    progress=int(100 * (idx + 1) / total_ligands),
                    ligand_index=idx,
                    total_ligands=total_ligands,
                    ligand_statuses=[ls.to_dict() for ls in ligand_statuses],
                ))

            # Batch summary log
            end_time = datetime.now(timezone.utc)
            runtime = (end_time - datetime.fromtimestamp(job_start_time, tz=timezone.utc)).total_seconds()
            summary_log = output_run_dir / f"{run_name}.log"
            command_parts = [
                "matcha-tui",
                "-r",
                str(receptor.resolve()),
                "--ligand-dir",
                str(ligand_dir_path.resolve()),
                "-o",
                str(output_path.resolve()),
                "--run-name",
                run_name,
                "--n-samples",
                str(n_samples),
            ]
            if n_confs is not None:
                command_parts.extend(["--n-confs", str(n_confs)])
            if gpu is not None:
                command_parts.extend(["--gpu", str(gpu)])
            if checkpoints:
                command_parts.extend(["--checkpoints", str(checkpoints)])
            if physical_only:
                command_parts.append("--physical-only")
            if box_mode == "manual":
                command_parts.extend([
                    "--center-x", str(center_x),
                    "--center-y", str(center_y),
                    "--center-z", str(center_z),
                ])
            elif box_mode == "autobox" and autobox_ligand:
                command_parts.extend(["--autobox-ligand", str(autobox_ligand)])

            command_line = " ".join(command_parts)

            log_lines = [
                BANNER.rstrip("\n"),
                "MATCHA DOCKING ENGINE  v0.1.0",
                "============================================================",
                "",
                "",
                "[ RUN INFO ]",
                f"  Start time       : {job_start_dt.isoformat()}Z",
                f"  Command          : {command_line}",
                f"  Workdir          : {output_run_dir.resolve()}",
                f"  Runtime          : {_format_runtime(runtime)}",
                "",
                "",
                "[ INPUT FILES ]",
                f"  Receptor         : {receptor.resolve()}",
                f"  Ligands          : {ligand_dir_path.resolve()} ({len(molecule_uids)} molecules)",
                f"  Output dir       : {output_run_dir.resolve()}",
                "",
                "",
                "[ PROCESSING SUMMARY ]",
                f"  Samples per molecule : {n_samples}",
                f"  Total molecules      : {len(molecule_uids)}",
                "",
                "",
                "[ RESULTS ]",
            ]
            for mol_uid in molecule_uids:
                metrics_key = mol_uid if mol_uid in metrics else f"{mol_uid}_mol0"
                if metrics_key not in metrics:
                    log_lines.append(f"  {mol_uid}: No results")
                    continue
                mdata = metrics[metrics_key]
                errs = np.array([float(s.get("error_estimate_0", float("inf"))) for s in mdata["sample_metrics"]])
                pb_counts = np.array([int(s.get("posebusters_filters_passed_count_fast", 0)) for s in mdata["sample_metrics"]])
                best_idx = _get_best_sample_idx(errs, pb_counts)
                log_lines.append(f"  {mol_uid}: Best err={errs[best_idx]:.3f}, pb={pb_counts[best_idx]}/4")

            log_lines.extend([
                "",
                "[ END ]",
                f"  Run finished at    : {end_time.isoformat()}Z",
                f"  Total runtime      : {_format_runtime(runtime)}",
                f"  Workdir preserved  : {output_run_dir.resolve()}",
                "============================================================",
            ])
            summary_log.write_text("\n".join(log_lines))

            emit(ProgressEvent(
                type="job_done",
                stage="done",
                output_path=str(output_run_dir),
                poses=[],
                total_ligands=total_ligands,
                ligand_statuses=[ls.to_dict() for ls in ligand_statuses],
            ))
            return

        # Single-ligand path
        # Determine ligand list
        ligands: list[Path] = []
        if ligand_path:
            ligands = [Path(ligand_path).expanduser().resolve()]

        if not ligands:
            emit(ProgressEvent(type="error", message="No ligand files found"))
            return

        total_ligands = len(ligands)
        ligand_statuses = [
            LigandStatus(name=lig.name, path=str(lig), status="pending")
            for lig in ligands
        ]

        if job.is_cancelled():
            emit(ProgressEvent(type="cancelled", message="Job cancelled"))
            return

        # Process each ligand
        receptor = Path(receptor_path).expanduser().resolve()
        all_results = []

        for idx, ligand in enumerate(ligands):
            ligand_name = ligand.stem
            ligand_statuses[idx].status = "running"

            emit(ProgressEvent(
                type="ligand_start",
                current_ligand=ligand.name,
                ligand_index=idx,
                total_ligands=total_ligands,
                message=f"Processing {ligand.name}",
                ligand_statuses=[ls.to_dict() for ls in ligand_statuses],
            ))

            try:
                result = process_single_ligand(
                    job=job,
                    emit=emit,
                    receptor=receptor,
                    ligand=ligand,
                    run_workdir=run_workdir,
                    run_name=run_name,
                    checkpoint_path=checkpoint_path,
                    n_samples=n_samples,
                    n_confs=n_confs,
                    box_mode=box_mode,
                    center_x=center_x,
                    center_y=center_y,
                    center_z=center_z,
                    autobox_ligand=autobox_ligand,
                    physical_only=physical_only,
                    ligand_index=idx,
                    total_ligands=total_ligands,
                    OmegaConf=OmegaConf,
                    compute_sequences=compute_sequences,
                    compute_esm_embeddings=compute_esm_embeddings,
                    run_inference_pipeline=run_inference_pipeline,
                    compute_fast_filters=compute_fast_filters,
                    save_best_pred_to_sdf=save_best_pred_to_sdf,
                )

                if result is None:
                    # Job was cancelled
                    return

                ligand_statuses[idx].status = "completed"
                if result.get("poses"):
                    best_pose = result["poses"][0]
                    ligand_statuses[idx].error_estimate = best_pose.get("error_estimate")
                    ligand_statuses[idx].pb_count = best_pose.get("pb_count")

                all_results.append({
                    "ligand": ligand.name,
                    "ligand_path": str(ligand),
                    **result,
                })

            except Exception as e:
                ligand_statuses[idx].status = "failed"
                ligand_statuses[idx].error_message = str(e)
                all_results.append({
                    "ligand": ligand.name,
                    "ligand_path": str(ligand),
                    "error": str(e),
                })

            emit(ProgressEvent(
                type="ligand_done",
                current_ligand=ligand.name,
                ligand_index=idx,
                total_ligands=total_ligands,
                ligand_statuses=[ls.to_dict() for ls in ligand_statuses],
            ))

        # Save summary
        output_run_dir = output_path / run_name
        output_run_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate best poses
        all_poses = []
        for result in all_results:
            if "poses" in result:
                for pose in result["poses"]:
                    pose["ligand"] = result["ligand"]
                all_poses.extend(result["poses"])

        # DEBUG: Log poses count and first pose
        import sys
        print(f"[DEBUG] all_results count: {len(all_results)}", file=sys.stderr)
        print(f"[DEBUG] all_poses count: {len(all_poses)}", file=sys.stderr)
        if all_poses:
            print(f"[DEBUG] First pose: {all_poses[0]}", file=sys.stderr)

        # Calculate total runtime
        total_runtime = time.time() - job_start_time

        # Create log file (like CLI does)
        log_path = output_run_dir / f"{run_name}.log"
        _write_log_file(
            log_path=log_path,
            run_name=run_name,
            output_dir=output_run_dir,
            receptor=receptor_path,
            ligand=ligands[0] if len(ligands) == 1 else None,
            ligand_dir=ligand_dir,
            n_samples=n_samples,
            box_mode=box_mode,
            center=(center_x, center_y, center_z) if box_mode == "manual" else None,
            all_poses=all_poses,
            physical_only=physical_only,
            runtime=total_runtime,
        )

        emit(ProgressEvent(
            type="job_done",
            stage="done",
            output_path=str(output_run_dir),
            poses=all_poses[:10] if all_poses else [],
            total_ligands=total_ligands,
            ligand_statuses=[ls.to_dict() for ls in ligand_statuses],
        ))

    except Exception as e:
        emit(ProgressEvent(type="error", message=str(e)))
        raise


def process_single_ligand(
    job: "DockingJob",
    emit,
    receptor: Path,
    ligand: Path,
    run_workdir: Path,
    run_name: str,
    checkpoint_path: Path,
    n_samples: int,
    n_confs: Optional[int],
    box_mode: str,
    center_x: Optional[float],
    center_y: Optional[float],
    center_z: Optional[float],
    autobox_ligand: Optional[str],
    physical_only: bool,
    ligand_index: int,
    total_ligands: int,
    # Pre-imported modules
    OmegaConf,
    compute_sequences,
    compute_esm_embeddings,
    run_inference_pipeline,
    compute_fast_filters,
    save_best_pred_to_sdf,
) -> Optional[dict]:
    """Process a single ligand through the docking pipeline."""
    from rdkit import Chem

    ligand_name = ligand.stem
    uid = f"sample_{ligand_index}"

    # Create dataset directory for this ligand
    dataset_dir = run_workdir / "datasets" / "any_conf"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = dataset_dir / uid
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    emit(ProgressEvent(
        type="stage_start",
        stage="dataset",
        name="Preparing dataset",
        current_ligand=ligand.name,
        ligand_index=ligand_index,
        total_ligands=total_ligands,
    ))
    start_time = time.time()

    shutil.copyfile(receptor, sample_dir / f"{uid}_protein.pdb")

    # Normalize ligand to SDF
    ext = ligand.suffix.lower()
    if ext == ".sdf":
        shutil.copyfile(ligand, sample_dir / f"{uid}_ligand.sdf")
    else:
        if ext in {".mol", ".mol2"}:
            mol = Chem.MolFromMolFile(str(ligand), removeHs=False, sanitize=False)
        else:
            mol = Chem.MolFromPDBFile(str(ligand), removeHs=False, sanitize=False)
        writer = Chem.SDWriter(str(sample_dir / f"{uid}_ligand.sdf"))
        writer.write(mol)
        writer.close()

    emit(ProgressEvent(
        type="stage_done",
        stage="dataset",
        elapsed=time.time() - start_time,
        current_ligand=ligand.name,
        ligand_index=ligand_index,
        total_ligands=total_ligands,
    ))

    if job.is_cancelled():
        emit(ProgressEvent(type="cancelled", message="Job cancelled"))
        return None

    # Prepare data directory
    data_dir = run_workdir / "data" / ligand_name
    data_dir.mkdir(parents=True, exist_ok=True)

    # Build config
    conf = OmegaConf.create({
        "seed": 42,
        "tr_std": 5.0,
        "use_time": True,
        "dropout_rate": 0.0,
        "num_kernel_pos_encoder": 128,
        "objective": "ranking",
        "llm_emb_dim": 480,
        "use_all_chains": True,
        "checkpoints_folder": str(checkpoint_path),
        "cache_path": str(run_workdir / "cache"),
        "inference_results_folder": str(run_workdir / "runs" / ligand_name),
        "any_data_dir": str(run_workdir / "datasets" / "any_conf"),
        "test_dataset_types": ["any_conf"],
        "data_folder": str(data_dir),
    })

    # ESM embeddings
    emit(ProgressEvent(
        type="stage_start",
        stage="esm",
        name="Computing ESM embeddings",
        current_ligand=ligand.name,
        ligand_index=ligand_index,
        total_ligands=total_ligands,
    ))
    start_time = time.time()

    compute_sequences(conf)
    emit(ProgressEvent(
        type="stage_progress",
        stage="esm",
        progress=30,
        current_ligand=ligand.name,
        ligand_index=ligand_index,
        total_ligands=total_ligands,
    ))

    if job.is_cancelled():
        emit(ProgressEvent(type="cancelled", message="Job cancelled"))
        return None

    compute_esm_embeddings(conf)
    emit(ProgressEvent(
        type="stage_done",
        stage="esm",
        elapsed=time.time() - start_time,
        current_ligand=ligand.name,
        ligand_index=ligand_index,
        total_ligands=total_ligands,
    ))

    if job.is_cancelled():
        emit(ProgressEvent(type="cancelled", message="Job cancelled"))
        return None

    # Pocket centers file (if specified)
    import copy
    pocket_centers_filename = None
    if box_mode == "manual" and all(v is not None for v in [center_x, center_y, center_z]):
        import numpy as np
        pocket_centers_filename = run_workdir / "pocket_centers.npy"
        pocket_centers = {}
        ligand_center = np.array([center_x, center_y, center_z])
        protein_center = np.zeros(3)
        for i in range(n_samples):
            pocket_centers[f'{uid}_mol0_conf{i}'] = [{'tr_pred_init': ligand_center, 'full_protein_center': protein_center}]
        np.save(pocket_centers_filename, [pocket_centers])
    elif box_mode == "autobox" and autobox_ligand:
        import numpy as np
        autobox_path = Path(autobox_ligand).expanduser().resolve()
        mol = Chem.MolFromMolFile(str(autobox_path), removeHs=False, sanitize=False)
        if mol is not None:
            pocket_centers_filename = run_workdir / "pocket_centers.npy"
            conformer = mol.GetConformer()
            coords = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            center = coords.mean(axis=0)
            pocket_centers = {}
            protein_center = np.zeros(3)
            for i in range(n_samples):
                pocket_centers[f'{uid}_mol0_conf{i}'] = [{'tr_pred_init': center, 'full_protein_center': protein_center}]
            np.save(pocket_centers_filename, [pocket_centers])

    # Run inference pipeline with progress callback
    # Create a callback that emits progress for all stages (1, 2, 3, scoring)
    def inference_progress_callback(event_type, stage, name, elapsed=None, progress=None):
        """Callback to emit progress events from inference pipeline."""
        if event_type == 'stage_start':
            emit(ProgressEvent(
                type="stage_start",
                stage=stage,
                name=name,
                current_ligand=ligand.name,
                ligand_index=ligand_index,
                total_ligands=total_ligands,
            ))
        elif event_type == 'stage_progress':
            emit(ProgressEvent(
                type="stage_progress",
                stage=stage,
                progress=progress,
                current_ligand=ligand.name,
                ligand_index=ligand_index,
                total_ligands=total_ligands,
            ))
        elif event_type == 'stage_done':
            emit(ProgressEvent(
                type="stage_done",
                stage=stage,
                elapsed=elapsed,
                current_ligand=ligand.name,
                ligand_index=ligand_index,
                total_ligands=total_ligands,
            ))

    if job.is_cancelled():
        emit(ProgressEvent(type="cancelled", message="Job cancelled"))
        return None

    # Run inference pipeline - stages 1, 2, 3, and scoring will emit their own progress
    run_inference_pipeline(
        copy.deepcopy(conf),
        "any_conf",
        n_samples,
        pocket_centers_filename,
        num_workers=0,  # Use 0 workers to avoid multiprocessing deadlocks in TUI
        progress_callback=inference_progress_callback,  # Pass callback for all stages
    )

    if job.is_cancelled():
        emit(ProgressEvent(type="cancelled", message="Job cancelled"))
        return None

    # PoseBusters
    emit(ProgressEvent(
        type="stage_start",
        stage="posebusters",
        name="Physical validation",
        current_ligand=ligand.name,
        ligand_index=ligand_index,
        total_ligands=total_ligands,
    ))
    start_time = time.time()

    compute_fast_filters(conf, "any_conf", n_samples)

    emit(ProgressEvent(
        type="stage_done",
        stage="posebusters",
        elapsed=time.time() - start_time,
        current_ligand=ligand.name,
        ligand_index=ligand_index,
        total_ligands=total_ligands,
    ))

    # Save results for this ligand
    output_ligand_dir = run_workdir.parent / "outputs" / ligand_name
    output_ligand_dir.mkdir(parents=True, exist_ok=True)

    save_best_pred_to_sdf(conf, "any_conf")

    # Copy final SDF to top-level output directory (like CLI does)
    run_output_dir = run_workdir.parent  # This is output_path / run_name
    sdf_preds_dir = Path(conf.inference_results_folder) / "any_conf" / "any_conf" / "sdf_predictions"
    pred_sdf_files = list(sdf_preds_dir.glob("*.sdf"))
    if pred_sdf_files:
        # For batch mode, prefix with ligand name; for single mode, use run_name
        if total_ligands > 1:
            best_dest = run_output_dir / f"{ligand_name}_best.sdf"
            all_dest = run_output_dir / f"{ligand_name}_poses.sdf"
        else:
            best_dest = run_output_dir / f"{run_name}_best.sdf"
            all_dest = run_output_dir / f"{run_name}_poses.sdf"
        shutil.copyfile(pred_sdf_files[0], best_dest)
        shutil.copyfile(pred_sdf_files[0], all_dest)

    # Get pose info
    import numpy as np

    metrics_file = list(Path(conf.inference_results_folder).glob("**/*_final_preds_fast_metrics.npy"))
    poses = []
    if metrics_file:
        data = np.load(metrics_file[0], allow_pickle=True).item()
        for uid_key, uid_data in data.items():
            # uid_data is a dict with 'sample_metrics' and 'orig_mol' keys
            sample_metrics = uid_data.get('sample_metrics', [])
            for i, sample in enumerate(sample_metrics[:10]):
                pb_filters = sample.get("posebusters_filters_fast", [False] * 5)
                poses.append({
                    "rank": i + 1,
                    "error_estimate": float(sample.get("error_estimate_0", 0)),
                    "pb_count": int(sample.get("posebusters_filters_passed_count_fast", 0)),
                    "checks": {
                        "not_too_far_away": bool(pb_filters[0]) if len(pb_filters) > 0 else False,
                        "no_internal_clash": bool(pb_filters[1]) if len(pb_filters) > 1 else False,
                        "no_clashes": bool(pb_filters[2]) if len(pb_filters) > 2 else False,
                        "no_volume_clash": bool(pb_filters[3]) if len(pb_filters) > 3 else False,
                    },
                    "buried_fraction": float(pb_filters[4]) if len(pb_filters) > 4 else 0.0,
                })
            break

    emit(ProgressEvent(
        type="poses_update",
        poses=poses,
        best_error=poses[0]["error_estimate"] if poses else None,
        best_pb=poses[0]["pb_count"] if poses else None,
        current_ligand=ligand.name,
        ligand_index=ligand_index,
        total_ligands=total_ligands,
    ))

    return {
        "poses": poses,
        "output_path": str(run_output_dir),
    }
