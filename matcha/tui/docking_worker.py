"""
Docking worker - runs the docking pipeline with progress reporting.
Supports both single-ligand and batch processing modes.
"""

import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from matcha.tui.protocol import ProgressEvent, LigandStatus

if TYPE_CHECKING:
    from matcha.tui.backend import DockingJob


def find_ligands(ligand_dir: Path, extensions: tuple = (".sdf", ".mol", ".mol2")) -> list[Path]:
    """Find all ligand files in a directory."""
    ligands = []
    for ext in extensions:
        ligands.extend(ligand_dir.glob(f"*{ext}"))
    return sorted(ligands, key=lambda p: p.name.lower())


def run_docking(job: "DockingJob") -> None:
    """Run docking job with progress reporting."""
    config = job.config
    emit = job.progress_callback

    try:
        # Extract config
        receptor_path = config.get("receptor")
        ligand_path = config.get("ligand")
        ligand_dir = config.get("ligand_dir")
        output_dir = config.get("output_dir", "./results")
        run_name = config.get("run_name", "matcha_tui_run")
        n_samples = config.get("n_samples", 40)
        n_confs = config.get("n_confs")
        gpu = config.get("gpu")
        checkpoints = config.get("checkpoints")
        physical_only = config.get("physical_only", False)

        # Box settings
        box_mode = config.get("box_mode", "blind")
        center_x = config.get("center_x")
        center_y = config.get("center_y")
        center_z = config.get("center_z")
        autobox_ligand = config.get("autobox_ligand")

        # Determine ligand list
        ligands: list[Path] = []
        if ligand_dir:
            ligand_dir_path = Path(ligand_dir).expanduser().resolve()
            if ligand_dir_path.is_dir():
                ligands = find_ligands(ligand_dir_path)
        elif ligand_path:
            ligands = [Path(ligand_path).expanduser().resolve()]

        if not ligands:
            emit(ProgressEvent(type="error", message="No ligand files found"))
            return

        is_batch = len(ligands) > 1
        total_ligands = len(ligands)

        # Initialize ligand statuses for batch mode
        ligand_statuses = [
            LigandStatus(name=lig.name, path=str(lig), status="pending")
            for lig in ligands
        ]

        if is_batch:
            emit(ProgressEvent(
                type="batch_start",
                total_ligands=total_ligands,
                message=f"Processing {total_ligands} ligands",
                ligand_statuses=[ls.to_dict() for ls in ligand_statuses],
            ))

        if job.is_cancelled():
            emit(ProgressEvent(type="cancelled", message="Job cancelled"))
            return

        # Set GPU
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # Import matcha modules (once for all ligands)
        emit(ProgressEvent(type="stage_start", stage="init", name="Initializing"))

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

            if is_batch:
                completed = sum(1 for ls in ligand_statuses if ls.status in ("completed", "failed"))
                emit(ProgressEvent(
                    type="batch_progress",
                    progress=int(100 * completed / total_ligands),
                    ligand_index=idx + 1,
                    total_ligands=total_ligands,
                    ligand_statuses=[ls.to_dict() for ls in ligand_statuses],
                ))

        # Save batch summary
        output_run_dir = output_path / run_name
        output_run_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate best poses
        all_poses = []
        for result in all_results:
            if "poses" in result:
                for pose in result["poses"]:
                    pose["ligand"] = result["ligand"]
                all_poses.extend(result["poses"])

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

    # Run inference pipeline (all stages at once)
    emit(ProgressEvent(
        type="stage_start",
        stage="stage1",
        name="Running inference pipeline",
        current_ligand=ligand.name,
        ligand_index=ligand_index,
        total_ligands=total_ligands,
    ))
    start_time = time.time()

    if job.is_cancelled():
        emit(ProgressEvent(type="cancelled", message="Job cancelled"))
        return None

    run_inference_pipeline(
        copy.deepcopy(conf),
        "any_conf",
        n_samples,
        pocket_centers_filename,
    )

    emit(ProgressEvent(
        type="stage_done",
        stage="scoring",
        elapsed=time.time() - start_time,
        current_ligand=ligand.name,
        ligand_index=ligand_index,
        total_ligands=total_ligands,
    ))

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
    sdf_preds_dir = Path(conf.inference_results_folder) / "any_conf" / "sdf_predictions"
    pred_sdf_files = list(sdf_preds_dir.glob("*.sdf"))
    if pred_sdf_files:
        best_dest = run_output_dir / f"{ligand_name}_best.sdf"
        all_dest = run_output_dir / f"{ligand_name}_poses.sdf"
        shutil.copyfile(pred_sdf_files[0], best_dest)
        shutil.copyfile(pred_sdf_files[0], all_dest)

    # Get pose info
    import numpy as np

    metrics_file = list(Path(conf.inference_results_folder).glob("**/*_final_preds_fast_metrics.npy"))
    poses = []
    if metrics_file:
        data = np.load(metrics_file[0], allow_pickle=True).item()
        for uid_key, samples in data.items():
            for i, sample in enumerate(samples[:10]):
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
