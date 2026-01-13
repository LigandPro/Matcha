"""
Docking worker - runs the docking pipeline with progress reporting.
"""

import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from matcha.tui.protocol import ProgressEvent, PipelineStage

if TYPE_CHECKING:
    from matcha.tui.backend import DockingJob


def run_docking(job: "DockingJob") -> None:
    """Run docking job with progress reporting."""
    config = job.config
    emit = job.progress_callback

    try:
        emit(ProgressEvent(type="stage_start", stage="init", name="Initializing"))

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

        if job.is_cancelled():
            emit(ProgressEvent(type="cancelled", message="Job cancelled"))
            return

        # Set GPU
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # Import matcha modules
        emit(ProgressEvent(type="stage_progress", stage="init", progress=20))

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

        dataset_dir = run_workdir / "datasets" / "any_conf"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        emit(ProgressEvent(type="stage_done", stage="init", elapsed=0))

        if job.is_cancelled():
            emit(ProgressEvent(type="cancelled", message="Job cancelled"))
            return

        # Checkpoints
        emit(ProgressEvent(type="stage_start", stage="checkpoints", name="Loading checkpoints"))
        start_time = time.time()

        if checkpoints:
            checkpoint_path = Path(checkpoints).expanduser().resolve()
        else:
            from huggingface_hub import snapshot_download

            emit(
                ProgressEvent(
                    type="stage_progress",
                    stage="checkpoints",
                    progress=10,
                    message="Downloading from HuggingFace...",
                )
            )
            checkpoint_path = Path(
                snapshot_download("LigandPro/Matcha", allow_patterns="pipeline/*")
            )

        emit(
            ProgressEvent(
                type="stage_done", stage="checkpoints", elapsed=time.time() - start_time
            )
        )

        if job.is_cancelled():
            emit(ProgressEvent(type="cancelled", message="Job cancelled"))
            return

        # Prepare dataset
        emit(ProgressEvent(type="stage_start", stage="dataset", name="Preparing dataset"))
        start_time = time.time()

        receptor = Path(receptor_path).expanduser().resolve()
        uid = "sample_0"
        sample_dir = dataset_dir / uid
        sample_dir.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(receptor, sample_dir / f"{uid}_protein.pdb")

        # Handle ligand
        from rdkit import Chem

        if ligand_path:
            ligand = Path(ligand_path).expanduser().resolve()
            # Normalize to SDF
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

        emit(ProgressEvent(type="stage_done", stage="dataset", elapsed=time.time() - start_time))

        if job.is_cancelled():
            emit(ProgressEvent(type="cancelled", message="Job cancelled"))
            return

        # Build config
        conf = OmegaConf.create(
            {
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
                "inference_results_folder": str(run_workdir / "runs"),
                "any_data_dir": str(run_workdir / "datasets"),
            }
        )

        # ESM embeddings
        emit(ProgressEvent(type="stage_start", stage="esm", name="Computing ESM embeddings"))
        start_time = time.time()

        data_dir = run_workdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        compute_sequences(conf, "any_conf", data_dir)
        emit(ProgressEvent(type="stage_progress", stage="esm", progress=30))

        if job.is_cancelled():
            emit(ProgressEvent(type="cancelled", message="Job cancelled"))
            return

        compute_esm_embeddings(conf, "any_conf", data_dir)
        emit(ProgressEvent(type="stage_done", stage="esm", elapsed=time.time() - start_time))

        if job.is_cancelled():
            emit(ProgressEvent(type="cancelled", message="Job cancelled"))
            return

        # Pocket centers (if specified)
        pocket_centers = None
        if box_mode == "manual" and all(v is not None for v in [center_x, center_y, center_z]):
            import numpy as np

            pocket_centers = {uid: np.array([center_x, center_y, center_z])}
        elif box_mode == "autobox" and autobox_ligand:
            import numpy as np

            autobox_path = Path(autobox_ligand).expanduser().resolve()
            mol = Chem.MolFromMolFile(str(autobox_path), removeHs=False, sanitize=False)
            if mol is not None:
                conformer = mol.GetConformer()
                coords = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
                center = coords.mean(axis=0)
                pocket_centers = {uid: center}

        # Run inference stages
        stages = ["stage1", "stage2", "stage3"]
        for stage_name in stages:
            emit(
                ProgressEvent(
                    type="stage_start",
                    stage=stage_name,
                    name=f"Stage {stage_name[-1]} inference",
                )
            )
            start_time = time.time()

            if job.is_cancelled():
                emit(ProgressEvent(type="cancelled", message="Job cancelled"))
                return

            # Run pipeline stage
            run_inference_pipeline(
                conf,
                "any_conf",
                data_dir,
                stage_name,
                n_samples=n_samples,
                n_confs=n_confs,
                pocket_centers=pocket_centers,
            )

            emit(
                ProgressEvent(
                    type="stage_done", stage=stage_name, elapsed=time.time() - start_time
                )
            )

        if job.is_cancelled():
            emit(ProgressEvent(type="cancelled", message="Job cancelled"))
            return

        # Scoring
        emit(ProgressEvent(type="stage_start", stage="scoring", name="Scoring poses"))
        start_time = time.time()

        run_inference_pipeline(
            conf,
            "any_conf",
            data_dir,
            "scoring",
            n_samples=n_samples,
            n_confs=n_confs,
            pocket_centers=pocket_centers,
        )

        emit(ProgressEvent(type="stage_done", stage="scoring", elapsed=time.time() - start_time))

        if job.is_cancelled():
            emit(ProgressEvent(type="cancelled", message="Job cancelled"))
            return

        # PoseBusters
        emit(ProgressEvent(type="stage_start", stage="posebusters", name="Physical validation"))
        start_time = time.time()

        compute_fast_filters(conf, "any_conf", conf.inference_results_folder)

        emit(
            ProgressEvent(
                type="stage_done", stage="posebusters", elapsed=time.time() - start_time
            )
        )

        # Save results
        emit(ProgressEvent(type="stage_start", stage="done", name="Saving results"))

        output_run_dir = output_path / run_name
        output_run_dir.mkdir(parents=True, exist_ok=True)

        save_best_pred_to_sdf(
            conf,
            "any_conf",
            conf.inference_results_folder,
            output_run_dir,
            physical_only=physical_only,
        )

        # Get best pose info
        import numpy as np

        metrics_file = list(Path(conf.inference_results_folder).glob("**/*_final_preds_fast_metrics.npy"))
        poses = []
        if metrics_file:
            data = np.load(metrics_file[0], allow_pickle=True).item()
            for uid_key, samples in data.items():
                for i, sample in enumerate(samples[:10]):  # Top 10
                    pb_filters = sample.get("posebusters_filters_fast", [False] * 5)
                    poses.append(
                        {
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
                        }
                    )
                break

        emit(
            ProgressEvent(
                type="poses_update",
                poses=poses,
                best_error=poses[0]["error_estimate"] if poses else None,
                best_pb=poses[0]["pb_count"] if poses else None,
            )
        )

        emit(
            ProgressEvent(
                type="job_done",
                stage="done",
                output_path=str(output_run_dir),
                poses=poses,
            )
        )

    except Exception as e:
        emit(ProgressEvent(type="error", message=str(e)))
        raise
