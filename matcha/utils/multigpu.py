from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ShardSpec:
    gpu_id: int
    shard_dir: Path
    out_dir: Path
    run_name: str
    run_dir: Path
    command: list[str]
    env: dict[str, str]
    launcher_log_path: Path
    ligand_count: int


def parse_gpus(raw: str) -> list[int]:
    values = [chunk.strip() for chunk in raw.split(",")]
    if not values or any(not chunk for chunk in values):
        raise ValueError("Invalid --gpus value. Use comma-separated GPU ids, e.g. '2,3'.")

    gpu_ids: list[int] = []
    seen: set[int] = set()
    for chunk in values:
        try:
            gpu = int(chunk)
        except ValueError as exc:
            raise ValueError(f"Invalid GPU id in --gpus: {chunk}") from exc
        if gpu < 0:
            raise ValueError(f"GPU id must be >= 0, got: {gpu}")
        if gpu in seen:
            raise ValueError(f"Duplicate GPU id in --gpus: {gpu}")
        seen.add(gpu)
        gpu_ids.append(gpu)
    return gpu_ids


def find_ligand_files(ligand_dir: Path, recursive: bool) -> list[Path]:
    if not ligand_dir.exists():
        raise ValueError(f"Ligand directory does not exist: {ligand_dir}")
    if not ligand_dir.is_dir():
        raise ValueError("Multi-GPU mode requires --ligand-dir to be a directory.")

    if recursive:
        files = list(ligand_dir.rglob("*.sdf")) + list(ligand_dir.rglob("*.mol2"))
    else:
        files = list(ligand_dir.glob("*.sdf")) + list(ligand_dir.glob("*.mol2"))
    files = sorted(files, key=lambda p: str(p).lower())
    if not files:
        raise ValueError(f"No .sdf/.mol2 files found in {ligand_dir}")
    return files


def shard_ligand_files(files: list[Path], n_shards: int) -> list[list[Path]]:
    if n_shards <= 0:
        raise ValueError(f"n_shards must be > 0, got {n_shards}")
    shards: list[list[Path]] = [[] for _ in range(n_shards)]
    for idx, file_path in enumerate(sorted(files, key=lambda p: str(p).lower())):
        shards[idx % n_shards].append(file_path)
    return shards


def _link_or_copy_overwrite(src: Path, dst: Path) -> None:
    """Symlink src to dst, overwriting if dst already exists. Falls back to copy."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(str(src.resolve()), str(dst))
    except OSError:
        shutil.copy2(src, dst)


def materialize_shards(sharded_files: list[list[Path]], target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs: list[Path] = []
    for shard_idx, shard_files in enumerate(sharded_files):
        shard_dir = target_dir / f"shard_{shard_idx:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for file_idx, src in enumerate(shard_files):
            dst = shard_dir / f"{file_idx:05d}__{src.name}"
            _link_or_copy_overwrite(src, dst)
        shard_dirs.append(shard_dir)
    return shard_dirs


def _append_optional_flag(cmd: list[str], value: bool | None, enabled: str, disabled: str) -> None:
    if value is None:
        return
    cmd.append(enabled if value else disabled)


def build_shard_command(
    *,
    receptor: Path,
    shard_ligand_dir: Path,
    out_dir: Path,
    run_name: str,
    checkpoints: Optional[Path],
    config: Optional[Path],
    center_x: Optional[float],
    center_y: Optional[float],
    center_z: Optional[float],
    autobox_ligand: Optional[Path],
    box_json: Optional[Path],
    n_samples: int,
    n_confs: Optional[int],
    docking_batch_limit: int,
    recursive: bool,
    num_workers: Optional[int],
    pin_memory: Optional[bool],
    prefetch_factor: int,
    persistent_workers: Optional[bool],
    scorer_type: str,
    scorer_path: Optional[Path],
    scorer_minimize: bool,
    gnina_batch_mode: str,
    physical_only: bool,
) -> list[str]:
    cmd: list[str] = [
        "uv",
        "run",
        "matcha",
        "-r",
        str(receptor),
        "--ligand-dir",
        str(shard_ligand_dir),
        "-o",
        str(out_dir),
        "--run-name",
        run_name,
        "--device",
        "cuda:0",
        "--n-samples",
        str(n_samples),
        "--docking-batch-limit",
        str(docking_batch_limit),
        "--prefetch-factor",
        str(prefetch_factor),
        "--scorer",
        scorer_type,
        "--gnina-batch-mode",
        gnina_batch_mode,
        "--overwrite",
        "--keep-workdir",
    ]
    cmd.append("--recursive" if recursive else "--no-recursive")

    # Box specification (mutually exclusive)
    if box_json is not None:
        cmd.extend(["--box-json", str(box_json)])
    elif autobox_ligand is not None:
        cmd.extend(["--autobox-ligand", str(autobox_ligand)])
    elif center_x is not None and center_y is not None and center_z is not None:
        cmd.extend(["--center-x", str(center_x), "--center-y", str(center_y), "--center-z", str(center_z)])

    # Optional path arguments
    for flag, value in [("--checkpoints", checkpoints), ("--config", config),
                        ("--n-confs", n_confs), ("--num-workers", num_workers),
                        ("--scorer-path", scorer_path)]:
        if value is not None:
            cmd.extend([flag, str(value)])

    _append_optional_flag(cmd, pin_memory, "--pin-memory", "--no-pin-memory")
    _append_optional_flag(cmd, persistent_workers, "--persistent-workers", "--no-persistent-workers")
    cmd.append("--scorer-minimize" if scorer_minimize else "--no-scorer-minimize")
    cmd.append("--physical-only" if physical_only else "--keep-all-poses")
    return cmd


def _make_shard_specs(
    *,
    gpu_ids: list[int],
    shard_dirs: list[Path],
    shard_counts: list[int],
    run_workdir: Path,
    run_name: str,
    receptor: Path,
    checkpoints: Optional[Path],
    config: Optional[Path],
    center_x: Optional[float],
    center_y: Optional[float],
    center_z: Optional[float],
    autobox_ligand: Optional[Path],
    box_json: Optional[Path],
    n_samples: int,
    n_confs: Optional[int],
    docking_batch_limit: int,
    recursive: bool,
    num_workers: Optional[int],
    pin_memory: Optional[bool],
    prefetch_factor: int,
    persistent_workers: Optional[bool],
    scorer_type: str,
    scorer_path: Optional[Path],
    scorer_minimize: bool,
    gnina_batch_mode: str,
    physical_only: bool,
) -> list[ShardSpec]:
    specs: list[ShardSpec] = []
    for gpu_id, shard_dir, ligand_count in zip(gpu_ids, shard_dirs, shard_counts):
        if ligand_count <= 0:
            continue
        shard_run_name = f"{run_name}__gpu{gpu_id}"
        shard_out_dir = run_workdir / "shards" / f"gpu{gpu_id}"
        shard_run_dir = shard_out_dir / shard_run_name
        launcher_log = shard_out_dir / f"{shard_run_name}.launcher.log"
        cmd = build_shard_command(
            receptor=receptor,
            shard_ligand_dir=shard_dir,
            out_dir=shard_out_dir,
            run_name=shard_run_name,
            checkpoints=checkpoints,
            config=config,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            autobox_ligand=autobox_ligand,
            box_json=box_json,
            n_samples=n_samples,
            n_confs=n_confs,
            docking_batch_limit=docking_batch_limit,
            recursive=recursive,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            scorer_type=scorer_type,
            scorer_path=scorer_path,
            scorer_minimize=scorer_minimize,
            gnina_batch_mode=gnina_batch_mode,
            physical_only=physical_only,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        specs.append(
            ShardSpec(
                gpu_id=gpu_id,
                shard_dir=shard_dir,
                out_dir=shard_out_dir,
                run_name=shard_run_name,
                run_dir=shard_run_dir,
                command=cmd,
                env=env,
                launcher_log_path=launcher_log,
                ligand_count=ligand_count,
            )
        )
    if not specs:
        raise RuntimeError("No non-empty shards were created.")
    return specs


def launch_shards(specs: list[ShardSpec], *, cwd: Path) -> list[dict]:
    processes: list[tuple[ShardSpec, subprocess.Popen, object, float]] = []
    run_records: list[dict] = []
    failure: Optional[tuple[ShardSpec, int]] = None

    for spec in specs:
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        log_handle = open(spec.launcher_log_path, "w", encoding="utf-8")
        log_handle.write(f"Command: {shlex.join(spec.command)}\n")
        log_handle.write(f"CUDA_VISIBLE_DEVICES={spec.env.get('CUDA_VISIBLE_DEVICES')}\n")
        log_handle.flush()
        proc = subprocess.Popen(
            spec.command,
            cwd=str(cwd),
            env=spec.env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((spec, proc, log_handle, time.time()))

    running = set(range(len(processes)))
    while running:
        for idx in list(running):
            spec, proc, log_handle, started_at = processes[idx]
            rc = proc.poll()
            if rc is None:
                continue
            elapsed = time.time() - started_at
            log_handle.flush()
            log_handle.close()
            run_records.append(
                {
                    "gpu_id": spec.gpu_id,
                    "run_name": spec.run_name,
                    "run_dir": str(spec.run_dir),
                    "ligand_count": spec.ligand_count,
                    "return_code": int(rc),
                    "elapsed_sec": float(elapsed),
                    "launcher_log_path": str(spec.launcher_log_path),
                }
            )
            running.discard(idx)
            if rc != 0 and failure is None:
                failure = (spec, int(rc))
                for other_idx in list(running):
                    processes[other_idx][1].terminate()
        if running:
            time.sleep(0.5)

    if failure is not None:
        failed_spec, failed_rc = failure
        raise RuntimeError(
            "Shard process failed: "
            f"gpu={failed_spec.gpu_id}, rc={failed_rc}, log={failed_spec.launcher_log_path}"
        )
    return run_records


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise RuntimeError(f"Merge collision: destination already exists: {dst}")
    try:
        os.symlink(str(src.resolve()), str(dst))
    except OSError:
        shutil.copy2(src, dst)


def _load_keyed_data(path: Path) -> dict:
    """Load a keyed dict from either .npy or .json file."""
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True).item()  # noqa: S301
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_keyed_data(data: dict, output_path: Path) -> None:
    """Save a keyed dict to either .npy or .json based on extension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".npy":
        np.save(output_path, [data])
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def _merge_keyed_files(paths: list[Path], output_path: Path, label: str) -> dict:
    """Merge multiple keyed dict files, raising on duplicate keys."""
    merged: dict = {}
    for path in paths:
        if not path.exists():
            continue
        loaded = _load_keyed_data(path)
        for uid, value in loaded.items():
            if uid in merged:
                raise RuntimeError(f"Duplicate uid in merged {label}: {uid}")
            merged[uid] = value
    _save_keyed_data(merged, output_path)
    return merged


def merge_shard_outputs(
    *,
    shard_records: list[dict],
    merged_root: Path,
    expected_ligands: int,
) -> dict:
    merged_root.mkdir(parents=True, exist_ok=True)
    for folder in ("best_poses", "all_poses", "best_scored_predictions", "scored_sdf_predictions"):
        (merged_root / folder).mkdir(parents=True, exist_ok=True)

    pred_paths: list[Path] = []
    filter_paths: list[Path] = []
    merged_uids_from_best: set[str] = set()

    for record in shard_records:
        run_dir = Path(record["run_dir"])
        run_name = str(record["run_name"])

        for folder in ("best_poses", "all_poses"):
            src_dir = run_dir / folder
            if not src_dir.exists():
                continue
            for src_file in sorted(src_dir.glob("*.sdf")):
                uid = src_file.stem
                if folder == "all_poses" and uid.endswith("_poses"):
                    uid = uid[:-6]
                if folder == "best_poses":
                    if uid in merged_uids_from_best:
                        raise RuntimeError(f"Duplicate uid in best_poses merge: {uid}")
                    merged_uids_from_best.add(uid)
                _link_or_copy(src_file, merged_root / folder / src_file.name)

        shard_run_inner = run_dir / "work" / "runs" / run_name / "any_conf"
        for folder in ("best_scored_predictions", "scored_sdf_predictions"):
            src_dir = shard_run_inner / folder
            if not src_dir.exists():
                continue
            for src_file in sorted(src_dir.glob("*.sdf")):
                _link_or_copy(src_file, merged_root / folder / src_file.name)

        pred_paths.append(run_dir / "work" / "runs" / run_name / "any_conf_final_preds.npy")
        filter_paths.append(run_dir / "work" / "runs" / run_name / "any_conf" / "filters_results.json")

    merged_preds = _merge_keyed_files(pred_paths, merged_root / "any_conf_final_preds.npy", "predictions")
    merged_filters = _merge_keyed_files(filter_paths, merged_root / "filters_results.json", "filters")

    if expected_ligands > 0 and len(merged_preds) != expected_ligands:
        raise RuntimeError(
            "Merged prediction count mismatch: "
            f"expected={expected_ligands}, got={len(merged_preds)}"
        )

    return {
        "merged_root": str(merged_root),
        "merged_prediction_uids": len(merged_preds),
        "merged_filter_uids": len(merged_filters),
        "merged_best_poses_uids": len(merged_uids_from_best),
    }


def _safe_read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_benchmark_report(
    *,
    run_workdir: Path,
    total_ligands: int,
    shard_records: list[dict],
) -> dict:
    stage_keys = ("stage1_sec", "stage2_sec", "stage3_sec", "sdf_save_sec", "posebusters_sec")
    pipeline_stage_sums = dict.fromkeys(stage_keys, 0.0)
    shard_reports = []
    total_sec_values = []

    for record in shard_records:
        run_dir = Path(record["run_dir"])
        run_name = str(record["run_name"])
        timing = _safe_read_json(run_dir / "run_timing.json") or {}
        pipeline_timing = _safe_read_json(run_dir / "work" / "runs" / run_name / "timings_pipeline.json") or {}
        total_sec = float(timing.get("total_sec", record.get("elapsed_sec", 0.0)))
        total_sec_values.append(total_sec)

        for key in stage_keys:
            pipeline_stage_sums[key] += float(pipeline_timing.get(key, 0.0))

        shard_reports.append(
            {
                **record,
                "run_timing_path": str(run_dir / "run_timing.json"),
                "pipeline_timing_path": str(run_dir / "work" / "runs" / run_name / "timings_pipeline.json"),
                "total_sec": total_sec,
                "timing": timing,
                "pipeline_timing": pipeline_timing,
            }
        )

    critical_path_sec = max(total_sec_values) if total_sec_values else 0.0
    estimated_single_sec = sum(total_sec_values)
    complexes_per_hour = (total_ligands * 3600.0 / critical_path_sec) if critical_path_sec > 0 else 0.0
    speedup_estimated = (estimated_single_sec / critical_path_sec) if critical_path_sec > 0 else 0.0

    report = {
        "total_ligands": int(total_ligands),
        "num_shards": len(shard_reports),
        "critical_path_sec": critical_path_sec,
        "estimated_single_sec": estimated_single_sec,
        "speedup_vs_single_estimated": speedup_estimated,
        "complexes_per_hour": complexes_per_hour,
        "pipeline_stage_sums": pipeline_stage_sums,
        "shards": shard_reports,
    }

    summary_json = run_workdir / "benchmark_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary_md = run_workdir / "benchmark_summary.md"
    lines = [
        "# Multi-GPU Benchmark Summary",
        "",
        f"- Total ligands: {total_ligands}",
        f"- Shards: {len(shard_reports)}",
        f"- Critical path (sec): {critical_path_sec:.2f}",
        f"- Estimated single-GPU time (sec): {estimated_single_sec:.2f}",
        f"- Estimated speedup vs single: {speedup_estimated:.3f}x",
        f"- Throughput: {complexes_per_hour:.2f} complexes/hour",
        "",
        "## Stage Sums (seconds)",
        "",
        f"- stage1: {pipeline_stage_sums['stage1_sec']:.2f}",
        f"- stage2: {pipeline_stage_sums['stage2_sec']:.2f}",
        f"- stage3: {pipeline_stage_sums['stage3_sec']:.2f}",
        f"- sdf_save: {pipeline_stage_sums['sdf_save_sec']:.2f}",
        f"- posebusters: {pipeline_stage_sums['posebusters_sec']:.2f}",
    ]
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return report


def run_multigpu_batch(
    *,
    gpu_ids: list[int],
    run_workdir: Path,
    run_name: str,
    receptor: Path,
    ligand_dir: Path,
    recursive: bool,
    checkpoints: Optional[Path],
    config: Optional[Path],
    center_x: Optional[float],
    center_y: Optional[float],
    center_z: Optional[float],
    autobox_ligand: Optional[Path],
    box_json: Optional[Path],
    n_samples: int,
    n_confs: Optional[int],
    docking_batch_limit: int,
    num_workers: Optional[int],
    pin_memory: Optional[bool],
    prefetch_factor: int,
    persistent_workers: Optional[bool],
    scorer_type: str,
    scorer_path: Optional[Path],
    scorer_minimize: bool,
    gnina_batch_mode: str,
    physical_only: bool,
) -> dict:
    ligand_files = find_ligand_files(ligand_dir, recursive=recursive)
    sharded = shard_ligand_files(ligand_files, n_shards=len(gpu_ids))
    shard_counts = [len(chunk) for chunk in sharded]
    shard_dirs = materialize_shards(sharded, run_workdir / "shard_inputs")

    specs = _make_shard_specs(
        gpu_ids=gpu_ids,
        shard_dirs=shard_dirs,
        shard_counts=shard_counts,
        run_workdir=run_workdir,
        run_name=run_name,
        receptor=receptor,
        checkpoints=checkpoints,
        config=config,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        autobox_ligand=autobox_ligand,
        box_json=box_json,
        n_samples=n_samples,
        n_confs=n_confs,
        docking_batch_limit=docking_batch_limit,
        recursive=recursive,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        scorer_type=scorer_type,
        scorer_path=scorer_path,
        scorer_minimize=scorer_minimize,
        gnina_batch_mode=gnina_batch_mode,
        physical_only=physical_only,
    )

    repo_root = Path(__file__).resolve().parents[2]
    shard_records = launch_shards(specs, cwd=repo_root)

    merged = merge_shard_outputs(
        shard_records=shard_records,
        merged_root=run_workdir / "merged",
        expected_ligands=len(ligand_files),
    )
    benchmark = write_benchmark_report(
        run_workdir=run_workdir,
        total_ligands=len(ligand_files),
        shard_records=shard_records,
    )
    return {
        "ligand_count": len(ligand_files),
        "gpu_ids": gpu_ids,
        "shard_counts": shard_counts,
        "merged": merged,
        "benchmark": benchmark,
    }
