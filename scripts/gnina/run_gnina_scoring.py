#!/usr/bin/env python3
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tqdm import tqdm

from matcha.utils.gnina_scoring_batch import (
    build_gnina_command,
    count_molecules_in_sdf,
    resolve_gnina_receptor_path,
    run_gnina_subprocess,
)
from matcha.utils.log import get_logger

logger = get_logger(__name__)


def score_one_dataset(
    *,
    exp_name: str,
    dataset_name: str,
    minimize: bool,
    gnina_script: Path,
    inference_folder: Path,
    receptors_folder: Path,
    device: str,
    workers: int,
) -> None:
    """Run GNINA for every ``*.sdf`` under ``{run}/{dataset}/sdf_predictions``."""
    results_folder = inference_folder / exp_name
    ligands_dir = results_folder / dataset_name / "sdf_predictions"

    logger.info("ligands_dir: %s", ligands_dir)

    if minimize:
        out_ligands_dir = results_folder / dataset_name / "minimized_sdf_predictions"
    else:
        out_ligands_dir = results_folder / dataset_name / "base_sdf_predictions"

    logger.info("out_ligands_dir: %s", out_ligands_dir)
    out_ligands_dir.mkdir(parents=True, exist_ok=True)

    sdf_files = sorted(ligands_dir.glob("*.sdf"))
    total_files = len(sdf_files)
    logger.info("[%s] Found %d SDF files under %s", dataset_name, total_files, ligands_dir)

    if total_files == 0:
        logger.warning("[%s] No SDF inputs; skipping", dataset_name)
        return

    def score_ligand(ligand_path: Path) -> tuple[str, int]:
        ligand_filename = ligand_path.name
        uid = ligand_path.stem
        num_molecules = count_molecules_in_sdf(ligand_path)

        receptor_path = resolve_gnina_receptor_path(
            dataset_name=dataset_name,
            uid=uid,
            receptors_folder=receptors_folder,
        )

        if not receptor_path.is_file():
            logger.warning("[%s] Receptor not found: %s", dataset_name, receptor_path)
            return uid, num_molecules

        output_sdf = out_ligands_dir / ligand_filename
        cmd = build_gnina_command(
            gnina_script,
            receptor_path,
            ligand_path,
            output_sdf,
            minimize=minimize,
            device=device,
        )
        completed = run_gnina_subprocess(cmd)
        if completed.returncode != 0:
            logger.warning(
                "[%s] GNINA exited with code %s for %s (cmd[0]=%s)",
                dataset_name,
                completed.returncode,
                ligand_filename,
                cmd[0],
            )
        return uid, num_molecules

    logger.info("[%s] Running up to %d GNINA processes concurrently", dataset_name, workers)
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        tqdm(total=total_files, desc=f"GNINA {dataset_name}", unit="file") as pbar,
    ):
        futures = [executor.submit(score_ligand, ligand_path) for ligand_path in sdf_files]
        for future in as_completed(futures):
            uid, num_molecules = future.result()
            mol_word = "mol" if num_molecules == 1 else "mols"
            pbar.set_postfix_str(f"{uid} · {num_molecules} {mol_word}", refresh=False)
            pbar.update()

    logger.info(
        "[%s] Done. Touched %d/%d ligand files under %s",
        dataset_name,
        total_files,
        total_files,
        out_ligands_dir,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GNINA scoring/minimization over sdf_predictions. "
            "By default uses test_dataset_types from config file; "
            "use --dataset to restrict to one benchmark."
        )
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="exp_name",
        required=True,
        help="Inference run name (folder under inference_results_folder)",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_name",
        default=None,
        help="Process only this dataset subdirectory (default: all test_dataset_types)",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        type=Path,
        required=True,
        help="Experiment YAML",
    )
    parser.add_argument(
        "-p",
        "--paths-config",
        dest="paths_config_file",
        type=Path,
        required=True,
        help="Paths YAML",
    )
    parser.add_argument(
        "--gnina-script",
        dest="gnina_script",
        type=Path,
        required=True,
        help="GNINA wrapper or binary",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Minimize ligands and write minimized_sdf_predictions (else base_sdf_predictions)",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default="0",
        help="Logical CUDA device passed to GNINA (default: 0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum concurrent GNINA processes (default: 4)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    minimize = args.minimize
    gnina_script = args.gnina_script

    if args.workers < 1:
        logger.error("--workers must be at least 1")
        return 1
    conf: Any = OmegaConf.merge(
        OmegaConf.load(args.config_file),
        OmegaConf.load(args.paths_config_file),
    )
    inference_folder = Path(str(conf.inference_results_folder))
    receptors_folder = Path(str(conf.preprocessed_receptors_base))

    if not gnina_script.exists():
        logger.error("GNINA script not found: %s", gnina_script)
        return 1

    if args.dataset_name is not None:
        datasets = [args.dataset_name]
    else:
        datasets = conf.test_dataset_types

    logger.info("GNINA datasets to process (%d): %s", len(datasets), datasets)

    for dataset_name in datasets:
        score_one_dataset(
            exp_name=args.exp_name,
            dataset_name=dataset_name,
            minimize=minimize,
            gnina_script=gnina_script,
            inference_folder=inference_folder,
            receptors_folder=receptors_folder,
            device=args.device,
            workers=args.workers,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
