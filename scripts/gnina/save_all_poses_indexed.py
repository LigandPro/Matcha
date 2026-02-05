#!/usr/bin/env python3
"""
Script to save all poses from SDF predictions with indexed filenames.

This script reads SDF files containing multiple poses (all poses in one file),
and saves each i-th pose into a separate file named '{uid}_{i}_minimized.sdf'.

Expected input structure:
- {inference_results_folder}/{run_name}/{dataset_name}/minimized_sdf_predictions/
  - {uid}.sdf
  - ...

Output structure:
- {inference_results_folder}/{run_name}/{dataset_name}/indexed_minimized_predictions/
  - {uid}_0_minimized.sdf
  - {uid}_1_minimized.sdf
  - {uid}_2_minimized.sdf
  - ...

Usage:
    python save_all_poses_indexed.py \
        -p configs/paths/paths.yaml \
        -n my_experiment
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import os
from omegaconf import OmegaConf

# Suppress RDKit warnings about 2D/3D molecule tags (common after gnina minimization)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

from rdkit import Chem
from rdkit.Chem import SDWriter

from matcha.utils.log import get_logger

logger = get_logger(__name__)


def extract_uid_from_filename(filename: str) -> str:
    """
    Extract UID from filename. Assumes format like '{uid}_minimized.sdf' or '{uid}.sdf'.
    
    Args:
        filename: Input filename
    
    Returns:
        Extracted UID
    """
    # Remove extension
    base = Path(filename).stem
    
    # Remove common suffixes
    for suffix in ['_minimized', '_poses', '_predictions']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    
    return base


def save_all_poses_indexed(
    sdf_path: Path,
    output_folder: Path,
    uid: str,
) -> int:
    """
    Read SDF file and save each pose with an indexed filename.
    
    Args:
        sdf_path: Path to SDF file containing multiple poses
        output_folder: Output folder to save indexed poses
        uid: UID for this molecule
    
    Returns:
        Number of poses saved
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    
    poses_saved = 0
    
    for i, mol in enumerate(supplier):
        if mol is None:
            logger.warning(f"Skipping invalid molecule at index {i} in {sdf_path.name}")
            continue
        
        # Save this pose with indexed filename
        os.makedirs(output_folder / f"{i}_minimized", exist_ok=True)
        output_sdf = output_folder / f"{i}_minimized" / f"{uid}.sdf"
        writer = SDWriter(str(output_sdf))
        writer.write(mol)
        writer.close()
        
        poses_saved += 1
    
    return poses_saved


def process_sdf_folder(
    inference_results_folder: Path,
    inference_run_name: str,
    dataset_name: str,
):
    """
    Process all SDF files in input folder and save all poses with indexed filenames.
    
    Args:
        inference_results_folder: Base inference results folder from config
        inference_run_name: Name of the inference run
        dataset_name: Name of the dataset (used in folder structure, e.g., 'pdbbind', 'astex')
    """
    inference_results_folder = Path(inference_results_folder)
    exp_folder = inference_results_folder / inference_run_name
    
    # Input from minimized predictions
    input_folder = exp_folder / dataset_name / "minimized_sdf_predictions"
    output_folder = exp_folder / dataset_name / "indexed_minimized_predictions"
    
    # Create output directory structure
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all SDF files
    sdf_pattern = '*.sdf'
    sdf_files = list(input_folder.glob(sdf_pattern))
    
    if len(sdf_files) == 0:
        logger.warning(f"No SDF files found in {input_folder} with pattern {sdf_pattern}")
        return
    
    logger.info(f"Found {len(sdf_files)} SDF files to process")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    
    successful = 0
    failed = 0
    total_poses = 0
    for sdf_file in tqdm(sdf_files, desc="Processing SDF files"):
        try:
            # Extract UID from filename
            uid = extract_uid_from_filename(sdf_file.name)
            
            # Save all poses with indexed filenames
            num_poses = save_all_poses_indexed(
                sdf_file,
                output_folder,
                uid
            )
            
            if num_poses == 0:
                logger.warning(f"No valid molecules found in {sdf_file.name}")
                failed += 1
            else:
                successful += 1
                total_poses += num_poses
            
        except Exception as e:
            logger.error(f"Error processing {sdf_file.name}: {e}")
            failed += 1
            continue
    
    logger.info(f"\nProcessing complete:")
    logger.info(f"  Files processed successfully: {successful}")
    logger.info(f"  Files failed: {failed}")
    logger.info(f"  Total poses saved: {total_poses}")
    logger.info(f"\nOutput saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Save all poses from SDF predictions with indexed filenames"
    )
    parser.add_argument(
        "-p", "--paths-config",
        dest="paths_config",
        type=str,
        required=True,
        help="Path to paths configuration YAML file"
    )
    parser.add_argument(
        "-n", "--run-name",
        dest="inference_run_name",
        type=str,
        required=True,
        help="Inference run name (folder name under inference_results_folder)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    conf = OmegaConf.load(args.paths_config)
    
    # Determine which datasets to process
    datasets_to_process = conf.test_dataset_types
    logger.info(f"Processing all datasets from config: {datasets_to_process}")
    
    # Process each dataset
    for dataset_name in datasets_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        process_sdf_folder(
            inference_results_folder=Path(conf.inference_results_folder),
            inference_run_name=args.inference_run_name,
            dataset_name=dataset_name,
        )


if __name__ == "__main__":
    main()
