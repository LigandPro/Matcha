#!/usr/bin/env python3
"""
Script to select top-scored poses from SDF predictions based on gnina scores.

This script reads SDF files containing multiple poses (all poses in one file),
extracts gnina scores from each pose, selects the top-scored pose, and saves it
in the folder structure required for compute_aligned_rmsd.py.

Supports both base and minimized predictions, and optional filter-based selection.

Expected input structure:
- {inference_results_folder}/{run_name}/{dataset_name}/base_sdf_predictions/ (or minimized_sdf_predictions/)
  - {uid}.sdf
  - ...
- {inference_results_folder}/{run_name}/{dataset_name}/filters_results.json (or filters_results_minimized.json)

Output structure:
- {inference_results_folder}/{run_name}/{dataset_name}/best_base_predictions/ (or best_minimized_predictions/)
  - {uid}.sdf

Usage:
    python select_top_gnina_poses.py \
        -p configs/paths/paths.yaml \
        -n my_experiment \
        --score-type Affinity \
        --use-minimized \
        --use-filters
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf

# Suppress RDKit warnings about 2D/3D molecule tags (common after gnina minimization)
# This must be done before importing Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

from rdkit import Chem
from rdkit.Chem import SDWriter

from matcha.utils.log import get_logger

logger = get_logger(__name__)


def extract_gnina_score(mol: Chem.Mol, score_type: str = "CNNscore", use_minimized: bool = True) -> Optional[float]:
    """
    Extract gnina score from molecule properties.
    
    Args:
        mol: RDKit molecule object
        score_type: Type of score to extract ("CNNscore", "CNNaffinity", or "Affinity")
        use_minimized: Whether to look for minimized property names
    
    Returns:
        Score value or None if not found
    """
    # Try different possible property names
    if use_minimized:
        possible_names = [
            f"minimized{score_type}" if score_type != "Affinity" else "minimizedAffinity",
            f"minimizedCNNscore" if score_type == "CNNscore" else None,
            f"minimizedCNNaffinity" if score_type == "CNNaffinity" else None,
            score_type,
        ]
    else:
        possible_names = [
            score_type,
            f"minimized{score_type}" if score_type != "Affinity" else "minimizedAffinity",
        ]
    
    for prop_name in possible_names:
        if prop_name is None:
            continue
        if mol.HasProp(prop_name):
            try:
                return float(mol.GetProp(prop_name))
            except (ValueError, TypeError):
                continue
    
    # Try to get all properties and search for score-like names
    props = mol.GetPropsAsDict(includePrivate=True, includeComputed=True)
    for key, value in props.items():
        if score_type.lower() in key.lower():
            try:
                return float(value)
            except (ValueError, TypeError):
                continue
    
    return None


def find_top_scored_molecule(
    sdf_path: Path, 
    score_type: str = "CNNscore",
    use_minimized: bool = True,
    filters_data: Optional[Dict[str, Any]] = None,
    uid: Optional[str] = None,
    n_samples: int = 20,
) -> Optional[Tuple[Chem.Mol, float, int]]:
    """
    Read SDF file and find the molecule with the highest gnina score.
    
    Args:
        sdf_path: Path to SDF file containing multiple poses
        score_type: Type of score to use for ranking ("CNNscore", "CNNaffinity", or "Affinity")
        use_minimized: Whether to look for minimized property names
        filters_data: Optional dict with filter results {uid: {filter_field: [values]}}
        uid: UID for this molecule (required if using filters)
    
    Returns:
        Tuple of (best molecule, best score, index) or None if no valid molecules found
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    
    # Read all molecules and extract scores
    mols = []
    scores = []
    
    # Try direct property access first (most common case for minimized files)
    if use_minimized:
        prop_name = f"minimized{score_type}" if score_type != "Affinity" else "minimizedAffinity"
    else:
        prop_name = score_type

    # logger.warning('POCKET_AWARE SETUP')
    # keep_mask = np.concatenate([np.arange(n_samples), 
    #                             np.arange(40, 40 + n_samples), 
    #                             # np.arange(80, 80 + n_samples)
    #                             ])

    # keep_mask = np.concatenate([
    #                             np.arange(n_samples), 
    #                             np.arange(40, 40 + n_samples), 
    #                             # np.arange(80, 80 + n_samples)
    #                             ])
    # logger.warning(f"stage3 {uid}, {keep_mask}")

    # keep_mask = np.concatenate([
    #                             np.arange(n_samples), 
    #                             np.arange(40, 40 + n_samples), 
    #                             np.arange(80, 80 + n_samples)
    #                             ])

    # keep_mask = np.arange(2 * n_samples)
    keep_mask = np.arange(3 * n_samples)
    
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        
        if i not in keep_mask:
            # logger.info(f'Skip {uid} {i}')
            continue
        
        # Try direct property access first (faster)
        if mol.HasProp(prop_name):
            try:
                score = float(mol.GetProp(prop_name))
                mols.append(mol)
                scores.append(score)
                continue
            except (ValueError, TypeError):
                pass
        
        # Fall back to search if direct access fails
        score = extract_gnina_score(mol, score_type, use_minimized)
        if score is not None:
            mols.append(mol)
            scores.append(score)
    
    if len(mols) == 0:
        return None
    
    scores = np.array(scores)
    
    # Apply filters if provided
    valid_indices = np.arange(len(mols))
    if filters_data is not None and uid is not None and uid in filters_data:
        filter_counts = np.array(filters_data[uid]['posebusters_filters_passed_count_fast'])[keep_mask]
        
        # Ensure filter counts match number of molecules
        if len(filter_counts) != len(mols):
            logger.warning(f"Filter count mismatch for {uid}: {len(filter_counts)} filters vs {len(mols)} molecules")
        
        # Filter by minimum passed count
        max_filters_passed = np.max(filter_counts)
        logger.debug(f'Use filtering with {max_filters_passed} for {uid}')
        valid_mask = filter_counts >= max_filters_passed
        valid_indices = valid_indices[valid_mask]
        
        if len(valid_indices) == 0:
            # No molecules pass filters, fall back to all molecules
            print(f"Warning: No molecules pass {max_filters_passed} filters for {uid}, using all molecules")
            valid_indices = np.arange(len(mols))
    else:
        logger.debug(f'No filtering for {uid}')
    
    # Select best among valid indices
    valid_scores = scores[valid_indices]
    
    # For CNNscore and CNNaffinity, higher is better
    # For Affinity, lower is better (more negative = better binding)
    if score_type == "Affinity" or score_type == "minimizedAffinity":
        # Lower (more negative) affinity is better - use argmin
        best_valid_idx = np.argmin(valid_scores)
    else:
        # Higher score is better for CNNscore and CNNaffinity - use argmax
        best_valid_idx = np.argmax(valid_scores)
    
    best_idx = valid_indices[best_valid_idx]
    best_mol = mols[best_idx]
    best_score = float(scores[best_idx])
    
    return best_mol, best_score, best_idx


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


def process_sdf_folder(
    inference_results_folder: Path,
    inference_run_name: str,
    dataset_name: str,
    score_type: str = "CNNscore",
    use_minimized: bool = True,
    use_filters: bool = False,
    n_samples: int = 20,
):
    """
    Process all SDF files in input folder and save top-scored poses.
    
    Args:
        inference_results_folder: Base inference results folder from config
        inference_run_name: Name of the inference run
        dataset_name: Name of the dataset (used in folder structure, e.g., 'pdbbind', 'astex')
        score_type: Type of gnina score to use for ranking
        use_minimized: Whether to process minimized predictions (default: True)
        use_filters: Whether to use filter-based selection (default: False)
    """
    inference_results_folder = Path(inference_results_folder)
    exp_folder = inference_results_folder / inference_run_name

    logger.info(f"N samples: {n_samples}")
    
    # Determine input/output paths based on use_minimized
    if use_minimized:
        input_folder = exp_folder / dataset_name / "minimized_sdf_predictions"
        if n_samples != 40:
            output_folder_name = f"best_minimized_predictions_{n_samples}"
        else:
            output_folder_name = "best_minimized_predictions"
        filters_path = exp_folder / dataset_name / "filters_results_minimized.json"
    else:
        input_folder = exp_folder / dataset_name / "base_sdf_predictions"
        output_folder_name = "best_base_predictions"
        filters_path = exp_folder / dataset_name / "filters_results.json"

    if use_filters:
        output_folder_name = f"{output_folder_name}_filtered"

    # logger.warning(f"Stage 3")
    # output_folder_name = f"{output_folder_name}_stage3"

    logger.info(f"output_folder_name: {output_folder_name}")
    output_folder = exp_folder / dataset_name / output_folder_name

    # Create output directory structure
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all SDF files
    sdf_pattern = '*.sdf'
    sdf_files = list(input_folder.glob(sdf_pattern))
    
    if len(sdf_files) == 0:
        print(f"Warning: No SDF files found in {input_folder} with pattern {sdf_pattern}")
        return
    
    print(f"Found {len(sdf_files)} SDF files to process")
    print(f"Using {'minimized' if use_minimized else 'base'} predictions")
    
    # Load filters if requested
    filters_data = None
    if use_filters:
        if not filters_path.exists():
            logger.warning(f"Filters file not found at {filters_path}, proceeding without filters")
        else:
            logger.info(f"Loading filters from {filters_path}")
            with open(filters_path, 'r') as f:
                filters_data = json.load(f)
            logger.info(f"Loaded filters for {len(filters_data)} UIDs")
    
    successful = 0
    failed = 0
    
    for sdf_file in tqdm(sdf_files, desc="Processing SDF files"):
        try:
            # Extract UID from filename
            uid = extract_uid_from_filename(sdf_file.name)
            
            # Find top-scored molecule
            result = find_top_scored_molecule(
                sdf_file, 
                score_type, 
                use_minimized,
                filters_data,
                uid,
                n_samples,
            )
            
            if result is None:
                print(f"Warning: No valid molecules with {score_type} found in {sdf_file.name}")
                failed += 1
                continue
            
            best_mol, best_score, best_idx = result
            
            # Save top-scored molecule
            output_sdf = output_folder / f"{uid}.sdf"
            writer = SDWriter(str(output_sdf))
            writer.write(best_mol)
            writer.close()
            
            successful += 1
            
        except Exception as e:
            print(f"Error processing {sdf_file.name}: {e}")
            failed += 1
            continue
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nOutput saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Select top-scored poses from SDF predictions based on gnina scores"
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
    parser.add_argument(
        "--score-type",
        type=str,
        dest="score_type",
        default="Affinity",
        choices=["CNNscore", "CNNaffinity", "Affinity"],
        help="Type of gnina score to use for ranking (default: Affinity)"
    )
    parser.add_argument(
        "--use-filters",
        dest="use_filters",
        action="store_true",
        help="Use filter-based selection (filters JSON auto-detected from run folder)"
    )
    parser.add_argument(
        "--n-samples",
        dest="n_samples",
        type=int,
        help="Number of samples to use for filtering",
        default=40,
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

        # use_filters = args.use_filters
        use_filters = True
        
        # process_sdf_folder(
        #     inference_results_folder=Path(conf.inference_results_folder),
        #     inference_run_name=args.inference_run_name,
        #     dataset_name=dataset_name,
        #     score_type=args.score_type,
        #     use_minimized=False,
        #     use_filters=use_filters,
        # )

        process_sdf_folder(
            inference_results_folder=Path(conf.inference_results_folder),
            inference_run_name=args.inference_run_name,
            dataset_name=dataset_name,
            score_type=args.score_type,
            use_minimized=True,
            use_filters=use_filters,
            n_samples=args.n_samples,
        )


if __name__ == "__main__":
    main()
