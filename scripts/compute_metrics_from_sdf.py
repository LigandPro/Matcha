#!/usr/bin/env python3
"""
Script to compute metrics from best SDF predictions (best_minimized_predictions or best_base_predictions).

This script:
1. Reads single best pose predictions from SDF files
2. Computes symmetry-aware RMSD against all true poses (takes minimum)
3. Computes PoseBusters filters for the best true pose
4. Outputs metrics to CSV and detailed results to .npy file

Usage:
    python compute_metrics_from_sdf.py \
        --exp-folder /path/to/exp \
        --dataset-name pdbbind \
        --prediction-type best_minimized_predictions \
        --paths-config configs/paths/config.yaml
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

from rdkit import Chem
from rdkit.Chem import RemoveAllHs, rdMolAlign

from matcha.utils.paths import get_dataset_path, get_protein_path
from matcha.utils.preprocessing import read_molecule
from matcha.utils.log import get_logger
from matcha.utils.paths import get_ligand_path
import prody
from prody import confProDy

logger = get_logger(__name__)
confProDy(verbosity='none')


def compute_posebusters_tests(ligand_mol, true_mol, protein_path, posebusters_config='redock'):
    """
    Compute full PoseBusters tests for a single ligand-protein complex.
    
    Args:
        ligand_mol: RDKit molecule object (predicted, without hydrogens)
        true_mol: RDKit molecule object (true ligand, without hydrogens)
        protein_path: Path to protein PDB file
        posebusters_config: PoseBusters configuration ('redock' or 'dock')
    
    Returns:
        dict: Dictionary with PoseBusters test results or None if failed
    """
    try:
        from posebusters import PoseBusters
        import posebusters
        
        # Define test categories
        docking_only_tests = ['mol_pred_loaded', 'mol_cond_loaded', 'sanitization', 'inchi_convertible',
                              'all_atoms_connected', 'bond_lengths', 'bond_angles', 'internal_steric_clash',
                              'aromatic_ring_flatness', 'non-aromatic_ring_non-flatness', 'double_bond_flatness',
                              'internal_energy', 'protein-ligand_maximum_distance', 'minimum_distance_to_protein',
                              'minimum_distance_to_organic_cofactors', 'minimum_distance_to_inorganic_cofactors',
                              'minimum_distance_to_waters', 'volume_overlap_with_protein',
                              'volume_overlap_with_organic_cofactors', 'volume_overlap_with_inorganic_cofactors',
                              'volume_overlap_with_waters']
        
        if posebusters_config == 'redock':
            redock_extra_tests = ['double_bond_stereochemistry', 'mol_true_loaded', 'molecular_bonds',
                                  'molecular_formula', 'rmsd_≤_2å', 'tetrahedral_chirality']
        else:
            redock_extra_tests = []
        
        # Initialize PoseBusters
        buster = PoseBusters(config=posebusters_config, max_workers=0, chunk_size=None)
        
        # Run PoseBusters with timeout
        results = buster.bust(
            mol_pred=[ligand_mol],
            mol_true=true_mol if posebusters_config == 'redock' else None,
            mol_cond=protein_path,
            full_report=True,
        )
        
        # Extract results for the single molecule
        if len(results) == 0:
            return None
        
        # Get the first (and only) row
        result_row = results.iloc[0]
        
        # Extract test results
        posebusters_filters = result_row[docking_only_tests + redock_extra_tests].values
        docking_filters_passed = result_row[docking_only_tests].sum()
        
        if posebusters_config == 'redock':
            redock_passed_extra = result_row[redock_extra_tests].sum()
        else:
            redock_passed_extra = 0
        
        all_tests_passed = docking_filters_passed + redock_passed_extra
        
        # Build result dictionary
        filter_results = {
            'posebusters_filters': posebusters_filters.tolist(),
            'posebusters_filters_passed_count': int(docking_filters_passed),
            'all_posebusters_filters_passed_count': int(all_tests_passed),
        }
        
        # Add individual test results as separate fields
        for test_name in docking_only_tests + redock_extra_tests:
            filter_results[test_name] = bool(result_row[test_name])
        
        # Add fast filter shortcuts (for backward compatibility)
        filter_results['not_too_far_away'] = bool(result_row.get('protein-ligand_maximum_distance', False))
        filter_results['no_clashes'] = bool(result_row.get('minimum_distance_to_protein', False))
        filter_results['no_volume_clash'] = bool(result_row.get('volume_overlap_with_protein', False))
        filter_results['no_internal_clash'] = bool(result_row.get('internal_steric_clash', False))

        # overall pass rate
        filter_results['all_posebusters_filters_passed_count'] = sum(filter_results['posebusters_filters'])
        return filter_results
    
    except Exception as e:
        logger.error(f"Error computing PoseBusters tests: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def load_true_ligands(uid, dataset_name, dataset_path):
    """
    Load true ligand poses from dataset.
    
    Args:
        uid: Complex UID
        dataset_name: Dataset name
        dataset_path: Path to dataset
    
    Returns:
        list: List of RDKit molecules (true poses)
    """
    mols_true = []
    
    if dataset_name.startswith('dockgen'):
        ref_lig_path = get_ligand_path(uid, dataset_name, dataset_path)
        mol_tmp = read_molecule(ref_lig_path, remove_hs=False, sanitize=True)
        mols_true = [mol_tmp]
    elif dataset_name.startswith('pdbbind'):
        ref_lig_path = get_ligand_path(uid, dataset_name, dataset_path)
        mol_tmp = read_molecule(ref_lig_path, remove_hs=False, sanitize=True, strict=True)
        if mol_tmp is None:
            ref_lig_path = ref_lig_path.replace('.mol2', '.sdf')
            logger.warning(f'Failed to read mol2 file {ref_lig_path}. Trying to read sdf file instead.')
            mol_tmp = read_molecule(ref_lig_path, remove_hs=False, sanitize=True)
        mols_true = [mol_tmp]
        
        if len(mols_true) == 0:
            # Try SDF
            sdf_path = os.path.join(dataset_path, uid, f"{uid}_ligand.sdf")
            if os.path.exists(sdf_path):
                try:
                    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
                    mols_true = [mol for mol in supplier if mol is not None]
                except Exception as e:
                    logger.warning(f"Error reading SDF file {sdf_path}: {e}")
                    mols_true = []
    else: # Astex, PoseBusters
        # For Astex, PoseBusters - use {uid}_ligands.sdf (note: plural)
        sdf_path = os.path.join(dataset_path, uid, f"{uid}_ligands.sdf")
        if os.path.exists(sdf_path):
            try:
                supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
                mols_true = [mol for mol in supplier if mol is not None]
            except Exception as e:
                logger.warning(f"Error reading SDF file {sdf_path}: {e}")
                mols_true = []

    mols_true = [Chem.RemoveAllHs(mol, sanitize=True) for mol in mols_true if mol is not None]
    return mols_true


def compute_metrics_for_uid(uid, pred_sdf_path, dataset_name, dataset_path):
    """
    Compute metrics for a single UID.
    
    Args:
        uid: Complex UID
        pred_sdf_path: Path to predicted ligand SDF file
        dataset_name: Dataset name
        dataset_path: Path to dataset
    
    Returns:
        dict: Dictionary with metrics or None if failed
    """
    try:        
        # Load predicted molecule
        mol_pred = read_molecule(str(pred_sdf_path), sanitize=True)
        if mol_pred is None:
            logger.warning(f"Failed to read predicted molecule for {uid}")
            return None
        
        # Load true ligands
        mols_true = load_true_ligands(uid, dataset_name, dataset_path)
        
        if len(mols_true) == 0:
            logger.warning(f"No true ligands found for {uid}")
            return None
        
        # Remove hydrogens for RMSD calculation
        mol_pred_no_hs = RemoveAllHs(mol_pred, sanitize=True)
        pred_pos = mol_pred_no_hs.GetConformer(0).GetPositions()
        pred_center = pred_pos.mean(axis=0)
        
        # Compute RMSD against all true poses
        rmsds = []
        tr_errors = []
        
        for mol_true in mols_true:
            try:
                mol_true_no_hs = RemoveAllHs(mol_true, sanitize=True)
                true_pos = mol_true_no_hs.GetConformer(0).GetPositions()
                true_center = true_pos.mean(axis=0)
                
                # Compute RMSD (symmetry-aware)
                rmsd = rdMolAlign.CalcRMS(mol_pred_no_hs, mol_true_no_hs)
                rmsds.append(rmsd)
                
                # Compute translation error
                tr_err = np.linalg.norm(pred_center - true_center)
                tr_errors.append(tr_err)
                
            except Exception as e:
                logger.warning(f"Error computing RMSD for {uid} against true pose: {e}")
                continue
        
        if len(rmsds) == 0:
            logger.warning(f"No valid RMSD computations for {uid}")
            return None
        
        # Take minimum RMSD and corresponding tr_error
        rmsds = np.array(rmsds)
        tr_errors = np.array(tr_errors)
        best_true_idx = np.argmin(rmsds)
        best_rmsd = rmsds[best_true_idx]
        best_tr_error = tr_errors[best_true_idx]
        
        # Get protein path
        protein_path = get_protein_path(uid, dataset_name, dataset_path)
        
        # Get best matching true molecule
        best_true_mol = mols_true[best_true_idx]
        best_true_mol_no_hs = RemoveAllHs(best_true_mol, sanitize=True)
        
        # Compute PoseBusters tests
        posebusters_results = compute_posebusters_tests(mol_pred_no_hs, best_true_mol_no_hs, protein_path)
        
        results = {
            'uid': uid,
            'rmsd': best_rmsd,
            'tr_error': best_tr_error,
            'best_true_idx': best_true_idx,
            'all_rmsds': rmsds.tolist(),
            'all_tr_errors': tr_errors.tolist(),
        }
        
        if posebusters_results is not None:
            results.update(posebusters_results)
        else:
            # Add default values for PoseBusters tests
            logger.warning(f"No PoseBusters results for {uid}, using defaults")
            results.update({
                'posebusters_filters': [],
                'posebusters_filters_passed_count': 0,
                'all_posebusters_filters_passed_count': 0,
                'not_too_far_away': False,
                'no_clashes': False,
                'no_volume_clash': False,
                'no_internal_clash': False,
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {uid}: {e}")
        return None


def compute_metrics_from_sdf(conf, exp_name, dataset_name, prediction_type):
    """
    Compute metrics for all predictions in the specified folder.
    
    Args:
        conf: Configuration object
        exp_name: Experiment name
        dataset_name: Dataset name
        prediction_type: Type of predictions ('best_minimized_predictions' or 'best_base_predictions')
    """
    exp_folder = Path(conf.inference_results_folder, exp_name)
    dataset_path = get_dataset_path(dataset_name, conf)
    csv_path = exp_folder / f"{dataset_name}_{prediction_type}_sdf_metrics.csv"

    logger.info(f"csv_path: {csv_path}")

    if '/' in prediction_type:
        os.makedirs(exp_folder / f"{dataset_name}_{prediction_type.split('/')[0]}", exist_ok=True)
    
    # Path to predictions
    pred_folder = exp_folder / dataset_name / prediction_type
    
    if not pred_folder.exists():
        logger.error(f"Predictions folder not found: {pred_folder}")
        return
    
    logger.info(f"Processing predictions from {pred_folder}")
    
    # Get all UIDs
    uids = [d.split(".sdf")[0] for d in os.listdir(pred_folder)]
    logger.info(f"Found {len(uids)} UIDs to process")

    if dataset_name.startswith('posebusters'):
        posebusters_ids_path = 'data/posebusters_v2.txt'
        with open(posebusters_ids_path, 'r') as file:
            posebusters_ids = [line.strip() for line in file.readlines()]
        uids = [uid for uid in uids if uid in posebusters_ids]
        logger.info(f"Found {len(uids)} PoseBusters UIDs to process")

    # Process each UID
    all_results = []
    failed_uids = []
    
    for uid in tqdm(uids, desc=f"Computing metrics for {exp_name}, {dataset_name}"):
        pred_sdf_path = pred_folder / f"{uid}.sdf"
        
        if not pred_sdf_path.exists():
            logger.warning(f"Prediction file not found: {pred_sdf_path}")
            failed_uids.append(uid)
            continue
        
        result = compute_metrics_for_uid(
            uid, 
            pred_sdf_path, 
            dataset_name, 
            dataset_path,
        )
        
        if result is not None:
            all_results.append(result)
        else:
            failed_uids.append(uid)
    
    logger.info(f"Successfully processed {len(all_results)} UIDs")
    logger.info(f"Failed: {len(failed_uids)} UIDs")
    
    # Save results to CSV
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        
        # Select columns for CSV (exclude lists)
        csv_columns = ['uid', 'rmsd', 'tr_error', 'best_true_idx', 
                      'posebusters_filters_passed_count', 'all_posebusters_filters_passed_count',
                      'not_too_far_away', 'no_clashes', 'no_volume_clash', 
                      'no_internal_clash']
        
        # Add individual PoseBusters test columns if they exist
        posebuster_test_columns = [col for col in df.columns if col.startswith('mol_') or 
                                   col in ['sanitization', 'inchi_convertible', 'all_atoms_connected',
                                          'bond_lengths', 'bond_angles', 'internal_steric_clash',
                                          'aromatic_ring_flatness', 'double_bond_flatness',
                                          'internal_energy', 'double_bond_stereochemistry',
                                          'tetrahedral_chirality', 'molecular_bonds', 'molecular_formula']]
        csv_columns.extend([col for col in posebuster_test_columns if col in df.columns])
        
        # Filter to only existing columns
        csv_columns = [col for col in csv_columns if col in df.columns]
        df_csv = df[csv_columns]
        
        df_csv.to_csv(csv_path, index=False)
        logger.info(f"Saved metrics to {csv_path}")
        
        # Print summary statistics
        logger.info(f"\n=== Summary Statistics for {dataset_name} ===")
        logger.info(f"RMSD < 2Å: {(df['rmsd'] < 2.0).sum()} / {len(df)} ({100 * (df['rmsd'] < 2.0).mean():.1f}%)")
        logger.info(f"RMSD < 5Å: {(df['rmsd'] < 5.0).sum()} / {len(df)} ({100 * (df['rmsd'] < 5.0).mean():.1f}%)")
        logger.info(f"Median RMSD: {df['rmsd'].median():.3f} Å")
        logger.info(f"Mean RMSD: {df['rmsd'].mean():.3f} Å")
        logger.info(f"Median tr_error: {df['tr_error'].median():.3f} Å")

        if 'posebusters_filters_passed_count' in df.columns:
            logger.info(f"Mean PoseBusters docking filters passed: {df['posebusters_filters_passed_count'].mean():.2f} / 20")
        
        if 'all_posebusters_filters_passed_count' in df.columns:
            logger.info(f"Max all_posebusters_filters_passed_count: {df['all_posebusters_filters_passed_count'].max()}")
            logger.info(f"Mean all PoseBusters tests passed: {df['all_posebusters_filters_passed_count'].mean():.2f} / 27")
            all_pb_passed = (df['all_posebusters_filters_passed_count'] == 27).sum()
            logger.info(f"All PoseBusters tests passed: {all_pb_passed} / {len(df)} ({100 * all_pb_passed / len(df):.1f}%)")
            logger.info(f"RMSD < 2Å & All PoseBusters tests passed: {(df['all_posebusters_filters_passed_count'] == 27).mean():.3f}")

        # Save detailed results to .npy
        # npy_path = exp_folder / f"{dataset_name}_{prediction_type}_sdf_final_preds.npy"
        # np.save(npy_path, [all_results])
        # logger.info(f"Saved detailed results to {npy_path}")
    else:
        logger.error("No results to save")


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics from best SDF predictions"
    )
    parser.add_argument(
        "-n", "--run-name",
        type=str,
        required=True,
        help="Inference run name"
    )
    parser.add_argument(
        "--prediction-type",
        type=str,
        default="best_base_predictions",
        help="Type of predictions to process (default: best_minimized_predictions)"
    )
    parser.add_argument(
        "-p", "--paths-config",
        type=str,
        required=True,
        help="Path to paths configuration YAML file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    conf = OmegaConf.load(args.paths_config)
    logger.info(f"Prediction type: {args.prediction_type}")

    prediction_types = [args.prediction_type]
    # prediction_types = [f"indexed_minimized_predictions/{i}_minimized" for i in np.arange(100, 120)]
    # conf.test_dataset_types = ['posebusters_conf']

    run_names = [args.run_name]
    # run_names = ['af3', 
    #              'diffdock', 
    #              'SurfDock_p2rank', 'DynamicBind', 
    #              'DynamicBind_relaxed', 'chai', 'neuralplexer', 'FD', 
    #              'gnina_fpocket', 'gnina_p2rank', 'gnina_full_protein', 
    #              'gnina_ligand_box', 'vina_fpocket', 'vina_p2rank', 
    #              'vina_full_protein', 'vina_ligand_box', 'smina_fpocket', 
    #              'smina_p2rank', 'smina_full_protein', 'smina_ligand_box', 
    #              'unimol_NONE', 'unimol_p2rank', 'unimol_fpocket', 
    #              'boltz', 'boltz_pocket_4A', 'boltz_pocket_10A'
    #              ]
    # run_names = [f'{name}_base' for name in run_names] + [f'{name}_pocket' for name in run_names]


    for run_name in run_names:
        for prediction_type in prediction_types:
            for dataset_name in conf.test_dataset_types:
                logger.info(f'Computing metrics for {run_name} {dataset_name} {prediction_type}')
                compute_metrics_from_sdf(
                    conf=conf,
                    exp_name=run_name,
                    dataset_name=dataset_name,
                    prediction_type=prediction_type,
                )


if __name__ == "__main__":
    main()
