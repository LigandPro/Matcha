import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import warnings
import sys
import copy
import json
from collections import defaultdict

from rdkit.Chem import RemoveAllHs
from rdkit import Chem
import prody
from prody import confProDy
import torch
import datamol as dm
from rdkit import RDLogger

from matcha.utils.datasets import get_datasets
from matcha.utils.paths import get_ligand_path
from matcha.dataset.pdbbind import complex_collate_fn
from matcha.utils.posebusters_utils import calc_posebusters
from matcha.utils.preprocessing import read_molecule
from matcha.utils.log import get_logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

logger = get_logger(__name__)
confProDy(verbosity='none')

RDLogger.DisableLog('rdApp.warning')

KEYS_VALID = ['not_too_far_away', 'no_internal_clash',
              'no_clashes', 'no_volume_clash', 'is_buried_fraction']


def get_data_for_buster(preds, dataset_data_dir, dataset_name, use_all_samples=False):
    data_for_buster = defaultdict(list)
    for uid, pred_data in preds.items():
        if '_conf' in uid:
            uid_real = uid.split('_conf')[0]
        else:
            uid_real = uid

        try:
            true_mol_path = get_ligand_path(
                uid, dataset_name, dataset_data_dir)
            orig_mol = read_molecule(true_mol_path, sanitize=False)
            try:
                orig_mol = RemoveAllHs(orig_mol, sanitize=True)
            except Exception as e:
                if dataset_name.startswith('pdbbind'):
                    try:
                        # try with mol2
                        true_mol_path = true_mol_path.replace('.sdf', '.mol2')
                        logger.info(f'Trying to read mol2 file {true_mol_path}')
                        orig_mol = read_molecule(true_mol_path, sanitize=False)
                        orig_mol = RemoveAllHs(orig_mol, sanitize=True)
                    except Exception as e:
                        orig_mol = RemoveAllHs(orig_mol, sanitize=False)
            if orig_mol is not None:
                true_pos = np.copy(orig_mol.GetConformer().GetPositions())
            else:
                logger.warning(f'Skip: {uid}')
                continue
        except:
            logger.warning(f'Skip: {uid}')
            continue

        samples = pred_data['sample_metrics']
        if use_all_samples:
            all_preds = []
            for sample in samples:
                pred_new = {
                    'transformed_orig': sample['pred_pos'],
                    'error_estimate_0': sample['error_estimate_0'],
                    'true_pos': true_pos,
                    'orig_mol': orig_mol,
                    'full_protein_center': np.zeros(3),
                }
                all_preds.append(pred_new)
            data_for_buster[uid_real] = all_preds
        else:
            pb_passed_count = np.array(
                [sample.get('posebusters_filters_passed_count_fast', 0) for sample in samples])
            best_pb_count = max(pb_passed_count)
            samples = [sample for sample in samples
                    if sample.get('posebusters_filters_passed_count_fast', 0) == best_pb_count]
            scores = [sample['error_estimate_0'] for sample in samples]
            best_score_idx = np.argmin(scores)
            best_sample = samples[best_score_idx]

            pred_new = {
                'transformed_orig': best_sample['pred_pos'],
                'error_estimate_0': best_sample['error_estimate_0'],
                'true_pos': true_pos,
                'orig_mol': orig_mol,
                'full_protein_center': np.zeros(3),
            }
            data_for_buster[uid_real] = [pred_new]
    return data_for_buster


def load_and_merge_all_stages(conf, inference_run_name):
    for dataset_name in conf.test_dataset_types:
        all_stage_updated_metrics = []
        for stage_idx in [0, 1, 2]:
            predicted_ligand_transforms_path = os.path.join(
                conf.inference_results_folder, inference_run_name, f'{dataset_name}_final_preds_{stage_idx+1}stage.npy')
            if os.path.exists(predicted_ligand_transforms_path):
                logger.info(f'Loading metrics from {predicted_ligand_transforms_path}')
                metrics = np.load(predicted_ligand_transforms_path, allow_pickle=True).item()
                all_stage_updated_metrics.append(metrics)
            else:
                logger.error(f'Merge {stage_idx + 1} stage failed: {predicted_ligand_transforms_path} does not exist')
        updated_metrics = merge_stages(all_stage_updated_metrics)
        merged_metrics_path = os.path.join(conf.inference_results_folder, inference_run_name, f'{dataset_name}_final_preds_merged.npy')
        logger.info(f'Saving merged metrics to {merged_metrics_path}')
        np.save(merged_metrics_path, [updated_metrics])


def merge_stages(all_stage_updated_metrics):
    updated_metrics = copy.deepcopy(all_stage_updated_metrics[0])
    for uid in updated_metrics.keys():
        for i in range(1, len(all_stage_updated_metrics)):
            samples = copy.deepcopy(all_stage_updated_metrics[i][uid]['sample_metrics'])
            updated_metrics[uid]['sample_metrics'].extend(samples)
    return updated_metrics


def save_all_to_sdf(conf, inference_run_name, one_file: bool = False, merge_stages: bool = False):
    """Save all predictions (not just the best) to SDF files.
    
    Args:
        conf: Configuration object with inference_results_folder and test_dataset_types.
        inference_run_name: Name of the inference run folder.
        one_file: If True, save all predictions for each ligand into a single SDF file
                  with multiple conformers. If False, save each prediction as a separate file.
    """
    for dataset_name in conf.test_dataset_types:
        if merge_stages:
            preds_name = f'{dataset_name}_final_preds_merged.npy'
        else:
            preds_name = f'{dataset_name}_final_preds.npy'
        logger.info(f'Loading predictions from {preds_name}')
        a = np.load(os.path.join(conf.inference_results_folder, inference_run_name,
                                 preds_name), allow_pickle=True).item()

        save_path = os.path.join(
            conf.inference_results_folder, inference_run_name, dataset_name, 'sdf_predictions')
        os.makedirs(save_path, exist_ok=True)
        logger.info(f'Saving all predictions to {save_path}')
        
        for uid, sample_data in tqdm(a.items(), desc='Saving all predictions'):
            if len(sample_data) == 0:
                continue

            orig_mol = sample_data['orig_mol']
            uid_real = uid.split('_mol')[0]
            
            if one_file:
                # Save all predictions into a single SDF file
                try:
                    writer = Chem.SDWriter(os.path.join(save_path, f'{uid_real}.sdf'))
                    for idx, sample in enumerate(sample_data['sample_metrics']):
                        pred_positions = sample['pred_pos']
                        mol = copy.deepcopy(orig_mol)
                        mol.GetConformer().SetPositions(pred_positions.astype(np.float64))
                        mol.SetProp('pred_idx', str(idx))
                        writer.write(mol, confId=0)
                    writer.close()
                except Exception as e:
                    continue
            else:
                # Create folder for this ligand and save each prediction separately
                uid_save_path = os.path.join(save_path, uid_real)
                os.makedirs(uid_save_path, exist_ok=True)

                for idx, sample in enumerate(sample_data['sample_metrics']):
                    pred_positions = sample['pred_pos']
                    mol = copy.deepcopy(orig_mol)
                    try:
                        mol.GetConformer().SetPositions(pred_positions.astype(np.float64))
                        idx_str = f"{idx:02d}" if idx < 100 else f"{idx}"
                        writer = Chem.SDWriter(os.path.join(
                            uid_save_path, f'{uid_real}_{idx_str}.sdf'))
                        writer.write(mol, confId=0)
                        writer.close()
                    except Exception as e:
                        continue


def calc_posebusters_for_data(data, lig_pos, orig_mol):
    lig_pos_for_posebusters = lig_pos
    lig_types_for_posebusters = data.ligand.x[:, 0] - 1
    pro_types_for_posebusters = data.protein.all_atom_names
    pro_pos_for_posebusters = data.protein.all_atom_pos + data.protein.full_protein_center
    lig_mol_for_posebusters = orig_mol
    names = data.name
    posebusters_results = calc_posebusters(lig_pos_for_posebusters, pro_pos_for_posebusters,
                                           lig_types_for_posebusters, pro_types_for_posebusters, names, lig_mol_for_posebusters)
    if posebusters_results is None:
        return None
    return np.array([posebusters_results[key] for key in KEYS_VALID if key in posebusters_results.keys()], dtype=object).transpose()


def compute_fast_filters_from_sdf(conf, inference_run_name, sdf_type='base', n_preds_to_use=None):
    """
    Compute fast filters for all molecules in multi-conformer SDF files.
    
    Args:
        conf: Configuration object with inference_results_folder and test_dataset_types.
        inference_run_name: Name of the inference run folder.
        sdf_type: Type of SDF predictions to process ('base' or 'minimized').
        n_preds_to_use: Number of predictions to use (default: None for all).
    
    Saves results to {exp_folder}/{dataset_name}_conf/filters_results.json 
    (or filters_results_minimized.json) with structure:
    {uid: {filter_field: np.array([values for all samples])}}
    """
    # Load datasets to get cached protein structures
    all_datasets = get_datasets(conf, splits=['test'],
                                predicted_ligand_transforms_path=None,
                                use_predicted_tr_only=False,
                                is_train_dataset=False,
                                n_preds_to_use=n_preds_to_use,
                                complex_collate_fn=complex_collate_fn)
    test_datasets = all_datasets['test']
    
    for dataset_name, dataset in test_datasets.items():
        # Determine input/output paths based on sdf_type
        if sdf_type == 'minimized':
            sdf_folder = f"{dataset_name}/minimized_sdf_predictions"
            output_file = "filters_results_minimized.json"
        else:
            sdf_folder = f"{dataset_name}/sdf_predictions"
            output_file = "filters_results.json"
        
        sdf_path = os.path.join(conf.inference_results_folder, inference_run_name, sdf_folder)
        output_path = os.path.join(conf.inference_results_folder, inference_run_name, 
                                   dataset_name, output_file)
        
        if not os.path.exists(sdf_path):
            logger.warning(f"SDF folder not found: {sdf_path}, skipping {dataset_name}")
            continue
        
        logger.info(f"Computing filters for {dataset_name} from {sdf_folder}")
        
        # Create uid -> protein data mapping from cached dataset
        uid_to_data = {}
        for data in dataset.complexes:
            name, conf_num = data.name.split('_conf')
            if conf_num == '0':
                uid_to_data[name.split('_mol')[0]] = data
        
        logger.info(f"Loaded {len(uid_to_data)} protein structures from cache")
        
        # Process each SDF file
        filters_results = {}
        number_failed = 0
        
        sdf_files = [f for f in os.listdir(sdf_path) if f.endswith('.sdf')]
        logger.info(f"Found {len(sdf_files)} SDF files to process")
        
        for sdf_filename in tqdm(sdf_files, desc=f"Computing filters for {dataset_name}"):
            uid = sdf_filename.replace('.sdf', '')
            
            if uid not in uid_to_data:
                logger.warning(f"Protein data not found in cache for {uid}, skipping")
                number_failed += 1
                continue
            
            sdf_file_path = os.path.join(sdf_path, sdf_filename)
            
            try:
                # Read all molecules from SDF
                mols = dm.read_sdf(str(sdf_file_path))
                mols = [Chem.RemoveAllHs(mol) for mol in mols if mol is not None]
                
                if len(mols) == 0:
                    logger.warning(f"No valid molecules in {sdf_filename}")
                    number_failed += 1
                    continue
                
                # Get protein data from cache
                data = uid_to_data[uid]
                
                # Extract ligand positions for all molecules
                lig_pos_all = []
                for mol in mols:
                    try:
                        pos = mol.GetConformer(0).GetPositions()
                        lig_pos_all.append(pos)
                    except Exception as e:
                        logger.error(f"Error getting positions for molecule in {sdf_filename}: {e}")
                        number_failed += 1
                        continue
                
                if len(lig_pos_all) == 0:
                    logger.warning(f"No valid conformers in {sdf_filename}")
                    number_failed += 1
                    continue
                
                # Stack positions: [n_mols, n_atoms, 3]
                lig_pos_stacked = np.stack(lig_pos_all)
                
                # Use first molecule as reference for atom types
                orig_mol = mols[0]
                
                # Compute filters for all molecules
                posebusters_results = calc_posebusters_for_data(data, lig_pos_stacked, orig_mol)
                
                if posebusters_results is None:
                    logger.error(f"Fast filters computation failed for {uid}")
                    number_failed += 1
                    continue
                
                # Store results as dict of arrays
                # posebusters_results shape: [n_mols, n_filters]
                # KEYS_VALID = ['not_too_far_away', 'no_internal_clash', 'no_clashes', 'no_volume_clash', 'is_buried_fraction']
                filters_results[uid] = {
                    'not_too_far_away': posebusters_results[:, 0].tolist(),
                    'no_internal_clash': posebusters_results[:, 1].tolist(),
                    'no_clashes': posebusters_results[:, 2].tolist(),
                    'no_volume_clash': posebusters_results[:, 3].tolist(),
                    'is_buried_fraction': posebusters_results[:, 4].tolist(),
                    'posebusters_filters_passed_count_fast': (posebusters_results[:, :4] == True).sum(axis=1).tolist(),
                }
                
            except Exception as e:
                logger.error(f"Error processing {sdf_filename}: {e}")
                number_failed += 1
                continue
        
        # Save results to JSON
        logger.info(f"Dataset {dataset_name} Number of failed: {number_failed}")
        logger.info(f"Saving filters to {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(filters_results, f, indent=2)
        
        logger.info(f"Successfully processed {len(filters_results)} UIDs for {dataset_name}")


# Stubs for CLI compatibility (v1 pipeline). In v2 use scripts instead.
def run_inference_pipeline(conf, run_name, n_preds_to_use, pocket_centers_filename=None,
                           docking_batch_limit=15000, scoring_batch_size=4):
    raise NotImplementedError(
        "Matcha v2: for inference use scripts/run_inference_pipeline.py and scripts/full_inference.py. See README."
    )


def compute_fast_filters(conf, inference_run_name, n_preds_to_use):
    raise NotImplementedError(
        "Matcha v2: use scripts/fast_filters_from_sdf.py and scripts/final_inference_pipeline.sh. See README."
    )


def save_best_pred_to_sdf(conf, inference_run_name):
    raise NotImplementedError(
        "Matcha v2: use scripts/gnina/select_top_gnina_poses.py and scripts/final_inference_pipeline.sh. See README."
    )
