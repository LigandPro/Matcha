import os
import time
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
        suffix = '_final_preds_merged.npy' if merge_stages else '_final_preds.npy'
        preds_name = f'{dataset_name}{suffix}'
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
                    except Exception:
                        continue


def calc_posebusters_for_data(data, lig_pos, orig_mol):
    lig_types = data.ligand.x[:, 0] - 1
    pro_pos = data.protein.all_atom_pos + data.protein.full_protein_center
    pro_types = data.protein.all_atom_names
    posebusters_results = calc_posebusters(lig_pos, pro_pos, lig_types, pro_types, data.name, orig_mol)
    if posebusters_results is None:
        return None
    return np.array([posebusters_results[key] for key in KEYS_VALID if key in posebusters_results], dtype=object).transpose()


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
        # PDBBindWithSortedBatching wraps the actual dataset in .dataset
        uid_to_data = {}
        inner = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        for data in inner.complexes:
            name, conf_num = data.name.split('_conf')
            if conf_num == '0':
                uid_to_data[name.split('_mol')[0]] = data
        
        logger.info(f"Loaded {len(uid_to_data)} protein structures from cache")
        
        # Process each SDF file
        filters_results = {}
        number_failed = 0
        
        sdf_files = [Path(sdf_path) / f for f in os.listdir(sdf_path) if f.endswith('.sdf')]
        logger.info(f"Found {len(sdf_files)} SDF files to process")
        
        for sdf_file_path in tqdm(sdf_files, desc=f"Computing filters for {dataset_name}"):
            uid = sdf_file_path.stem
            
            if uid not in uid_to_data:
                logger.warning(f"Protein data not found in cache for {uid}, skipping")
                number_failed += 1
                continue
                        
            try:
                # Read all molecules from SDF
                mols = dm.read_sdf(str(sdf_file_path))
                mols = [Chem.RemoveAllHs(mol) for mol in mols if mol is not None]
                
                if not mols:
                    logger.warning(f"No valid molecules in {sdf_file_path.name}")
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
                        logger.error(f"Error getting positions for molecule in {sdf_file_path.name}: {e}")
                        number_failed += 1
                        continue
                
                if len(lig_pos_all) == 0:
                    logger.warning(f"No valid conformers in {sdf_file_path.name}")
                    number_failed += 1
                    continue
                
                # Stack positions: [n_mols, n_atoms, 3]
                lig_pos_stacked = np.stack(lig_pos_all)
                
                # Use first molecule as reference for atom types
                orig_mol = mols[0]
                
                # Compute filters for all molecules
                results = calc_posebusters_for_data(data, lig_pos_stacked, orig_mol)
                
                if results is None:
                    logger.error(f"Fast filters computation failed for {uid}")
                    number_failed += 1
                    continue
                
                # Store results as dict of arrays
                # results shape: [n_mols, n_filters]
                # KEYS_VALID = ['not_too_far_away', 'no_internal_clash', 'no_clashes', 'no_volume_clash', 'is_buried_fraction']
                passed_count_fast = results[:, :4].sum(axis=1).tolist()
                filters_results[uid] = {
                    'not_too_far_away': results[:, 0].tolist(),
                    'no_internal_clash': results[:, 1].tolist(),
                    'no_clashes': results[:, 2].tolist(),
                    'no_volume_clash': results[:, 3].tolist(),
                    'is_buried_fraction': results[:, 4].tolist(),
                    'posebusters_filters_passed_count_fast': passed_count_fast,
                }
                
            except Exception as e:
                logger.error(f"Error processing {sdf_file_path.name}: {e}")
                number_failed += 1
                continue
        
        # Save results to JSON
        logger.info(f"Dataset {dataset_name} Number of failed: {number_failed}")
        logger.info(f"Saving filters to {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(filters_results, f, indent=2)
        
        logger.info(f"Successfully processed {len(filters_results)} UIDs for {dataset_name}")


def run_v2_inference_pipeline(
    conf,
    run_name,
    n_preds_to_use,
    pocket_centers_filename=None,
    docking_batch_limit=15000,
    num_workers=0,
    progress_callback=None,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    compute_torsion_angles_pred: bool = False,
):
    """Run the full v2 3-stage inference pipeline.

    Replaces the v1 ``run_inference_pipeline`` and
    ``save_best_pred_to_sdf`` stubs with a single entry-point used by both
    CLI and TUI.

    Args:
        conf: OmegaConf config (uses ``results_folder`` or ``checkpoints_folder`` for checkpoints root).
        run_name: Name of the inference run (folder under ``inference_results_folder``).
        n_preds_to_use: Number of pose samples to generate per ligand.
        pocket_centers_filename: Optional path to precomputed pocket centres (skip stage 1).
        docking_batch_limit: Token budget per batch.
        num_workers: DataLoader workers (0 = main-process; safer for TUI).
        progress_callback: ``fn(event_type, stage, name, elapsed, progress)`` for UI updates.
    """
    from torch.utils.data import DataLoader
    from matcha.models import MatchaModel
    from matcha.utils.inference import euler, load_from_checkpoint, run_evaluation
    from matcha.utils.metrics import construct_output_dict
    from matcha.dataset.pdbbind import complex_collate_fn, dummy_ranking_collate_fn

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(conf.seed)

    from matcha.utils.device import resolve_device
    requested_device = conf.get("device", None) if hasattr(conf, "get") else getattr(conf, "device", None)
    device = resolve_device(requested_device)
    num_steps = 10

    conf.batch_limit = docking_batch_limit

    # v2 3-stage pipeline definition (no scoring model)
    pipeline = {
        'docking': [
            {
                'model_path': 'matcha_pipeline/stage1/',
                'model_kwargs': {},
                'dataset_kwargs': {'n_preds_to_use': n_preds_to_use},
            },
            {
                'model_path': 'matcha_pipeline/stage2/',
                'model_kwargs': {'use_qk_bias': False},
                'dataset_kwargs': {'n_preds_to_use': n_preds_to_use},
            },
            {
                'model_path': 'matcha_pipeline/stage3/',
                'model_kwargs': {},
                'dataset_kwargs': {'use_predicted_tr_only': False, 'n_preds_to_use': n_preds_to_use},
            },
        ],
    }

    # Save pipeline config
    os.makedirs(os.path.join(conf.inference_results_folder, run_name), exist_ok=True)
    with open(os.path.join(conf.inference_results_folder, run_name, 'config.json'), 'w') as f:
        json.dump(pipeline, f)

    checkpoint_root = (
        conf.get("results_folder", None)
        if hasattr(conf, "get")
        else getattr(conf, "results_folder", None)
    )
    if checkpoint_root is None:
        checkpoint_root = (
            conf.get("checkpoints_folder", None)
            if hasattr(conf, "get")
            else getattr(conf, "checkpoints_folder", None)
        )
    if checkpoint_root is None:
        raise ValueError("Missing checkpoints root in config: set `results_folder` or `checkpoints_folder`.")

    # Load all stage models
    docking_modules = pipeline['docking']
    for module in docking_modules:
        model = MatchaModel(**module['model_kwargs'], conf=conf)
        model = load_from_checkpoint(model, os.path.join(
            checkpoint_root, module['model_path']))
        model.to(device)
        model.eval()
        module['model'] = model

    logger.info(f"Starting v2 inference pipeline: {run_name}")
    logger.info(f"Generating {n_preds_to_use} samples per ligand, num_steps={num_steps}")

    def get_dataloader_docking(dataset):
        return DataLoader(
            dataset, batch_size=1, shuffle=False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=dummy_ranking_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )

    dataset_names = conf.test_dataset_types
    timings = {
        "stage1_sec": 0.0,
        "stage2_sec": 0.0,
        "stage3_sec": 0.0,
        "sdf_save_sec": 0.0,
        "stage_by_dataset": {},
    }

    for dataset_name in dataset_names:
        predicted_ligand_transforms_path = None
        timings["stage_by_dataset"].setdefault(dataset_name, {})

        conf.use_sorted_batching = True
        conf.test_dataset_types = [dataset_name]
        test_dataset_docking = get_datasets(
            conf, splits=['test'],
            predicted_ligand_transforms_path=predicted_ligand_transforms_path,
            is_train_dataset=False,
            complex_collate_fn=complex_collate_fn,
            stage_num=1,
            n_preds_to_use=n_preds_to_use,
        )['test']
        logger.info({ds_name: len(ds) for ds_name, ds in test_dataset_docking.items()})
        test_dataset_docking = test_dataset_docking[dataset_name]

        for stage_idx in [0, 1, 2]:
            if pocket_centers_filename is not None and stage_idx == 0:
                predicted_ligand_transforms_path = str(pocket_centers_filename)
                logger.info(f'Skipping stage 1, using pocket centers from {pocket_centers_filename}')
                continue

            module = pipeline['docking'][min(stage_idx, len(pipeline['docking']) - 1)]
            model = module['model']
            model.to(device)
            logger.info(f"Stage {stage_idx + 1}, transforms path: {predicted_ligand_transforms_path}")

            test_dataset_docking.stage_num = stage_idx + 1

            stage_names = ['Translation (R³)', 'Rotation (SO(3))', 'Torsion (SO(2))']
            if progress_callback is not None:
                progress_callback('stage_start', f'stage{stage_idx + 1}', stage_names[stage_idx], None, None)

            if predicted_ligand_transforms_path is not None:
                use_predicted_tr_only = docking_modules[stage_idx]['dataset_kwargs'].get('use_predicted_tr_only', True)
                test_dataset_docking.dataset.use_predicted_tr_only = use_predicted_tr_only
                test_dataset_docking.reset_predicted_ligand_transforms(
                    predicted_ligand_transforms_path, n_preds_to_use)

            test_loader = get_dataloader_docking(test_dataset_docking)
            stage_start_time = time.time()
            solver_kwargs = {
                "record_history": False,
                "record_trajectory": False,
            }
            metrics = run_evaluation(
                test_loader, num_steps=num_steps, solver=euler, model=model,
                progress_callback=progress_callback, current_stage=f'stage{stage_idx + 1}',
                device=device,
                compute_torsion_angles_pred=compute_torsion_angles_pred,
                solver_kwargs=solver_kwargs,
            )
            stage_elapsed = time.time() - stage_start_time
            stage_key = f"stage{stage_idx + 1}_sec"
            timings[stage_key] = float(timings.get(stage_key, 0.0)) + float(stage_elapsed)
            timings["stage_by_dataset"][dataset_name][stage_key] = float(stage_elapsed)

            # Save stage results
            predicted_ligand_transforms_path = os.path.join(
                conf.inference_results_folder, run_name, f'stage{stage_idx+1}_{dataset_name}.npy')
            np.save(predicted_ligand_transforms_path, [metrics])
            logger.info(f"Saved stage {stage_idx+1} metrics to {predicted_ligand_transforms_path}")

            if progress_callback is not None:
                progress_callback('stage_done', f'stage{stage_idx + 1}', None, stage_elapsed, None)

            # Build per-stage output dict
            updated_metrics = construct_output_dict(metrics, test_dataset_docking.dataset)
            final_metrics_path = os.path.join(
                conf.inference_results_folder, run_name, f'{dataset_name}_final_preds_{stage_idx + 1}stage.npy')
            np.save(final_metrics_path, [updated_metrics])

    # Save all predictions to SDF (used by GNINA scoring and for inspection)
    if progress_callback is not None:
        progress_callback('stage_start', 'sdf_save', 'Saving predictions to SDF', None, None)
    sdf_start = time.time()

    # Merge all prediction stages and save merged predictions to SDF
    load_and_merge_all_stages(conf, run_name)
    save_all_to_sdf(conf, run_name, one_file=True, merge_stages=True)
    timings["sdf_save_sec"] = float(time.time() - sdf_start)
    if progress_callback is not None:
        progress_callback('stage_done', 'sdf_save', None, timings["sdf_save_sec"], None)

    timings_path = os.path.join(conf.inference_results_folder, run_name, "timings_pipeline.json")
    with open(timings_path, "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)
    logger.info(f"Saved pipeline timings to {timings_path}")
    return timings
