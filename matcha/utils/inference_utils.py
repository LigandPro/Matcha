import os
from tqdm import tqdm
import numpy as np
import pandas as pd
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
from torch.utils.data import DataLoader

from matcha.utils.paths import get_dataset_path, get_ligand_path
from matcha.dataset.pdbbind import complex_collate_fn
from matcha.dataset.pdbbind_scoring import dummy_ranking_collate_fn
from matcha.models import MatchaModel
from matcha.models.scoring_model import MatchaScoringModel
from matcha.utils.datasets import get_datasets
from matcha.utils.inference import (
    euler, load_from_checkpoint, run_evaluation, scoring_inference)
from matcha.utils.metrics import (add_score_results, construct_output_dict,
                                  get_final_results_for_df, get_simple_metrics_df)
from matcha.utils.posebusters_utils import calc_posebusters
from matcha.utils.posebusters import get_posebusters_tests_updated
from matcha.utils.spyrmsd import compute_all_isomorphisms
from matcha.utils.preprocessing import read_molecule

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

confProDy(verbosity='none')

KEYS_VALID = ['not_too_far_away', 'no_internal_clash',
              'no_clashes', 'no_volume_clash', 'is_buried_fraction']


def get_data_for_buster(preds, dataset_data_dir, dataset_name):
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
            if orig_mol is None:
                print(f'Warning: Unable to read molecule for {uid}')
                continue

            # Remove hydrogens for consistency with predictions
            orig_mol_no_h = orig_mol
            try:
                orig_mol_no_h = RemoveAllHs(orig_mol, sanitize=True)
            except Exception:
                try:
                    orig_mol_no_h = RemoveAllHs(orig_mol, sanitize=False)
                except Exception:
                    print(f'Warning: Unable to remove hydrogens for {uid}, using as-is')

            true_pos = np.copy(orig_mol_no_h.GetConformer().GetPositions())
        except Exception as e:
            print(f'Error processing {uid}: {str(e)}')
            continue

        samples = pred_data['sample_metrics']
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


def compute_metrics_all(conf, inference_run_name):
    score_names_for_metrics = ['random', 'error_estimate_0', 'symm_rmsd']
    preds_path = os.path.join(
        conf.inference_results_folder, inference_run_name)

    for dataset_name in conf.test_dataset_types:
        print('---' * 30)

        # Load predictions
        preds_fname = os.path.join(
            preds_path, f'{dataset_name}_final_preds_fast_metrics.npy')
        final_preds_path = os.path.join(
            preds_path, f'{dataset_name}_final_preds_all_metrics.npy')
        preds = np.load(preds_fname, allow_pickle=True).item()
        if len(preds) == 0:
            print('No predictions for', inference_run_name, dataset_name)
            continue

        updated_metrics = {}
        dataset_data_dir = get_dataset_path(dataset_name, conf)
        data_for_buster = get_data_for_buster(
            preds, dataset_data_dir, dataset_name)

        # compute mol2isomorphisms for metrics computation
        mol2isomorphisms = {}
        for uid, uid_data in tqdm(data_for_buster.items(), desc='Computing isomorphisms'):
            mol = uid_data[0]['orig_mol']
            try:
                mol = RemoveAllHs(mol, sanitize=True)
            except Exception as e:
                mol = RemoveAllHs(mol, sanitize=False)
            mol2isomorphisms[uid] = compute_all_isomorphisms(mol)

        # compute metrics (without posebusters)
        results_df, _, updated_metrics = get_simple_metrics_df(
            data_for_buster, compute_symm_rmsd=True,
            mol2isomorphisms=mol2isomorphisms, score_names=score_names_for_metrics)

        # Save detailed information about skipped samples for debugging
        if len(updated_metrics) != len(data_for_buster):
            skipped_info = {
                'dataset': dataset_name,
                'inference_run': inference_run_name,
                'total_structures': len(data_for_buster),
                'processed_structures': len(updated_metrics),
                'timestamp': str(pd.Timestamp.now()),
            }
            skipped_path = os.path.join(preds_path, f'{dataset_name}_computation_summary.json')
            with open(skipped_path, 'w') as f:
                json.dump(skipped_info, f, indent=2)
            print(f'Saved computation summary to {skipped_path}')

        print('RMSD metrics for', dataset_name, inference_run_name)
        results_df.set_index('ranking', inplace=True)
        print(results_df.loc['error_estimate_0', [
              'SymRMSD < 2A', 'SymRMSD < 5A', 'tr_err < 1A', 'median SymRMSD', 'median tr_err']])

        # compute PoseBusters filters
        updated_metrics = get_posebusters_tests_updated(updated_metrics, dataset_name, dataset_data_dir=dataset_data_dir,
                                                        posebusters_config='redock')

        # compute metrics (with PoseBusters filters)
        rows_list, _ = get_final_results_for_df(updated_metrics, score_names=score_names_for_metrics,
                                                posebusters_filter=True, fast_filter=True)
        results_df = pd.DataFrame(rows_list)
        results_df.to_csv(os.path.join(
            preds_path, f'{dataset_name}_final_metrics.csv'), index=False)
        print('All metrics for', dataset_name, inference_run_name)
        results_df.set_index('ranking', inplace=True)
        print(results_df.loc['error_estimate_0_fast', ['SymRMSD < 2A', 'SymRMSD < 2A & PB valid',
              'SymRMSD < 5A', 'tr_err < 1A', 'median SymRMSD', 'median tr_err']])
        print(f'Saved posebusters results to {final_preds_path}')
        np.save(final_preds_path, [updated_metrics])


def save_best_pred_to_sdf(conf, inference_run_name):
    for dataset_name in conf.test_dataset_types:
        a = np.load(os.path.join(conf.inference_results_folder, inference_run_name,
                                 f'{dataset_name}_final_preds_fast_metrics.npy'), allow_pickle=True).item()

        save_path = os.path.join(
            conf.inference_results_folder, inference_run_name, dataset_name, 'sdf_predictions')
        os.makedirs(save_path, exist_ok=True)
        print(f'Saving predictions to {save_path}')
        for uid, sample_data in tqdm(a.items(), desc='Saving predictions'):
            if len(sample_data) == 0:
                continue

            orig_mol = sample_data['orig_mol']
            uid_real = uid.split('_mol')[0]

            samples = sample_data['sample_metrics']
            pb_passed_count = np.array(
                [sample.get('posebusters_filters_passed_count_fast', 0) for sample in samples])
            best_pb_count = max(pb_passed_count)
            samples = [sample for sample in samples
                       if sample.get('posebusters_filters_passed_count_fast', 0) == best_pb_count]
            scores = [sample['error_estimate_0'] for sample in samples]
            best_score_idx = np.argmin(scores)

            best_sample = samples[best_score_idx]
            pred_positions = best_sample['pred_pos']
            mol = copy.deepcopy(orig_mol)
            try:
                mol.GetConformer().SetPositions(pred_positions.astype(np.float64))
                writer = Chem.SDWriter(os.path.join(
                    save_path, f'{uid_real}.sdf'))
                writer.write(mol, confId=0)
            except Exception as e:
                continue


def calc_posebusters_for_data(data, lig_pos, orig_mol):
    lig_pos_for_posebusters = lig_pos
    lig_types_for_posebusters = data.ligand.x[:, 0] - 1
    pro_types_for_posebusters = data.protein.all_atom_names
    pro_pos_for_posebusters = data.protein.all_atom_pos
    lig_mol_for_posebusters = orig_mol
    names = data.name
    posebusters_results = calc_posebusters(lig_pos_for_posebusters, pro_pos_for_posebusters,
                                           lig_types_for_posebusters, pro_types_for_posebusters, names, lig_mol_for_posebusters)
    if posebusters_results is None:
        return None
    return np.array([posebusters_results[key] for key in KEYS_VALID if key in posebusters_results.keys()], dtype=object).transpose()


def compute_fast_filters(conf, inference_run_name, n_preds_to_use):
    all_datasets = get_datasets(conf, splits=['test'], return_separately=True,
                                predicted_ligand_transforms_path=None,
                                use_predicted_tr_only=False,
                                n_preds_to_use=n_preds_to_use,
                                complex_collate_fn=complex_collate_fn)
    test_datasets = all_datasets['test']
    for dataset_name, dataset in test_datasets.items():
        number_failed = 0
        file_name = f"{dataset_name}_final_preds.npy"
        file_name_save = f"{dataset_name}_final_preds_fast_metrics.npy"
        predicts = np.load(os.path.join(conf.inference_results_folder,
                           inference_run_name, file_name), allow_pickle=True).item()
        for data in tqdm(dataset.complexes, desc=f"Computing fast filters for {dataset_name}"):
            name = data.name.split('_conf')[0]
            if 'posebusters_filters_fast' in predicts[name]["sample_metrics"][0]:
                continue
            try:
                lig_pos = np.stack([predicts[name]["sample_metrics"][i]["pred_pos"]
                                   for i in range(len(predicts[name]["sample_metrics"]))])
            except Exception as e:
                print(f"Error in {name}")
                print(e)
                number_failed += 1
                continue
            posebusters_results = calc_posebusters_for_data(
                data, lig_pos, predicts[name]["orig_mol"])
            if posebusters_results is None:
                print(f"Fast filters computation failed for {name}")
                number_failed += 1
                continue
            for i, r in enumerate(posebusters_results):
                predicts[name]["sample_metrics"][i]["posebusters_filters_fast"] = r
                predicts[name]["sample_metrics"][i]["posebusters_filters_passed_count_fast"] = (
                    r[:4] == True).sum()

        print(f"Dataset {dataset_name} Number of failed: {number_failed}")
        np.save(os.path.join(conf.inference_results_folder,
                inference_run_name, file_name_save), predicts, allow_pickle=True)


def run_inference_pipeline(conf, run_name, n_preds_to_use, pocket_centers_filename=None, 
                           docking_batch_limit=15000, scoring_batch_size=4):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(conf.seed)

    # Load model
    model = MatchaModel(feature_dim=320, num_heads=8, num_transformer_blocks=12,
                        llm_emb_dim=conf.llm_emb_dim, use_time=conf.use_time,
                        dropout_rate=conf.dropout_rate, num_kernel_pos_encoder=conf.num_kernel_pos_encoder)
    print('Model parameters (M):', sum(p.numel()
          for p in model.parameters()) / 1e6)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_steps = 10
    print('Generate', n_preds_to_use, 'samples for each ligand')


    conf.batch_limit = docking_batch_limit
    num_workers = 8

    def get_dataloader_docking(dataset): return DataLoader(dataset, batch_size=1, shuffle=False,
                                                           collate_fn=dummy_ranking_collate_fn, num_workers=num_workers)
    def get_dataloader_scoring(dataset): return DataLoader(dataset, batch_size=scoring_batch_size, shuffle=False,
                                                           collate_fn=complex_collate_fn, num_workers=num_workers)
    dataset_names = conf.test_dataset_types
    print('DATASET NAMES:', dataset_names)

    pipeline = {
        'docking': [
            {
                'model_path': 'pipeline/stage1/',
                'dataset_kwargs': {},
            },
            {
                'model_path': 'pipeline/stage2/',
                'dataset_kwargs': {},
            },
            {
                'model_path': 'pipeline/stage3/',
                'dataset_kwargs': {'use_predicted_tr_only': False},
            },
        ],
        'scoring': {
            'model_path': 'pipeline/scoring/',
            'dataset_kwargs': {},
        }
    }

    # Save config to the run folder
    os.makedirs(os.path.join(
        conf.inference_results_folder, run_name), exist_ok=True)
    with open(os.path.join(conf.inference_results_folder, run_name, 'config.json'), 'w') as f:
        json.dump(pipeline, f)

    docking_modules = pipeline['docking']
    for module in docking_modules:
        model = MatchaModel(feature_dim=320, num_heads=8, num_transformer_blocks=12,
                            llm_emb_dim=conf.llm_emb_dim, use_time=conf.use_time,
                            dropout_rate=conf.dropout_rate, num_kernel_pos_encoder=conf.num_kernel_pos_encoder,)
        model = load_from_checkpoint(model, os.path.join(
            conf.checkpoints_folder, module['model_path']), strict=False)
        model.to(device)
        model.eval()
        module['model'] = model

    scoring_model = MatchaScoringModel(feature_dim=192, num_heads=4, num_transformer_blocks=6,
                                       llm_emb_dim=conf.llm_emb_dim, dropout_rate=conf.dropout_rate,
                                       objective='ranking')
    scoring_model = load_from_checkpoint(scoring_model, os.path.join(conf.checkpoints_folder,
                                                                     pipeline['scoring']['model_path']))
    scoring_model.to(device)
    scoring_model.eval()
    pipeline['scoring']['model'] = scoring_model

    print('Start inference pipeline', run_name)

    for dataset_name in dataset_names:
        predicted_ligand_transforms_path = None

        # # Load datasets
        conf.use_sorted_batching = True
        conf.test_dataset_types = [dataset_name]
        test_dataset_docking = get_datasets(conf, splits=['test'], return_separately=True,
                                            predicted_ligand_transforms_path=predicted_ligand_transforms_path,
                                            complex_collate_fn=complex_collate_fn,
                                            n_preds_to_use=n_preds_to_use,
                                            **docking_modules[0]['dataset_kwargs'])['test']
        print({ds_name: len(ds)
              for ds_name, ds in test_dataset_docking.items()})
        test_dataset_docking = test_dataset_docking[dataset_name]

        for stage_idx in [0, 1, 2]:
            if pocket_centers_filename is not None and stage_idx == 0:
                predicted_ligand_transforms_path = pocket_centers_filename
                print('Skipping docking stage 1')
                print(f'Using pocket centers from {pocket_centers_filename}')
                continue

            module = pipeline['docking'][min(
                stage_idx, len(pipeline['docking']) - 1)]
            model = module['model']
            model.to(device)

            print(f'Stage {stage_idx + 1}; predicted_ligand_transforms_path:',
                  predicted_ligand_transforms_path)
            if predicted_ligand_transforms_path is not None:
                use_predicted_tr_only = docking_modules[stage_idx]['dataset_kwargs'].get('use_predicted_tr_only', True)
                test_dataset_docking.dataset.use_predicted_tr_only = use_predicted_tr_only
                test_dataset_docking.reset_predicted_ligand_transforms(
                    predicted_ligand_transforms_path, n_preds_to_use)

            # Dataloaders
            test_loader = get_dataloader_docking(test_dataset_docking)
            metrics = run_evaluation(
                test_loader, num_steps=num_steps, solver=euler, model=model)

            # Save results
            predicted_ligand_transforms_path = os.path.join(
                conf.inference_results_folder, run_name, f'stage{stage_idx+1}_{dataset_name}.npy')
            np.save(predicted_ligand_transforms_path, [metrics])
            print(f'Saved metrics to {predicted_ligand_transforms_path}')

        # load dataset with predicted ligand transforms
        conf.use_sorted_batching = False
        test_dataset_scoring = get_datasets(conf, splits=['test'],
                                            return_separately=True,
                                            complex_collate_fn=complex_collate_fn,
                                            predicted_complex_positions_path=predicted_ligand_transforms_path,
                                            n_preds_to_use=n_preds_to_use,
                                            **pipeline['scoring']['dataset_kwargs'],
                                            )['test']
        test_dataset_scoring = test_dataset_scoring[dataset_name]
        print('Scoring', dataset_name, len(test_dataset_scoring))
        test_loader = get_dataloader_scoring(test_dataset_scoring)

        pred_scores = scoring_inference(
            loader=test_loader, model=scoring_model)
        metrics = add_score_results(
            metrics, pred_scores, score_name='error_estimate', n_samples=None)
        np.save(predicted_ligand_transforms_path, [metrics])

        updated_metrics = construct_output_dict(metrics, test_dataset_scoring)
        final_preds_path = os.path.join(
            conf.inference_results_folder, run_name, f'{dataset_name}_final_preds.npy')
        np.save(final_preds_path, [updated_metrics])
        print(f'Saved final predictions to {final_preds_path}')
        print('****' * 30)
        print()
