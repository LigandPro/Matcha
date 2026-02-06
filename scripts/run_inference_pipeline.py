import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from omegaconf import OmegaConf
from matcha.dataset.pdbbind import complex_collate_fn, dummy_ranking_collate_fn
from matcha.models import MatchaModel
from matcha.utils.datasets import get_datasets
from matcha.utils.inference import euler, load_from_checkpoint, run_evaluation
from matcha.utils.metrics import construct_output_dict
from matcha.utils.log import get_logger

logger = get_logger(__name__)


def main():
    parser = ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-c", "--config", dest="config_filename",
                        required=True, help="config file with model arguments")
    parser.add_argument("-p", "--paths-config", dest="paths_config_filename",
                        required=True, help="config file with paths")
    parser.add_argument("-n", "--name", dest="run_name",
                        required=True, help="name and the folder of the inference run")
    parser.add_argument("--n-samples", dest="n_samples",
                        required=False, help="number of samples to generate for each ligand", default=40, type=int)
    parser.add_argument("--n-confs", dest="n_confs",
                        required=False, help="number of ligand conformers to generate with RDKit", default=None, type=int)
    args = parser.parse_args()

    # Load main model config
    default_conf = OmegaConf.load("configs/default.yaml")
    conf = OmegaConf.load(args.config_filename)
    paths_conf = OmegaConf.load(args.paths_config_filename)
    conf = OmegaConf.merge(default_conf, conf, paths_conf)

    conf.ligand_mask_ratio = 0.
    conf.protein_mask_ratio = 0.
    conf.std_protein_pos = 0
    conf.std_lig_pos = 0
    conf.esm_emb_noise_std = 0
    conf.randomize_bond_neighbors = False

    if args.n_confs is not None:
        conf.n_confs_override = args.n_confs

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(conf.seed)

    run_name = args.run_name

    # Load model
    model = MatchaModel(conf=conf)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    solver = euler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_steps = 10
    n_preds_to_use = args.n_samples
    logger.info(f"Inference config: n_preds_to_use={n_preds_to_use}, num_steps={num_steps}")
    num_workers = 32

    def get_dataloader_docking(dataset): return DataLoader(dataset, batch_size=1, shuffle=False,
                                                           prefetch_factor=2,
                                                           collate_fn=dummy_ranking_collate_fn, num_workers=num_workers)

    dataset_names = conf.test_dataset_types
    logger.info(f"Dataset names: {dataset_names}")

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

    # Save config to the run folder
    os.makedirs(os.path.join(
        conf.inference_results_folder, run_name), exist_ok=True)
    with open(os.path.join(conf.inference_results_folder, run_name, 'config.json'), 'w') as f:
        json.dump(pipeline, f)

    docking_modules = pipeline['docking']
    for module in docking_modules:
        model = MatchaModel(**module['model_kwargs'], conf=conf)
        model = load_from_checkpoint(model, os.path.join(
            conf.checkpoints_folder, module['model_path']))
        model.to(device)
        model.eval()
        module['model'] = model

    logger.info(f"Starting inference pipeline: {run_name}")

    for dataset_name in dataset_names:
        predicted_ligand_transforms_path = None

        # # Load datasets
        conf.use_sorted_batching = True
        conf.test_dataset_types = [dataset_name]
        test_dataset_docking = get_datasets(conf, splits=['test'],
                                            predicted_ligand_transforms_path=predicted_ligand_transforms_path,
                                            is_train_dataset=False,
                                            complex_collate_fn=complex_collate_fn,
                                            stage_num=1,
                                            **module['dataset_kwargs'],
                                            )['test']

        logger.info({ds_name: len(ds)
                for ds_name, ds in test_dataset_docking.items()})
        test_dataset_docking = test_dataset_docking[dataset_name]

        # for stage_idx in [0, 1, 2]:
        for stage_idx in [0, 1, 2]:
            module = pipeline['docking'][min(
                stage_idx, len(pipeline['docking']) - 1)]
            model = module['model']
            model.to(device)
            solver = euler
            logger.info(f"Stage {stage_idx + 1}, transforms path: {predicted_ligand_transforms_path}")

            test_dataset_docking.stage_num = stage_idx + 1

            if predicted_ligand_transforms_path is not None:
                use_predicted_tr_only = docking_modules[stage_idx]['dataset_kwargs'].get('use_predicted_tr_only', True)
                test_dataset_docking.dataset.use_predicted_tr_only = use_predicted_tr_only
                test_dataset_docking.reset_predicted_ligand_transforms(
                    predicted_ligand_transforms_path, n_preds_to_use)

            # Dataloaders
            test_loader = get_dataloader_docking(test_dataset_docking)

            # In case of using true translations
            # if stage_idx == 0:
            #     predicted_ligand_transforms_path = path_to_true_translations
            #     continue

            metrics = run_evaluation(test_loader, num_steps=num_steps, solver=solver, model=model)

            # Save results
            predicted_ligand_transforms_path = os.path.join(
                conf.inference_results_folder, run_name, f'stage{stage_idx+1}_{dataset_name}.npy')
            np.save(predicted_ligand_transforms_path, [metrics])
            logger.info(f"Saved metrics to {predicted_ligand_transforms_path}")

            np.save(predicted_ligand_transforms_path, [metrics])

            updated_metrics = construct_output_dict(
                metrics, test_dataset_docking.dataset)
            final_metrics_path = os.path.join(
                conf.inference_results_folder, run_name, f'{dataset_name}_final_preds_{stage_idx + 1}stage.npy')
            np.save(final_metrics_path, [updated_metrics])

        final_metrics_path = os.path.join(
            conf.inference_results_folder, run_name, f'{dataset_name}_final_preds.npy')
        np.save(final_metrics_path, [updated_metrics])



if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
