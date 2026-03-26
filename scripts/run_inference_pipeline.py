import os
import json
import time
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
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="optional explicit dataset override")
    parser.add_argument("--num-steps", dest="num_steps", default=10, type=int,
                        help="number of Euler integration steps")
    parser.add_argument("--num-workers", dest="num_workers", default=32, type=int,
                        help="number of dataloader workers")
    parser.add_argument("--batch-limit", dest="batch_limit", default=None, type=int,
                        help="optional override for sorted batching token budget")
    parser.add_argument("--max-batches", dest="max_batches", default=None, type=int,
                        help="optional limit for benchmarking/debugging")
    parser.add_argument("--tf32", choices=["on", "off"], default="off",
                        help="toggle TF32 matmul/cudnn kernels")
    parser.add_argument("--benchmark-json", dest="benchmark_json", default=None,
                        help="optional path to write stage timing summary as JSON")
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
    if args.batch_limit is not None:
        conf.batch_limit = args.batch_limit

    allow_tf32 = args.tf32 == "on"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.allow_tf32 = allow_tf32

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(conf.seed)

    run_name = args.run_name

    # Load model
    model = MatchaModel(conf=conf)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    solver = euler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_steps = args.num_steps
    n_preds_to_use = args.n_samples
    num_workers = args.num_workers
    logger.info(
        "Inference config: "
        f"n_preds_to_use={n_preds_to_use}, "
        f"num_steps={num_steps}, "
        f"num_workers={num_workers}, "
        f"batch_limit={conf.batch_limit}, "
        f"max_batches={args.max_batches}, "
        f"tf32={allow_tf32}"
    )

    def get_dataloader_docking(dataset): return DataLoader(dataset, batch_size=1, shuffle=False,
                                                           prefetch_factor=2,
                                                           collate_fn=dummy_ranking_collate_fn, num_workers=num_workers)

    dataset_names = args.datasets or conf.test_dataset_types
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

    run_dir = os.path.join(conf.inference_results_folder, run_name)
    benchmark_summary = []

    # Save config to the run folder
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
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

        dataset_build_start_time = time.perf_counter()
        conf.use_sorted_batching = True
        conf.test_dataset_types = [dataset_name]
        test_dataset_docking = get_datasets(conf, splits=['test'],
                                            predicted_ligand_transforms_path=predicted_ligand_transforms_path,
                                            is_train_dataset=False,
                                            complex_collate_fn=complex_collate_fn,
                                            stage_num=1,
                                            **docking_modules[0]['dataset_kwargs'],
                                            )['test']
        dataset_build_seconds = time.perf_counter() - dataset_build_start_time

        logger.info({ds_name: len(ds)
                for ds_name, ds in test_dataset_docking.items()})
        test_dataset_docking = test_dataset_docking[dataset_name]

        # for stage_idx in [0, 1, 2]:
        for stage_idx in [0, 1, 2]:
            module = pipeline['docking'][min(
                stage_idx, len(pipeline['docking']) - 1)]
            model = module['model']
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
            stage_batch_timings = []
            stage_start_time = time.perf_counter()
            metrics = run_evaluation(
                test_loader,
                num_steps=num_steps,
                solver=solver,
                model=model,
                max_batches=args.max_batches,
                batch_timings=stage_batch_timings,
            )
            stage_wall_seconds = time.perf_counter() - stage_start_time

            benchmark_summary.append({
                "dataset": dataset_name,
                "stage": stage_idx + 1,
                "dataset_build_seconds": dataset_build_seconds,
                "stage_wall_seconds": stage_wall_seconds,
                "num_batches": len(stage_batch_timings),
                "mean_batch_seconds": (
                    float(np.mean(stage_batch_timings)) if stage_batch_timings else 0.0
                ),
                "total_batch_seconds": float(sum(stage_batch_timings)),
                "num_predictions": n_preds_to_use,
                "num_steps": num_steps,
                "num_workers": num_workers,
                "batch_limit": conf.batch_limit,
                "tf32": allow_tf32,
                "max_batches": args.max_batches,
            })

            # Save results
            predicted_ligand_transforms_path = os.path.join(
                conf.inference_results_folder, run_name, f'stage{stage_idx+1}_{dataset_name}.npy')
            np.save(predicted_ligand_transforms_path, [metrics])
            logger.info(f"Saved metrics to {predicted_ligand_transforms_path}")

            updated_metrics = construct_output_dict(
                metrics, test_dataset_docking.dataset)
            final_metrics_path = os.path.join(
                conf.inference_results_folder, run_name, f'{dataset_name}_final_preds_{stage_idx + 1}stage.npy')
            np.save(final_metrics_path, [updated_metrics])

        final_metrics_path = os.path.join(
            conf.inference_results_folder, run_name, f'{dataset_name}_final_preds.npy')
        np.save(final_metrics_path, [updated_metrics])

    if args.benchmark_json is not None:
        with open(args.benchmark_json, "w") as f:
            json.dump(benchmark_summary, f, indent=2)
        logger.info(f"Saved benchmark summary to {args.benchmark_json}")



if __name__ == "__main__":
    main()
