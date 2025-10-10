import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf
import copy

from matcha.utils.esm_utils import compute_sequences, compute_esm_embeddings
from matcha.utils.inference_utils import (run_inference_pipeline,
                                          compute_fast_filters,
                                          save_best_pred_to_sdf,
                                          compute_metrics_all)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False

    parser = ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-c", "--config", dest="config_filename",
                        required=True, help="config file with model arguments")
    parser.add_argument("-p", "--paths-config", dest="paths_config_filename",
                        required=True, help="config file with paths")
    parser.add_argument("-n", "--name", dest="inference_run_name",
                        required=True, help="name and the folder of the inference run")
    parser.add_argument("--n_samples", dest="n_preds_to_use",
                        required=False, help="number of samples to generate for each ligand", default=40)
    parser.add_argument("--compute_final_metrics", dest="compute_final_metrics",
                        required=False, help="compute final metrics", default=False, action='store_true')
    parser.add_argument("--no_compute_esm_embeddings", dest="no_compute_esm_embeddings",
                        required=False, help="compute ESM embeddings", default=False, action='store_true')
    args = parser.parse_args()

    # Load main model config
    conf = OmegaConf.load(args.config_filename)
    paths_conf = OmegaConf.load(args.paths_config_filename)
    conf = OmegaConf.merge(conf, paths_conf)
    n_preds_to_use = int(args.n_preds_to_use)
    if args.no_compute_esm_embeddings:
        print('Assuming ESM embeddings are already computed')
    else:
        compute_sequences(conf)
        compute_esm_embeddings(conf)
    run_inference_pipeline(copy.deepcopy(
        conf), args.inference_run_name, n_preds_to_use)
    compute_fast_filters(conf, args.inference_run_name, n_preds_to_use)
    save_best_pred_to_sdf(conf, args.inference_run_name)
    if args.compute_final_metrics:
        compute_metrics_all(conf, args.inference_run_name)
