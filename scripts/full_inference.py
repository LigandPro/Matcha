from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

from matcha.utils.inference_utils import (
    save_all_to_sdf,
    load_and_merge_all_stages,
)
from matcha.utils.log import get_logger
logger = get_logger(__name__)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False

    parser = ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-c", "--config", dest="config_filename",
                        required=True, help="config file with model arguments")
    parser.add_argument("-p", "--paths-config", dest="paths_config_filename",
                        required=True, help="config file with paths")
    parser.add_argument("-n", "--name", dest="inference_run_name",
                        required=True, help="name and the folder of the inference run")
    parser.add_argument("--merge-stages", dest="merge_stages",
                        required=False, help="merge stages", default=False, action='store_true')
    args = parser.parse_args()

    # Load main model config
    default_conf = OmegaConf.load('configs/default.yaml')
    conf = OmegaConf.load(args.config_filename)
    paths_conf = OmegaConf.load(args.paths_config_filename)
    conf = OmegaConf.merge(default_conf, conf, paths_conf)

    if args.merge_stages:
        load_and_merge_all_stages(conf, args.inference_run_name)
    
    save_all_to_sdf(conf, args.inference_run_name, one_file=True, merge_stages=args.merge_stages)
