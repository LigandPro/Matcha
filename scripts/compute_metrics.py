from omegaconf import OmegaConf
from argparse import ArgumentParser

from matcha.utils.inference_utils import compute_metrics_all


if __name__ == '__main__':
    parser = ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-p", "--paths-config", dest="paths_config_filename",
                        required=True, help="config file with paths")
    parser.add_argument("-n", "--run_name", dest="inference_run_name",
                        required=True, help="inference run name")
    args = parser.parse_args()

    # Load main model config
    conf = OmegaConf.load(args.paths_config_filename)
    compute_metrics_all(conf, args.inference_run_name)
