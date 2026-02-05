from argparse import ArgumentParser
from omegaconf import OmegaConf
from matcha.utils.inference_utils import compute_fast_filters_from_sdf


if __name__ == "__main__":
    parser = ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-c", "--config", dest="config_filename",
                        required=True, help="config file with model arguments")
    parser.add_argument("-p", "--paths-config", dest="paths_config_filename",
                        required=True, help="config file with paths")
    parser.add_argument("-n", "--run_name", dest="inference_run_name",
                        required=True, help="inference run name")
    parser.add_argument("--n_samples", dest="n_preds_to_use",
                        required=False, help="number of samples to generate for each ligand", default=40)
    args = parser.parse_args()

    default_conf = OmegaConf.load('configs/default.yaml')
    conf = OmegaConf.load(args.config_filename)
    paths_conf = OmegaConf.load(args.paths_config_filename)
    conf = OmegaConf.merge(default_conf, conf, paths_conf)

    compute_fast_filters_from_sdf(conf, args.inference_run_name,
                                  sdf_type='minimized',
                                  n_preds_to_use=int(args.n_preds_to_use))
