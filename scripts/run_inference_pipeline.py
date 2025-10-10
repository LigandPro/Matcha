
from argparse import ArgumentParser
from omegaconf import OmegaConf

from matcha.utils.inference_utils import run_inference_pipeline


if __name__ == "__main__":

    parser = ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-c", "--config", dest="config_filename",
                        required=True, help="config file with model arguments")
    parser.add_argument("-p", "--paths-config", dest="paths_config_filename",
                        required=True, help="config file with paths")
    parser.add_argument("-n", "--name", dest="run_name",
                        required=True, help="name and the folder of the inference run")
    parser.add_argument("--n_samples", dest="n_preds_to_use",
                        required=False, help="number of samples to generate for each ligand", default=40)
    args = parser.parse_args()

    # Load main model config
    conf = OmegaConf.load(args.config_filename)
    paths_conf = OmegaConf.load(args.paths_config_filename)
    conf = OmegaConf.merge(conf, paths_conf)
    run_inference_pipeline(conf, args.run_name, int(args.n_preds_to_use))
