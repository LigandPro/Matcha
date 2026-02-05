from argparse import ArgumentParser
from omegaconf import OmegaConf

from matcha.utils.esm_utils import compute_esm_embeddings


if __name__ == "__main__":

    parser = ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-p", "--paths-config", dest="paths_config_filename",
                        required=True, help="config file with paths")
    args = parser.parse_args()

    # Load main model config
    conf = OmegaConf.load(args.paths_config_filename)
    compute_esm_embeddings(conf)
    