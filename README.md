# Matcha: Multi-Stage Riemannian Flow Matching for Accurate and Physically Valid Molecular Docking

This is an official implementation of the paper [Matcha: Multi-Stage Riemannian Flow Matching for Accurate and Physically Valid Molecular Docking](https://arxiv.org/abs/2510.14586).

## Overview

Matcha, is a molecular docking pipeline that combines multi-stage flow matching with learned scoring and physical validity filtering. Our approach consists of three sequential stages applied consecutively
to progressively refine docking predictions, each implemented as a flow matching model operating on appropriate geometric spaces (R^3, SO(3), and SO(2)).
We enhance the prediction quality through a dedicated scoring model and apply
unsupervised physical validity filters to eliminate unrealistic poses. 
![pipeline](data/img/matcha_pipeline.png)
![architecture](data/img/matcha_architecture.png)

Compared to various approaches, Matcha demonstrates superior performance on Astex
and PDBBind test sets in terms of docking success rate and physical plausibility.
Moreover, our method works approximately 25× faster than modern large-scale
co-folding models.

<img src="data/img/time.png" alt="results" width="500"/>

## Content
- [Installation](#install)
- [Datasets](#datasets)
  - [Existing datasets](#exist_datasets)
  - [Adding new dataset](#new_datasets)
- [Preparing the config file](#config)
- [Running inference with one script](#inference)
- [Running inference step-by-step](#inference_steps)
  - [Preprocessing](#preproc)
  - [Inference](#inf)
  - [Metrics computation](#metrics)
- [Benchmarking and pocket-aligned RMSD computation](#benchmarking)
- [License](#license)
- [Citation](#citation)


## Installation <a name="install"></a>
To install the `matcha` package, do the following:

```bash
cd matcha
pip install -e .
```

### uv install and cli

```bash
# Install with uv
uv sync

# Basic usage - blind docking on entire protein (recommended)
uv run matcha \
  -r protein.pdb \
  -l ligand.sdf \
  -o results/ \
  --run-name my_docking \
  --n-samples 20 \
  --gpu 0

# With autobox - focus search around reference ligand (experimental)
# Note: This crops the protein and may not work as well as blind docking
# TODO: NOT WORK
uv run matcha \
  -r protein.pdb \
  -l ligand.sdf \
  -o results/ \
  --autobox-ligand reference_ligand.sdf \
  --autobox-add 4 \
  --run-name focused_docking \
  --n-samples 20 \
  --gpu 0

# Manual box specification
# TODO: NOT WORK
uv run matcha \
  -r protein.pdb \
  -l ligand.sdf \
  -o results/ \
  --center-x 10 --center-y 20 --center-z 30 \
  --size-x 20 --size-y 20 --size-z 20 \
  --run-name manual_box \
  --gpu 0
```

**Important notes:**
- **Blind docking (no box)** is the native Matcha workflow and recommended for best results
- **Autobox/manual box** features are experimental extensions, not part of original Matcha
- Checkpoints auto-download from HuggingFace if not specified
- Default `n_samples=20`, seed=42
- Workdir is preserved by default (use `--no-keep-workdir` to clean up)

## Datasets <a name="datasets"></a>
### Existing datasets <a name="exist_datasets"></a>
If you want to make predictions for existing test datasets, you need to download them first.
Astex and PoseBusters datasets can be downloaded [here](https://zenodo.org/records/8278563).
PDBBind_processed can be found [here](https://zenodo.org/records/6408497).
DockGen can be downloaded from [here](https://zenodo.org/records/10656052).

### Adding new dataset <a name="new_datasets"></a>
If you need to predict some new data, you need to construct a dataset folder with the following structure:
```
- dataset_path
    - uid1
        - f'{uid1}_protein.pdb' - protein structure file
        - f'{uid1}_ligand.sdf' - ligand sdf file (can be an arbitrary conformation or a true position: we compute random conformers anyway)
    - uid2
        - f'{uid2}_protein.pdb'
        - f'{uid2}_ligand.sdf' 
    - ...
```
## Preparing the config file <a name="config"></a>
1. You need to provide paths to the required datasets. Edit the `configs/paths/paths.yaml` file: add `posebusters_data_dir`, `astex_data_dir`, `pdbbind_data_dir`, `dockgen_data_dir`. For a new dataset provide its path in `any_data_dir`. 

If you do not need all datasets, comment unnecessary datasets in the config's `test_dataset_types`, keeping only those that are needed.

2. Provide data paths to keep intermediate and final data:
- `cache_path`: path where intermediate dataset cache files will be stored
- `data_folder`: path where intermediate files will be stored (eg. ESM embeddings)
- `inference_results_folder`: <path_to_inference_results>

3. Download checkpoints for Matcha for [here](https://huggingface.co/LigandPro/Matcha). You need to download the `pipeline` folder. Provide the path to the folder where you store them (folder that contains `pipeline`) in the `checkpoints_folder` in paths.yaml.


## Running inference with one script <a name="inference"></a>
The script to compute all preprocessing and inference scripts in one. 
Provide `--compute_final_metrics` if your dataset has true ligand positions in f'{uid_i}_ligand.sdf', so we can compute RMSD metrics and PoseBusters filters.
Argument `-n inference_folder_name` is a name of a folder where to store inference results for dataset.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/full_inference.py -c configs/base.yaml -p configs/paths/paths.yaml -n inference_folder_name --n_samples 40 --compute_final_metrics
```

This script will provide a step-by-step computation of protein ESM embeddings, docking predictions, physically-aware unsupervised post-filtration, scoring and saving predictions to sdf.


## Running inference step-by-step <a name="inference_steps"></a>
### Preprocessing <a name="preproc"></a>
Run scripts to compute protein sequences anf ESM embeddings for all test datasets.

```bash
python scripts/prepare_esm_sequences.py -p configs/paths/paths.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/compute_esm_embeddings.py -p configs/paths/paths.yaml
```

### Inference <a name="inf"></a>

To run inference:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_inference_pipeline.py -c configs/base.yaml -p configs/paths/paths.yaml -n inference_folder_name --n_samples 40
```

Then run `posebusters_unsupervised.py` to do fast unsupervised post-filtration and `save_preds_for_npbench.py` to get top-1 ranked pose. The top-ranked poses for each complex are stored in the `inference_results_folder/inference_folder_name/sdf_predictions`.
You need to provide the same `--n_samples`, as for the `run_inference_pipeline.py`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/posebusters_unsupervised.py -c configs/base.yaml -p configs/paths/paths.yaml -n inference_folder_name --n_samples 40
python scripts/save_preds_for_npbench.py -p configs/paths/paths.yaml -n inference_folder_name
```

### Metrics computation <a name="metrics"></a>

Optionally, you can compute sample-level and dataset-level metrics (such as symmetry-corrected RMSD and passing PoseBusters filters) having your predictions.
Individual metrics for each sample are stored in the folder with inference results in file `f'{dataset_name}_final_preds_all_metrics.npy'`.

Dataset-level statistics are stored in the folder with inference results in file `f'{dataset_name}_final_metrics.csv'`.
This can be done with the command:

```bash
python scripts/compute_metrics.py -p configs/paths/paths.yaml -n inference_folder_name
```

## Benchmarking and pocket-aligned RMSD computation <a name="benchmarking"></a>
If you need to compute metrics for other docking tool predictions, you firtsly need to construct a folder with method predictions with a certain structure. Docking methods can be divided into those that modify the protein structure (eg. co-folding methods, FlowDock, NeuralPlexer) and those that use the reference protein structure (eg. Matcha, DiffDock, SMINA). We have a separate flag `has_pred_proteins` to differentiate these cases. When `has_pred_proteins=False`, the folder needs to contain only ligands, while when `has_pred_proteins=True`, you need to store both proteins and ligands.

The structure of the folder with predictions:

```
- method_name
  - dataset_name
    - uid1
      - conf_0
        - 'prot.pdb' - protein structure file (optional, needed when has_pred_proteins=True)
        - 'lig_0.sdf' - ligand sdf file
    - uid2
      - conf_0
        - 'prot.pdb' - protein structure file (optional, needed when has_pred_proteins=True)
        - 'lig_0.sdf' - ligand sdf file
    - ...
```

Then, you need to run the script `compute_aligned_rmsd.py` to align protein pockets and to compute RMSD and posebusters metrics.
It has two types of alignment ('base' and 'pocket'), among which we prefer 'base' option because it does not produce overoptimistic results (see Appendix D in the original paper for the discussion on this topic).
You need to set `methods_data` dict inside the script, providing method names and their `has_pred_proteins` flags.
Also, you need to provide the path where you store initial method predictions (`--init-preds-path`) as well as dataset_names you need to evaluate (`dataset_names`).
Then, run the script:

```bash
python scripts/compute_aligned_rmsd.py -p configs/paths/paths.yaml -a base --init-preds-path <path_to_initial_preds>
```


## License <a name="license"></a>
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## Citation <a name="citation"></a>

If you use Matcha in your work, please cite our paper:

```
@misc{frolova2025matchamultistageriemannianflow,
      title={Matcha: Multi-Stage Riemannian Flow Matching for Accurate and Physically Valid Molecular Docking}, 
      author={Daria Frolova and Talgat Daulbaev and Egor Sevryugov and Sergei A. Nikolenko and Dmitry N. Ivankov and Ivan Oseledets and Marina A. Pak},
      year={2025},
      eprint={2510.14586},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.14586}, 
}
```
