"""


Example:
    uv run match -r receptor.pdb -l ligand.sdf -o out.sdf --checkpoints /path/to/pipeline --gpu 0
"""

import copy
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from rdkit import Chem
from rich.console import Console

from matcha.utils.esm_utils import compute_esm_embeddings, compute_sequences
from matcha.utils.inference_utils import (
    compute_fast_filters,
    run_inference_pipeline,
    save_best_pred_to_sdf,
)

console = Console()


DEFAULT_CONF = {
    "seed": 777,
    "tr_std": 5.0,
    "use_time": True,
    "dropout_rate": 0.0,
    "num_kernel_pos_encoder": 128,
    "objective": "ranking",
    "llm_emb_dim": 480,
    "use_all_chains": True,
    "pdbbind_split_test": "data/pdbbind_test.txt",
    "posebusters_split_test": "data/posebusters_v2.txt",
}


def _normalize_ligand(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower()
    if suffix == ".sdf":
        shutil.copyfile(src, dest)
        return
    if suffix in {".mol", ".mol2"}:
        mol = Chem.MolFromMolFile(str(src), removeHs=False, sanitize=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(src), removeHs=False, sanitize=False)
    else:
        raise typer.BadParameter(f"Unsupported ligand format: {src}")
    if mol is None:
        raise typer.BadParameter(f"RDKit failed to read ligand: {src}")
    writer = Chem.SDWriter(str(dest))
    writer.write(mol)
    writer.close()


def _prepare_singleton_dataset(
    receptor: Path, ligand: Path, dataset_dir: Path, uid: str
) -> None:
    sample_dir = dataset_dir / uid
    sample_dir.mkdir(parents=True, exist_ok=True)

    receptor_dest = sample_dir / f"{uid}_protein.pdb"
    ligand_dest = sample_dir / f"{uid}_ligand.sdf"

    if receptor.suffix.lower() != ".pdb":
        raise typer.BadParameter("Receptor must be a .pdb file.")
    shutil.copyfile(receptor, receptor_dest)
    _normalize_ligand(ligand, ligand_dest)


def _load_base_conf(user_config: Optional[Path]) -> OmegaConf:
    base = OmegaConf.create(DEFAULT_CONF)
    if user_config is None:
        return base
    if not user_config.exists():
        raise typer.BadParameter(f"Config file not found: {user_config}")
    user_conf = OmegaConf.load(user_config)
    return OmegaConf.merge(base, user_conf)


def _build_conf(base_conf: OmegaConf, workdir: Path, checkpoints: Path) -> OmegaConf:
    conf = OmegaConf.merge(
        base_conf,
        {
            "test_dataset_types": ["any_conf"],
            "cache_path": str(workdir / "cache"),
            "data_folder": str(workdir / "data"),
            "inference_results_folder": str(workdir / "runs"),
            "checkpoints_folder": str(checkpoints),
            "any_data_dir": str(workdir / "datasets" / "any_conf"),
        },
    )
    if not conf.get("llm_emb_dim"):
        conf.llm_emb_dim = DEFAULT_CONF["llm_emb_dim"]
    if not conf.get("use_all_chains"):
        conf.use_all_chains = DEFAULT_CONF["use_all_chains"]
    return conf


def _resolve_output_path(out: Path, run_name: str) -> Path:
    """
    Allow passing a directory (or '.') as output; if so, drop the pose as <run_name>.sdf inside it.
    If no extension is provided, default to .sdf.
    """
    out = out.expanduser()
    if out.exists() and out.is_dir():
        return out / f"{run_name}.sdf"
    # If user gives something like "./results/" that doesn't exist yet but endswith slash
    if str(out).endswith(os.sep):
        return out / f"{run_name}.sdf"
    if out.suffix == "":
        return out.with_suffix(".sdf")
    return out


def _ensure_checkpoints(path: Path) -> Path:
    """
    Make sure checkpoints exist locally; if missing, download from HuggingFace.
    """
    path = path.expanduser()
    stage1 = path / "pipeline" / "stage1" / "model.safetensors"
    if stage1.exists():
        return path

    console.print(f"[bold yellow][matcha][/bold yellow] checkpoints not found, downloading to {path} ...")
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="LigandPro/Matcha",
        allow_patterns=["pipeline/**"],
        local_dir=str(path),
        local_dir_use_symlinks=False,
    )
    if not stage1.exists():
        raise typer.BadParameter(f"Failed to download checkpoints into {path}")
    console.print(f"[bold green][matcha][/bold green] checkpoints ready at {path}")
    return path


def _print_usage_and_exit() -> None:
    usage = """
Matcha docking

Required:
  -r, --receptor PATH        Protein structure (.pdb)
  -l, --ligand PATH          Ligand (.sdf/.mol/.mol2/.pdb)
  -o, --out PATH             Output SDF path or directory (if dir, saves <run_name>.sdf)
      --checkpoints PATH     Folder containing pipeline checkpoints

Common:
      --run-name TEXT        Run name (folder name for intermediates)
      --n-samples INT        Samples per ligand (default 40)
  -g, --gpu INT              CUDA device index
      --overwrite            Overwrite existing run folder
      --keep-workdir         Keep temp data instead of deleting

Optional:
      --checkpoints PATH     Folder for pipeline checkpoints (auto-download if missing; default ./checkpoints)
      --run-name TEXT        Run name (folder for intermediates)
      --n-samples INT        Samples per ligand (default 40)
      --n-confs INT          Number of ligand conformers to generate (default min(10, n-samples))
  -g, --gpu INT              CUDA device index
      --overwrite            Overwrite existing run folder
      --keep-workdir         Keep temp data instead of deleting
      --log PATH             Tee console output into a log file

Examples:
  uv run matcha -r prot.pdb -l lig.sdf -o out.sdf --gpu 0
  uv run matcha -r prot.pdb -l lig.sdf -o ./poses/ --gpu 0 --run-name test --overwrite
"""
    console.print(usage.rstrip())
    sys.exit(1)


def run_match(
    receptor: Optional[Path] = typer.Option(None, "-r", "--receptor", help="Protein structure (.pdb)."),
    ligand: Optional[Path] = typer.Option(None, "-l", "--ligand", help="Ligand with 3D coords (.sdf/.mol/.mol2/.pdb)."),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output SDF path for the top-ranked pose."),
    checkpoints: Optional[Path] = typer.Option(None, "--checkpoints", help="Folder containing the Matcha 'pipeline' checkpoints (optional)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional base config to merge with defaults."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Scratch dir (default: ~/.cache/matcha_cli/<run-name>)."),
    run_name: str = typer.Option("matcha_cli_run", "--run-name", help="Name for this docking run."),
    n_samples: int = typer.Option(40, "--n-samples", help="Number of samples to generate per ligand."),
    n_confs: Optional[int] = typer.Option(None, "--n-confs", help="Number of ligand conformers to generate (default min(10, n-samples))."),
    gpu: Optional[int] = typer.Option(
        None, "--gpu", "-g", "-gpu", help="CUDA device index (sets CUDA_VISIBLE_DEVICES)."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Remove existing run folder if present."),
    keep_workdir: bool = typer.Option(False, "--keep-workdir", help="Keep working data instead of deleting it."),
    log: Optional[Path] = typer.Option(None, "--log", help="Path to log file (tee console output)."),
) -> None:
    if receptor is None or ligand is None or out is None:
        _print_usage_and_exit()

    log_file = None
    log_console = None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log, "w")
        log_console = Console(file=log_file, force_terminal=False, color_system=None)

    def _print(msg: str) -> None:
        console.print(msg)
        if log_console is not None:
            log_console.print(msg)

    banner = r"""
███╗   ███╗ █████╗ ████████╗ ██████╗██╗  ██╗ █████╗ 
████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║  ██║██╔══██╗
██╔████╔██║███████║   ██║   ██║     ███████║███████║
██║╚██╔╝██║██╔══██║   ██║   ██║     ██╔══██║██╔══██║
██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗██║  ██║██║  ██║
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝
"""
    _print(banner.rstrip("\n"))

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    checkpoints = checkpoints or Path("checkpoints")
    checkpoints = _ensure_checkpoints(checkpoints)

    base_workdir = workdir or Path.home() / ".cache" / "matcha_cli"
    run_workdir = base_workdir / run_name
    if run_workdir.exists():
        if overwrite:
            shutil.rmtree(run_workdir)
        else:
            raise typer.BadParameter(
                f"Working directory {run_workdir} already exists. Use --overwrite or change --run-name."
            )
    run_workdir.mkdir(parents=True, exist_ok=True)

    dataset_dir = run_workdir / "datasets" / "any_conf"
    _prepare_singleton_dataset(receptor, ligand, dataset_dir, run_name)

    base_conf = _load_base_conf(config)
    conf = _build_conf(base_conf, run_workdir, checkpoints)

    if n_confs is not None and n_confs < 1:
        raise typer.BadParameter("--n-confs must be >= 1")
    if n_confs is not None and n_samples > n_confs:
        n_samples = n_confs

    _print(f"[bold green][matcha][/bold green] workdir: {run_workdir}")
    _print(f"[bold green][matcha][/bold green] checkpoints: {checkpoints}")
    _print(f"[bold green][matcha][/bold green] samples per ligand: {n_samples}")
    if n_confs is not None:
        _print(f"[bold green][matcha][/bold green] ligand conformers: {n_confs}")
        conf.n_confs_override = int(n_confs)
    if gpu is not None:
        _print(f"[bold green][matcha][/bold green] using CUDA device #{gpu}")

    compute_sequences(conf)
    compute_esm_embeddings(conf)
    run_inference_pipeline(copy.deepcopy(conf), run_name, n_samples)
    compute_fast_filters(conf, run_name, n_samples)
    save_best_pred_to_sdf(conf, run_name)

    pred_sdf = (
        Path(conf.inference_results_folder)
        / run_name
        / "any_conf"
        / "sdf_predictions"
        / f"{run_name}.sdf"
    )
    if not pred_sdf.exists():
        raise typer.Exit(code=1)

    resolved_out = _resolve_output_path(out, run_name)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pred_sdf, resolved_out)
    _print(f"[bold green][matcha][/bold green] saved top-ranked pose to {resolved_out}")

    if keep_workdir:
        _print(f"[bold green][matcha][/bold green] keeping workdir at {run_workdir}")
    else:
        shutil.rmtree(run_workdir, ignore_errors=True)

    if log_file is not None:
        log_file.close()


def main() -> None:
    if len(sys.argv) == 1:
        _print_usage_and_exit()
    typer.run(run_match)


if __name__ == "__main__":
    main()
