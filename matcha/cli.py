"""
Example:
    uv run matcha -r receptor.pdb -l ligand.sdf -o out.sdf --gpu 0
"""

from __future__ import annotations

import copy
import errno
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple, List

import numpy as np
from rdkit import Chem
import typer
from rich.console import Console

from matcha.utils.log import get_logger
from matcha.utils.multigpu import _remove_tree_if_exists

console = Console()
logger = get_logger(__name__)


DEFAULT_CONF = {
    "seed": 777,
    "tr_std": 5.0,
    "use_time": True,
    "dropout_rate": 0.0,
    "num_kernel_pos_encoder": 128,
    "llm_emb_dim": 480,
    "feature_dim": 320,
    "num_heads": 8,
    "num_transformer_blocks": 12,
    "predict_torsion_angles": True,
    "stage_num": 1,
    "use_all_chains": True,
    # Inference overrides (disable noise/masking)
    "ligand_mask_ratio": 0.0,
    "protein_mask_ratio": 0.0,
    "std_protein_pos": 0.0,
    "std_lig_pos": 0.0,
    "esm_emb_noise_std": 0.0,
    "randomize_bond_neighbors": False,
    "batch_limit": 15000,
    "pdbbind_split_test": "data/pdbbind_test.txt",
    "posebusters_split_test": "data/posebusters_v2.txt",
}


def _format_runtime(seconds: float) -> str:
    secs = int(round(seconds))
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, rem = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{rem}s")
    return " ".join(parts)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _normalize_ligand(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower()
    mol = None
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(src), removeHs=False, sanitize=False)
        for candidate in supplier:
            if candidate is not None:
                mol = candidate
                break
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(src), sanitize=False)
    elif suffix == ".mol":
        mol = Chem.MolFromMolFile(str(src), removeHs=False, sanitize=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(src), removeHs=False, sanitize=False)
    else:
        raise typer.BadParameter(f"Unsupported ligand format: {src}")
    if mol is None:
        raise typer.BadParameter(f"RDKit failed to read ligand: {src}")

    writer = Chem.SDWriter(str(dest))
    writer.SetKekulize(False)
    for cid in range(mol.GetNumConformers()):
        writer.write(mol, confId=cid)
    writer.close()


def _prepare_singleton_dataset(
    receptor: Path,
    ligand: Path,
    dataset_dir: Path,
    uid: str,
) -> None:
    sample_dir = dataset_dir / uid
    _remove_tree_if_exists(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    receptor_dest = sample_dir / f"{uid}_protein.pdb"
    ligand_dest = sample_dir / f"{uid}_ligand.sdf"

    if receptor.suffix.lower() != ".pdb":
        raise typer.BadParameter("Receptor must be a .pdb file.")
    shutil.copyfile(receptor, receptor_dest)
    _normalize_ligand(ligand, ligand_dest)


def _split_multi_sdf(sdf_path: Path) -> List[Tuple[str, Any]]:
    molecules = []
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    for idx, mol in enumerate(suppl):
        if mol is None:
            console.print(f"[yellow]Warning: Failed to read molecule {idx} from {sdf_path.name}[/yellow]")
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{idx}"
        name = name.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
        if not name:
            name = f"mol_{idx}"
        molecules.append((name, mol))
    return molecules


def _split_ligand_file(ligand_file: Path) -> List[Tuple[str, Any]]:
    suffix = ligand_file.suffix.lower()
    if suffix == ".sdf":
        return _split_multi_sdf(ligand_file)
    if suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(ligand_file), sanitize=False)
        if mol is None:
            console.print(f"[yellow]Warning: Failed to read ligand file {ligand_file.name}[/yellow]")
            return []
        name = ligand_file.stem
        return [(name, mol)]
    raise ValueError(f"Unsupported ligand file format: {ligand_file.suffix}")


def _prepare_batch_dataset(
    protein: Path,
    molecules: List[Tuple[str, Any]],
    dataset_dir: Path,
) -> List[str]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    uids = []
    uid_counts: dict[str, int] = {}
    protein_real = protein.resolve()
    for name, mol in molecules:
        base_uid = name
        seen = uid_counts.get(base_uid, 0)
        uid_counts[base_uid] = seen + 1
        uid = base_uid if seen == 0 else f"{base_uid}_{seen}"
        sample_dir = dataset_dir / uid
        _remove_tree_if_exists(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        # Avoid copying the same receptor hundreds of times in batch mode.
        # A symlink keeps a single underlying file and enables downstream caching.
        receptor_dest = sample_dir / f"{uid}_protein.pdb"
        if receptor_dest.exists() or receptor_dest.is_symlink():
            receptor_dest.unlink()
        try:
            os.symlink(str(protein_real), str(receptor_dest))
        except OSError:
            shutil.copyfile(protein_real, receptor_dest)
        writer = Chem.SDWriter(str(sample_dir / f"{uid}_ligand.sdf"))
        writer.SetKekulize(False)
        writer.write(mol)
        writer.close()
        uids.append(uid)
    return uids


def _create_pocket_centers_file(
    pocket_center: Tuple[float, float, float],
    molecule_uids: List[str],
    n_samples: int,
    output_path: Path,
) -> None:
    pocket_centers = {}
    ligand_center = np.array(pocket_center, dtype=np.float32)
    protein_center = np.zeros(3)
    for uid in molecule_uids:
        for conf_idx in range(n_samples):
            pocket_centers[f"{uid}_mol0_conf{conf_idx}"] = [{"tr_pred_init": ligand_center, "full_protein_center": protein_center}]
    np.save(output_path, [pocket_centers])


def _load_base_conf(user_config: Optional[Path]) -> "OmegaConf":
    base = OmegaConf.create(DEFAULT_CONF)
    if user_config is None:
        return base
    if not user_config.exists():
        raise typer.BadParameter(f"Config file not found: {user_config}")
    user_conf = OmegaConf.load(user_config)
    return OmegaConf.merge(base, user_conf)


def _build_conf(base_conf: "OmegaConf", workdir: Path, checkpoints: Path) -> "OmegaConf":
    conf = OmegaConf.merge(
        base_conf,
        {
            "test_dataset_types": ["any_conf"],
            "cache_path": str(workdir / "cache"),
            "data_folder": str(workdir / "data"),
            "inference_results_folder": str(workdir / "runs"),
            "checkpoints_folder": str(checkpoints),
            "results_folder": str(checkpoints),
            "any_data_dir": str(workdir / "datasets" / "any_conf"),
        },
    )
    if conf.get("llm_emb_dim") is None:
        conf.llm_emb_dim = DEFAULT_CONF["llm_emb_dim"]
    if conf.get("use_all_chains") is None:
        conf.use_all_chains = DEFAULT_CONF["use_all_chains"]
    return conf


def _ensure_checkpoints(path: Path) -> Path:
    path = path.expanduser()
    stage1 = path / "matcha_pipeline" / "stage1" / "model.safetensors"
    if stage1.exists():
        return path
    console.print(f"[bold yellow][matcha][/bold yellow] checkpoints not found, downloading to {path} ...")
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="LigandPro/Matcha", local_dir=str(path), local_dir_use_symlinks=False)
    if not stage1.exists():
        raise typer.BadParameter(f"Failed to download checkpoints into {path}")
    console.print(f"[bold green][matcha][/bold green] checkpoints ready at {path}")
    return path


def _autobox_from_ligand(ligand: Path) -> Tuple[float, float, float]:
    suffix = ligand.suffix.lower()
    readers = {
        ".mol2": lambda: Chem.MolFromMol2File(str(ligand), sanitize=False),
        ".mol": lambda: Chem.MolFromMolFile(str(ligand), removeHs=False, sanitize=False),
        ".pdb": lambda: Chem.MolFromPDBFile(str(ligand), removeHs=False, sanitize=False),
    }
    mol = readers[suffix]() if suffix in readers else None
    if mol is None:
        suppl = Chem.SDMolSupplier(str(ligand), removeHs=False, sanitize=False)
        mol = suppl[0] if len(suppl) > 0 else None
    if mol is None or mol.GetNumConformers() == 0:
        raise typer.BadParameter(f"Failed to read ligand for autobox: {ligand}")
    mol = Chem.RemoveAllHs(mol, sanitize=False)
    return tuple(mol.GetConformer().GetPositions().mean(axis=0).tolist())


def _center_from_box_json(path: Path) -> Tuple[float, float, float]:
    """Extract a docking box center from a JSON file.

    Supports common formats, e.g.:
      - {"center": [x, y, z]}
      - {"center_x": x, "center_y": y, "center_z": z}
      - {"box": {"center": [x, y, z]}}
    """
    with open(path) as f:
        data = json.load(f)

    def _as_center(val: Any) -> Optional[Tuple[float, float, float]]:
        if isinstance(val, (list, tuple)) and len(val) == 3:
            try:
                return (float(val[0]), float(val[1]), float(val[2]))
            except (TypeError, ValueError):
                return None
        return None

    if isinstance(data, dict):
        if "center" in data:
            center = _as_center(data["center"])
            if center is not None:
                return center
        for key in ("box_center", "pocket_center", "centroid"):
            if key in data:
                center = _as_center(data[key])
                if center is not None:
                    return center
        if all(k in data for k in ("center_x", "center_y", "center_z")):
            return (float(data["center_x"]), float(data["center_y"]), float(data["center_z"]))
        if "box" in data and isinstance(data["box"], dict):
            box = data["box"]
            if "center" in box:
                center = _as_center(box["center"])
                if center is not None:
                    return center
            if all(k in box for k in ("center_x", "center_y", "center_z")):
                return (float(box["center_x"]), float(box["center_y"]), float(box["center_z"]))

    raise typer.BadParameter(f"Unsupported box JSON format: {path}")


def _print_usage_and_exit() -> None:
    usage = """\
[bold]Matcha v2.0.0[/bold] — AI-powered molecular docking

[bold]Usage:[/bold]
  matcha -r protein.pdb -l ligand.sdf -o output/
  matcha -r protein.pdb --ligand-dir ligands/ -o output/

[bold]Required:[/bold]
  -r, --receptor PATH     Protein structure (.pdb)
  -l, --ligand PATH       Ligand (.sdf/.mol/.mol2/.pdb)
  --ligand-dir PATH       Multi-ligand file/dir (.sdf/.mol2)
  -o, --out PATH          Output directory

[bold]Options:[/bold]
  -g, --device TEXT        Device: auto, cpu, cuda, cuda:N, mps
  --gpus TEXT              Multi-GPU ids for batch mode, e.g. 2,3
  --n-samples INT          Poses per ligand (default: 20)
  --scorer TEXT            gnina / custom / none (default: gnina)
  --autobox-ligand PATH   Box center from reference ligand
  --center-x/y/z FLOAT    Manual box center (Å)
  --overwrite              Overwrite existing run

  Run [bold]matcha --help[/bold] for all options."""
    console.print(usage)
    sys.exit(1)


def run_matcha(
    receptor: Optional[Path] = typer.Option(None, "-r", "--receptor", help="Protein structure (.pdb)."),
    ligand: Optional[Path] = typer.Option(None, "-l", "--ligand", help="Ligand with 3D coords (.sdf/.mol/.mol2/.pdb)."),
    ligand_dir: Optional[Path] = typer.Option(None, "--ligand-dir", help="File/dir with multiple ligands (.sdf/.mol2)."),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output directory."),
    checkpoints: Optional[Path] = typer.Option(None, "--checkpoints", help="Folder containing Matcha checkpoints (optional)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional base config to merge with defaults."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Working dir (defaults to --out)."),
    center_x: Optional[float] = typer.Option(None, "--center-x", "--center_x", help="X coordinate of box center (Å)"),
    center_y: Optional[float] = typer.Option(None, "--center-y", "--center_y", help="Y coordinate of box center (Å)"),
    center_z: Optional[float] = typer.Option(None, "--center-z", "--center_z", help="Z coordinate of box center (Å)"),
    autobox_ligand: Optional[Path] = typer.Option(None, "--autobox-ligand", "--autobox_ligand", help="Reference ligand file for autobox (.sdf/.mol/.pdb)"),
    box_json: Optional[Path] = typer.Option(None, "--box-json", help="JSON with binding-site box center (e.g. *_box.json)."),
    run_name: str = typer.Option("matcha_cli_run", "--run-name", help="Name for this docking run."),
    n_samples: int = typer.Option(20, "--n-samples", help="Number of samples (poses) to generate per ligand."),
    n_confs: Optional[int] = typer.Option(None, "--n-confs", help="Number of ligand conformers to generate with RDKit (default min(10, n-samples))."),
    device: Optional[str] = typer.Option(None, "--device", "-g", "--gpu", help="Device: auto, cpu, cuda, cuda:N, or mps."),
    gpus: Optional[str] = typer.Option(None, "--gpus", help="Comma-separated GPU ids for batch sharding, e.g. 2,3."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Remove existing run folder if present."),
    keep_workdir: bool = typer.Option(False, "--keep-workdir/--no-keep-workdir", help="Keep working data (default: False)."),
    log: Optional[Path] = typer.Option(None, "--log", help="Path to log file (defaults to <out>/<run-name>.log)."),
    docking_batch_limit: int = typer.Option(15000, "--docking-batch-limit", help="Number of tokens per batch for docking models."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recursively search ligand-dir for .sdf/.mol2 files (default: recursive)."),
    num_workers: Optional[int] = typer.Option(None, "--num-workers", help="DataLoader workers (default: auto for CUDA, else 0)."),
    pin_memory: Optional[bool] = typer.Option(None, "--pin-memory/--no-pin-memory", help="Pin DataLoader memory for faster H2D on CUDA (default: on for CUDA)."),
    prefetch_factor: int = typer.Option(2, "--prefetch-factor", help="DataLoader prefetch factor (workers only)."),
    persistent_workers: Optional[bool] = typer.Option(None, "--persistent-workers/--no-persistent-workers", help="Keep DataLoader workers alive (default: on when num-workers>0)."),
    scorer_type: str = typer.Option("gnina", "--scorer", help="Pose scorer: gnina, custom, or none."),
    scorer_path: Optional[Path] = typer.Option(None, "--scorer-path", help="Path to gnina binary or custom scorer script."),
    scorer_minimize: bool = typer.Option(True, "--scorer-minimize/--no-scorer-minimize", help="Minimize poses during scoring (gnina)."),
    gnina_batch_mode: str = typer.Option("per-ligand", "--gnina-batch-mode", help="GNINA scoring mode for batch runs (currently only per-ligand)."),
    physical_only: bool = typer.Option(False, "--physical-only/--keep-all-poses", help="Keep only PoseBusters-passing poses in outputs (default: False)."),
) -> None:
    if out is None:
        _print_usage_and_exit()
    batch_mode = ligand_dir is not None
    if (ligand is None) == (ligand_dir is None):
        _print_usage_and_exit()

    if n_samples < 1:
        console.print("[bold red]Error:[/bold red] --n-samples must be >= 1")
        raise typer.Exit(code=1)

    # Lazy imports — heavy libraries loaded only when actually running docking.
    # `global` so module-level helpers (_normalize_ligand, _autobox_from_ligand, etc.) can use them.
    global snapshot_download, np, OmegaConf, Chem
    from huggingface_hub import snapshot_download  # noqa: F811
    import numpy as np  # noqa: F811
    from omegaconf import OmegaConf  # noqa: F811
    from rdkit import Chem  # noqa: F811
    from matcha.utils.esm_utils import compute_esm_embeddings, compute_sequences
    from matcha.utils.inference_utils import run_v2_inference_pipeline, compute_fast_filters_from_sdf
    from matcha.utils.multigpu import parse_gpus, run_multigpu_batch
    from matcha.scoring import create_scorer
    from matcha.utils.device import resolve_device
    import torch

    multi_gpu_ids: list[int] = []
    if gpus is not None:
        try:
            multi_gpu_ids = parse_gpus(gpus)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    # Resolve device: auto, cpu, cuda, cuda:N, mps
    if multi_gpu_ids:
        resolved_device = f"cuda:{multi_gpu_ids[0]}"
    elif device is None or device == "auto":
        resolved_device = resolve_device()
    elif device.isdigit():
        # Bare number → treat as CUDA index (backwards compat with --gpu N)
        resolved_device = f"cuda:{int(device)}"
    elif device.startswith("cuda:"):
        # Explicit CUDA device string, e.g. cuda:0
        try:
            int(device.split(":")[1])
        except Exception:
            console.print(f"[bold red]Error:[/bold red] Invalid CUDA device '{device}'. Use cuda:N where N is an integer.")
            raise typer.Exit(code=1)
        resolved_device = device
    elif device in ("cuda", "mps", "cpu"):
        resolved_device = device
    else:
        console.print(f"[bold red]Error:[/bold red] Unknown device '{device}'. Use auto, cpu, cuda, cuda:N, or mps.")
        raise typer.Exit(code=1)

    cuda_device_idx = int(resolved_device.split(":")[1]) if resolved_device.startswith("cuda:") else 0

    if resolved_device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        console.print("[bold red]Error:[/bold red] MPS (Apple Metal) requested but not available.")
        raise typer.Exit(code=1)
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        console.print("[bold red]Error:[/bold red] CUDA requested but not available.")
        raise typer.Exit(code=1)
    if multi_gpu_ids and not torch.cuda.is_available():
        console.print("[bold red]Error:[/bold red] --gpus requires CUDA, but CUDA is not available.")
        raise typer.Exit(code=1)
    if multi_gpu_ids:
        visible_gpu_count = torch.cuda.device_count()
        for gpu_id in multi_gpu_ids:
            if gpu_id >= visible_gpu_count:
                raise typer.BadParameter(
                    f"--gpus contains GPU {gpu_id}, but only {visible_gpu_count} devices are visible."
                )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    checkpoints = _ensure_checkpoints(checkpoints or Path("checkpoints"))

    start_time = datetime.now(timezone.utc)
    base_workdir = workdir or out
    run_workdir = base_workdir / run_name
    if run_workdir.exists():
        if overwrite:
            _remove_tree_if_exists(run_workdir)
        else:
            raise typer.BadParameter(
                f"Working directory {run_workdir} already exists. Use --overwrite or change --run-name."
            )
    run_workdir.mkdir(parents=True, exist_ok=True)
    run_timer_start = time.perf_counter()
    prepare_input_sec = 0.0
    esm_sec = 0.0
    inference_sec = 0.0
    scoring_sec = 0.0
    pipeline_timings: dict = {}

    work_dir = run_workdir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    best_dir = all_dir = None
    if ligand_dir is not None:
        best_dir = run_workdir / "best_poses"
        all_dir = run_workdir / "all_poses"
        best_dir.mkdir(parents=True, exist_ok=True)
        all_dir.mkdir(parents=True, exist_ok=True)

    banner = r"""
███╗   ███╗ █████╗ ████████╗ ██████╗██╗  ██╗ █████╗
████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║  ██║██╔══██╗
██╔████╔██║███████║   ██║   ██║     ███████║███████║
██║╚██╔╝██║██╔══██║   ██║   ██║     ██╔══██║██╔══██║
██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗██║  ██║██║  ██║
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝
"""
    console.print(banner.rstrip("\n"))
    console.print("MATCHA DOCKING ENGINE  v2.0.0")
    console.print("=" * 60)
    console.print("")

    # Box handling
    manual_box_specified = center_x is not None or center_y is not None or center_z is not None
    autobox_specified = autobox_ligand is not None
    box_center_val: Optional[Tuple[float, float, float]] = None

    if sum((manual_box_specified, autobox_specified, box_json is not None)) > 1:
        raise typer.BadParameter("Cannot combine --box-json with manual box or --autobox-ligand.")
    if manual_box_specified:
        if center_x is None or center_y is None or center_z is None:
            raise typer.BadParameter("Manual box requires --center-x/--center-y/--center-z")
        box_center_val = (center_x, center_y, center_z)
        console.print(f"[bold green][matcha][/bold green] manual center: {box_center_val}")
    elif autobox_specified:
        if not autobox_ligand.exists():
            raise typer.BadParameter(f"Autobox ligand file not found: {autobox_ligand}")
        box_center_val = _autobox_from_ligand(autobox_ligand)
        console.print(f"[bold green][matcha][/bold green] autobox from reference ligand {autobox_ligand.name}")
        console.print(f"[bold green][matcha][/bold green] center: {box_center_val}")
    elif box_json is not None:
        if not box_json.exists():
            raise typer.BadParameter(f"Box JSON not found: {box_json}")
        box_center_val = _center_from_box_json(box_json)
        console.print(f"[bold green][matcha][/bold green] box center from {box_json.name}")
        console.print(f"[bold green][matcha][/bold green] center: {box_center_val}")
    else:
        console.print("[bold yellow][matcha][/bold yellow] No box specified - running blind docking on entire protein")

    if multi_gpu_ids and len(multi_gpu_ids) > 1:
        if not batch_mode:
            raise typer.BadParameter("Multi-GPU mode (--gpus with 2+ devices) is supported only with --ligand-dir.")
        if ligand_dir is None or not ligand_dir.is_dir():
            raise typer.BadParameter("Multi-GPU mode requires --ligand-dir to be a directory.")

        console.print(
            f"[bold cyan][matcha][/bold cyan] Multi-GPU batch mode on GPUs {multi_gpu_ids}: "
            "launching one shard process per GPU."
        )
        result = run_multigpu_batch(
            gpu_ids=multi_gpu_ids,
            run_workdir=run_workdir,
            run_name=run_name,
            receptor=receptor,
            ligand_dir=ligand_dir,
            recursive=recursive,
            checkpoints=checkpoints,
            config=config,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            autobox_ligand=autobox_ligand,
            box_json=box_json,
            n_samples=n_samples,
            n_confs=n_confs,
            docking_batch_limit=docking_batch_limit,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            scorer_type=scorer_type,
            scorer_path=scorer_path,
            scorer_minimize=scorer_minimize,
            gnina_batch_mode=gnina_batch_mode,
            physical_only=physical_only,
        )
        total_sec = time.perf_counter() - run_timer_start
        _write_json(
            run_workdir / "run_timing.json",
            {
                "mode": "multigpu",
                "run_name": run_name,
                "gpus": multi_gpu_ids,
                "prepare_input_sec": 0.0,
                "esm_sec": 0.0,
                "inference_sec": 0.0,
                "scoring_sec": 0.0,
                "total_sec": total_sec,
                "ligand_count": int(result["ligand_count"]),
                "benchmark_summary_path": str((run_workdir / "benchmark_summary.json").resolve()),
            },
        )
        console.print(f"[bold green][matcha][/bold green] merged output: {run_workdir / 'merged'}")
        console.print(
            f"[bold green][matcha][/bold green] benchmark summary: {run_workdir / 'benchmark_summary.json'}"
        )
        console.print(f"[bold green][matcha][/bold green] runtime: {_format_runtime(total_sec)}")
        return

    receptor_for_run = receptor
    pocket_centers_filename = None

    dataset_dir = work_dir / "datasets" / "any_conf"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    molecule_uids: List[str] = []

    if batch_mode:
        console.print(f"[bold cyan][matcha][/bold cyan] Batch mode: processing multiple molecules from {ligand_dir.name}")
        if ligand_dir.is_file():
            if ligand_dir.suffix.lower() not in {".sdf", ".mol2"}:
                raise typer.BadParameter("--ligand-dir file input supports .sdf or .mol2")
            molecules = _split_ligand_file(ligand_dir)
        elif ligand_dir.is_dir():
            molecules = []
            if recursive:
                ligand_files = list(ligand_dir.rglob("*.sdf")) + list(ligand_dir.rglob("*.mol2"))
            else:
                ligand_files = list(ligand_dir.glob("*.sdf")) + list(ligand_dir.glob("*.mol2"))
            ligand_files = sorted(ligand_files, key=lambda p: str(p).lower())
            for ligand_file in ligand_files:
                try:
                    molecules.extend(_split_ligand_file(ligand_file))
                except ValueError as e:
                    console.print(f"[yellow]Warning: {e}[/yellow]")
        else:
            raise typer.BadParameter("--ligand-dir must be a file or directory")
        if not molecules:
            raise typer.BadParameter(f"No molecules found in {ligand_dir}")
        console.print(f"[bold green][matcha][/bold green] Found {len(molecules)} molecules to process")
        molecule_uids = _prepare_batch_dataset(receptor, molecules, dataset_dir)
    else:
        _prepare_singleton_dataset(
            receptor_for_run,
            ligand,
            dataset_dir,
            run_name,
        )
        molecule_uids = [run_name]

    if box_center_val is not None:
        pocket_centers_filename = work_dir / 'stage1_any_conf.npy'
        _create_pocket_centers_file(box_center_val, molecule_uids, n_samples, pocket_centers_filename)

    base_conf = _load_base_conf(config)
    base_conf.device = resolved_device
    conf = _build_conf(base_conf, work_dir, checkpoints)
    if n_confs is not None and n_confs < 1:
        raise typer.BadParameter("--n-confs must be >= 1")
    console.print(f"[bold green][matcha][/bold green] workdir: {run_workdir}")
    console.print(f"[bold green][matcha][/bold green] checkpoints: {checkpoints}")
    console.print(f"[bold green][matcha][/bold green] samples per ligand: {n_samples}")
    console.print(f"[bold green][matcha][/bold green] device: {resolved_device}")

    if num_workers is None:
        if resolved_device.startswith("cuda"):
            cpu = os.cpu_count() or 8
            num_workers = min(8, max(1, cpu // 2))
        else:
            num_workers = 0
    if pin_memory is None:
        pin_memory = resolved_device.startswith("cuda")
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    prepare_input_sec = time.perf_counter() - run_timer_start
    esm_start = time.perf_counter()
    compute_sequences(conf)
    compute_esm_embeddings(conf)
    esm_sec = time.perf_counter() - esm_start
    inference_start = time.perf_counter()
    pipeline_timings = run_v2_inference_pipeline(
        copy.deepcopy(conf), run_name, n_samples,
        pocket_centers_filename=pocket_centers_filename,
        docking_batch_limit=docking_batch_limit,
        num_workers=num_workers,
    )
    inference_sec = time.perf_counter() - inference_start

    def _read_scored_sdf_affinity(sdf_path: Path) -> List[float]:
        """Read minimizedAffinity values from a GNINA-scored SDF file (one value per pose)."""
        if sdf_path is None or not sdf_path.exists():
            return []
        scores = []
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        for mol in suppl:
            if mol is None:
                scores.append(float("inf"))
                continue
            for prop in ("minimizedAffinity", "Affinity"):
                if mol.HasProp(prop):
                    try:
                        scores.append(float(mol.GetProp(prop)))
                        break
                    except (ValueError, TypeError):
                        continue
            else:
                scores.append(float("inf"))
        return scores

    preds_root = Path(conf.inference_results_folder) / run_name
    dataset_name = 'any_conf'

    # Optional GNINA scoring
    scorer_used = False
    sdf_scored = preds_root / dataset_name / "minimized_sdf_predictions"
    best_scored_dir = preds_root / dataset_name / "best_minimized_predictions"
    if scorer_type != "none":
        scoring_start = time.perf_counter()
        try:
            scorer = create_scorer(scorer_type, scorer_path=str(scorer_path) if scorer_path else None,
                                   minimize=scorer_minimize)
            sdf_input = preds_root / dataset_name / "sdf_predictions"
            filters_path = preds_root / dataset_name / "filters_results_minimized.json"
            if scorer_type.startswith("gnina") and batch_mode:
                if gnina_batch_mode != "per-ligand":
                    raise typer.BadParameter("--gnina-batch-mode currently supports only 'per-ligand'")
            scorer.score_poses(str(receptor), str(sdf_input), str(sdf_scored), device=cuda_device_idx)
            compute_fast_filters_from_sdf(conf, run_name, sdf_type='minimized', n_preds_to_use=n_samples)
            scorer.select_top_poses(str(sdf_scored), str(best_scored_dir),
                                    filters_path=str(filters_path), n_samples=n_samples)
            scorer_used = True
            console.print(f"[bold green][matcha][/bold green] scoring complete ({scorer.name})")
        except RuntimeError as e:
            console.print(f"[bold yellow][matcha][/bold yellow] scoring skipped: {e}")
        scoring_sec = time.perf_counter() - scoring_start

    metrics = np.load(preds_root / f"{dataset_name}_final_preds_merged.npy", allow_pickle=True).item()

    # Load PB filters from JSON
    filters_json_path = preds_root / dataset_name / "filters_results_minimized.json"
    pb_filters = {}
    if filters_json_path.exists():
        with open(filters_json_path) as f:
            pb_filters = json.load(f)

    # Enrich metrics with PB filter data
    _PB_BOOL_KEYS = ('not_too_far_away', 'no_internal_clash', 'no_clashes', 'no_volume_clash')
    for uid_key, uid_data in metrics.items():
        uid_real = uid_key.split('_mol')[0] if '_mol' in uid_key else uid_key
        if uid_real in pb_filters:
            fdata = pb_filters[uid_real]
            for i, sample in enumerate(uid_data.get('sample_metrics', [])):
                if i < len(fdata.get('posebusters_filters_passed_count_fast', [])):
                    sample['posebusters_filters_passed_count_fast'] = fdata['posebusters_filters_passed_count_fast'][i]
                    bool_flags = [fdata.get(k, [])[i] if i < len(fdata.get(k, [])) else False for k in _PB_BOOL_KEYS]
                    buried = fdata.get('is_buried_fraction', [])
                    bool_flags.append(buried[i] if i < len(buried) else 0.0)
                    sample['posebusters_filters_fast'] = bool_flags

    # Read GNINA scores back into metrics when available
    if scorer_used and sdf_scored.exists():
        for uid_key, uid_data in metrics.items():
            uid_real = uid_key.split('_mol')[0] if '_mol' in uid_key else uid_key
            scored_sdf = sdf_scored / f"{uid_real}.sdf"
            if not scored_sdf.exists():
                continue
            suppl = Chem.SDMolSupplier(str(scored_sdf), removeHs=False, sanitize=False)
            for i, mol in enumerate(suppl):
                if mol is None or i >= len(uid_data.get('sample_metrics', [])):
                    continue
                for prop in ['minimizedAffinity', 'Affinity', 'minimizedCNNscore', 'CNNscore']:
                    if mol.HasProp(prop):
                        try:
                            uid_data['sample_metrics'][i]['gnina_score'] = float(mol.GetProp(prop))
                            break
                        except (ValueError, TypeError):
                            continue

    def _rank_samples(sample_metrics: List[dict]) -> List[Tuple[int, dict]]:
        has_gnina = any('gnina_score' in s for s in sample_metrics)
        ranked_indices = sorted(
            range(len(sample_metrics)),
            key=lambda i: (
                -int(sample_metrics[i].get("posebusters_filters_passed_count_fast", 0)),
                float(sample_metrics[i].get("gnina_score", float("inf"))) if has_gnina else 0,
            ),
        )
        return [(rank, sample_metrics[i]) for rank, i in enumerate(ranked_indices, start=1)]

    def _save_all_poses_for_uid(metrics_data, uid, out_path, filter_non_physical: bool = True):
        if uid not in metrics_data:
            return [], 0, 0
        sample_data = metrics_data[uid]
        orig_mol = sample_data["orig_mol"]
        ranked = _rank_samples(sample_data["sample_metrics"])
        best_pb_count = max(int(s.get("posebusters_filters_passed_count_fast", 0)) for s in sample_data["sample_metrics"])

        # Optionally filter out poses failing PoseBusters fast checks
        ranked_filtered = [(r, s) for r, s in ranked if not filter_non_physical or int(s.get("posebusters_filters_passed_count_fast", 0)) == best_pb_count]
        ranked_to_use = ranked_filtered if ranked_filtered else ranked

        writer = Chem.SDWriter(str(out_path))
        writer.SetKekulize(False)
        for rank, sample in ranked_to_use:
            mol = copy.deepcopy(orig_mol)
            conf = Chem.Conformer(orig_mol.GetNumAtoms())
            for idx, (x, y, z) in enumerate(sample["pred_pos"]):
                conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            mol.SetProp("_Name", f"{uid}_rank{rank}")
            try:
                writer.write(mol)
            except Exception as exc:
                logger.warning(f"Failed to write pose SDF for {uid} rank {rank}: {exc}")
        writer.close()
        return ranked_to_use, len(ranked_filtered), len(ranked)

    def _get_best_sample_idx(pb_counts, gnina_scores=None):
        best_pb_count = max(pb_counts)
        pb_count_indices = np.arange(len(pb_counts))[pb_counts == best_pb_count]
        if gnina_scores is not None:
            scores = gnina_scores[pb_count_indices]
            best_score_idx = np.argmin(scores)
        else:
            best_score_idx = 0
        return pb_count_indices[best_score_idx]

    def _get_sample_stats(sample_metrics):
        pb_counts = np.array([int(s.get("posebusters_filters_passed_count_fast", 0)) for s in sample_metrics])
        has_gnina = any('gnina_score' in s for s in sample_metrics)
        gnina_scores = None
        if has_gnina:
            gnina_scores = np.array([float(s.get("gnina_score", float("inf"))) for s in sample_metrics])
        best_idx = _get_best_sample_idx(pb_counts, gnina_scores)
        return pb_counts, gnina_scores, has_gnina, best_idx

    def _write_best_pose_sdf(mdata, best_idx, uid_label, dest_path, scored_path=None):
        if scored_path is not None and scored_path.exists():
            shutil.copyfile(scored_path, dest_path)
        else:
            best_sample = mdata["sample_metrics"][best_idx]
            orig_mol = mdata["orig_mol"]
            writer = Chem.SDWriter(str(dest_path))
            writer.SetKekulize(False)
            mol = copy.deepcopy(orig_mol)
            conf = Chem.Conformer(orig_mol.GetNumAtoms())
            for idx, (x, y, z) in enumerate(best_sample["pred_pos"]):
                conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            mol.SetProp("_Name", f"{uid_label}_best")
            try:
                writer.write(mol)
            except Exception as exc:
                logger.warning(f"Failed to write best pose SDF for {uid_label}: {exc}")
            writer.close()

    def _format_pose_ranking_lines(ranked_samples, has_gnina):
        lines = []
        if has_gnina:
            lines.append("  rank  affinity  pb  checks  buried_frac")
            lines.append("  -----------------------------------------")
        else:
            lines.append("  rank  pb  checks  buried_frac")
            lines.append("  ------------------------------")
        for rank, sample in ranked_samples:
            pb_count = int(sample.get("posebusters_filters_passed_count_fast", 0))
            pb_flags = sample.get("posebusters_filters_fast", [])
            checks = "".join(
                "✓" if len(pb_flags) > j and pb_flags[j] else "✗"
                for j in range(4)
            )
            buried_val = pb_flags[4] if len(pb_flags) > 4 else None
            if isinstance(buried_val, (list, tuple)):
                buried_val = buried_val[0] if buried_val else None
            try:
                buried_frac = f"{float(buried_val):.2f}" if buried_val is not None else " n/a"
            except Exception:
                buried_frac = " n/a"
            if has_gnina:
                aff = f"{sample.get('gnina_score', float('inf')):>8.2f}"
                lines.append(f"  {rank:<4}  {aff}  {pb_count}/4  {checks}   {buried_frac:>6}")
            else:
                lines.append(f"  {rank:<4}  {pb_count}/4  {checks}   {buried_frac:>6}")
        lines.extend([
            "",
            "Legend: checks = not_far | no_int_clash | no_clash | no_vol_clash (✓=pass ✗=fail)",
            "",
            "",
        ])
        return lines

    def _format_box_log_section(box_center_val, autobox_specified, autobox_ligand):
        if box_center_val is not None:
            mode = 'manual center' if not autobox_specified else f'autobox from {autobox_ligand.name}'
            return [
                "[ AUTODOCKING BOX ]",
                f"  Mode             : {mode}",
                f"  Center (Å)       : ({box_center_val[0]:.3f}, {box_center_val[1]:.3f}, {box_center_val[2]:.3f})",
                "", "",
            ]
        return [
            "[ DOCKING MODE ]",
            "  Mode             : blind docking (entire protein)",
            "", "",
        ]

    def _format_summary_section(n_samples, scorer_type, scorer_used, scorer_name,
                                pb_counts, gnina_scores, has_gnina, best_idx,
                                kept_physical, total_samples, physical_only):
        lines = [
            "[ SUMMARY ]",
            f"  Samples per ligand     : {n_samples}",
            f"  Scorer                 : {scorer_type}" + (f" ({scorer_name})" if scorer_used else ""),
        ]
        if has_gnina:
            lines.append(f"  GNINA Affinity (kcal/mol): min={min(gnina_scores):.2f}, mean={float(np.mean(gnina_scores)):.2f}, max={max(gnina_scores):.2f}")
        filter_warning = "  [WARNING: none passed, keeping originals]" if physical_only and kept_physical == 0 else ""
        affinity_str = f", affinity={gnina_scores[best_idx]:.2f}" if has_gnina else ""
        lines.extend([
            f"  PoseBusters checks     : min={min(pb_counts)}/4, max={max(pb_counts)}/4",
            f"  Best sample            : rank={best_idx+1}, pb={pb_counts[best_idx]}/4{affinity_str}",
            f"  Filtered poses (pb_4/4): kept {kept_physical}/{total_samples}{filter_warning}",
            "", "",
            "  PoseBusters checks (4 boolean tests):",
            "    1. not_too_far_away   : ligand is close to protein (distance check)",
            "    2. no_internal_clash  : no bad bonds/angles in ligand geometry",
            "    3. no_clashes         : no inter-molecular clashes (ligand-protein)",
            "    4. no_volume_clash    : no vdW volume overlaps",
            "  Additional metric:",
            "    - buried_fraction     : fraction of ligand buried in protein (shown separately)",
            "",
        ])
        return lines

    scorer_name = scorer.name if scorer_used else ""

    # single mode output
    if not batch_mode:
        resolved_out = run_workdir / f"{run_name}_best.sdf"
        all_poses_dest = run_workdir / f"{run_name}_poses.sdf"

        uid, mdata = next(iter(metrics.items()))
        pb_counts, gnina_scores, has_gnina, best_idx = _get_sample_stats(mdata["sample_metrics"])

        scored_path = best_scored_dir / f"{run_name}.sdf" if scorer_used else None
        _write_best_pose_sdf(mdata, best_idx, uid, resolved_out, scored_path=scored_path)

        ranked_samples, kept_physical, total_samples = _save_all_poses_for_uid(metrics, uid, all_poses_dest, filter_non_physical=physical_only)

        end_time = datetime.now(timezone.utc)
        runtime = (end_time - start_time).total_seconds()
        _write_json(
            run_workdir / "run_timing.json",
            {
                "mode": "single",
                "run_name": run_name,
                "prepare_input_sec": prepare_input_sec,
                "esm_sec": esm_sec,
                "inference_sec": inference_sec,
                "scoring_sec": scoring_sec,
                "total_sec": runtime,
                "pipeline_timings": pipeline_timings or {},
            },
        )

        log_path = (log or run_workdir / f"{run_name}.log").resolve()
        receptor_abs = receptor.resolve()
        ligand_abs = ligand.resolve()
        resolved_out_abs = resolved_out.resolve()
        all_poses_abs = all_poses_dest.resolve()
        command_line = "uv run matcha " + " ".join(sys.argv[1:]) if ".venv/bin/matcha" in sys.argv[0] else " ".join(sys.argv)
        log_lines = [
            banner.rstrip("\n"),
            "MATCHA DOCKING ENGINE  v2.0.0",
            "============================================================",
            "", "",
            "[ RUN INFO ]",
            f"  Start time       : {start_time.isoformat()}Z",
            f"  Command          : {command_line}",
            f"  Workdir          : {run_workdir.resolve()}",
            f"  Runtime          : {_format_runtime(runtime)}",
            "", "",
        ]
        log_lines.extend(_format_box_log_section(box_center_val, autobox_specified, autobox_ligand))
        log_lines.extend([
            "[ INPUT / OUTPUT FILES ]",
            f"  Receptor         : {receptor_abs}",
            f"  Ligand           : {ligand_abs}",
            f"  Best pose SDF    : {resolved_out_abs}",
            f"  All poses SDF    : {all_poses_abs}",
            f"  Log file         : {log_path}",
            "", "",
        ])
        log_lines.extend(_format_summary_section(
            n_samples, scorer_type, scorer_used, scorer_name,
            pb_counts, gnina_scores, has_gnina, best_idx,
            kept_physical, total_samples, physical_only))
        log_lines.append("[ POSE RANKING ]")
        log_lines.extend(_format_pose_ranking_lines(ranked_samples, has_gnina))

        if max(pb_counts) < 4:
            log_lines.extend([
                "WARNING: No poses passed all PoseBusters checks (pb_4/4).",
                "         Inspect poses or regenerate with more samples/box adjustments.",
                "", "",
            ])

        log_lines.extend([
            "[ END ]",
            f"  Run finished at    : {end_time.isoformat()}Z",
            f"  Total runtime      : {_format_runtime(runtime)}",
            f"  Workdir preserved  : {run_workdir}",
            "============================================================",
        ])
        with open(log_path, "w") as log_file:
            log_file.write("\n".join(log_lines))

        # Print summary to console
        console.print("")
        for line in _format_summary_section(
                n_samples, scorer_type, scorer_used, scorer_name,
                pb_counts, gnina_scores, has_gnina, best_idx,
                kept_physical, total_samples, physical_only):
            console.print(line)
        console.print("[ POSE RANKING ]")
        for line in _format_pose_ranking_lines(ranked_samples, has_gnina):
            console.print(line)
        if max(pb_counts) < 4:
            console.print("[bold yellow]WARNING: No poses passed all PoseBusters checks (pb_4/4).[/bold yellow]")
            console.print("")
        console.print(f"  Best pose SDF    : {resolved_out_abs}")
        console.print(f"  All poses SDF    : {all_poses_abs}")
        console.print(f"  Log file         : {log_path}")
        console.print(f"  Runtime          : {_format_runtime(runtime)}")
        console.print("")

        if not keep_workdir:
            _remove_tree_if_exists(work_dir)
            console.print(f"[bold green][matcha][/bold green] cleaned workdir {work_dir}")
        else:
            console.print(f"[bold green][matcha][/bold green] keeping workdir at {work_dir}")
        return

    # batch mode output
    molecule_uids = sorted([p.name for p in (work_dir / "datasets" / "any_conf").iterdir() if p.is_dir()])
    receptor_abs = receptor.resolve()
    ligand_dir_abs = ligand_dir.resolve()
    run_workdir_abs = run_workdir.resolve()
    per_log_dir = run_workdir / "logs"
    per_log_dir.mkdir(parents=True, exist_ok=True)

    for mol_uid in molecule_uids:
        metrics_key = mol_uid if mol_uid in metrics else f"{mol_uid}_mol0"
        if metrics_key not in metrics:
            console.print(f"[yellow]Warning: No results for {mol_uid}[/yellow]")
            continue
        mdata = metrics[metrics_key]
        pb_counts, gnina_scores, has_gnina, best_idx = _get_sample_stats(mdata["sample_metrics"])

        best_dest = best_dir / f"{mol_uid}.sdf"
        all_dest = all_dir / f"{mol_uid}_poses.sdf"

        scored_path = best_scored_dir / f"{mol_uid}.sdf" if scorer_used else None
        _write_best_pose_sdf(mdata, best_idx, mol_uid, best_dest, scored_path=scored_path)

        ranked_samples, kept_physical, total_samples = _save_all_poses_for_uid(
            metrics, metrics_key, all_dest, filter_non_physical=physical_only
        )

        ligand_input = (run_workdir / "work" / "datasets" / "any_conf" / mol_uid / f"{mol_uid}_ligand.sdf").resolve()
        per_log_path = per_log_dir / f"{mol_uid}.log"

        end_time_local = datetime.now(timezone.utc)
        runtime_local = (end_time_local - start_time).total_seconds()

        log_lines = [
            banner.rstrip("\n"),
            "MATCHA DOCKING ENGINE  v2.0.0",
            "============================================================",
            "", "",
            "[ RUN INFO ]",
            f"  Start time       : {start_time.isoformat()}Z",
            f"  Command          : batch run {run_name} (ligand {mol_uid})",
            f"  Workdir          : {run_workdir_abs}",
            f"  Runtime          : {_format_runtime(runtime_local)}",
            "", "",
        ]
        log_lines.extend(_format_box_log_section(box_center_val, autobox_specified, autobox_ligand))
        log_lines.extend([
            "[ INPUT / OUTPUT FILES ]",
            f"  Receptor         : {receptor_abs}",
            f"  Ligand           : {ligand_input}",
            f"  Best pose SDF    : {best_dest.resolve()}",
            f"  All poses SDF    : {all_dest.resolve()}",
            f"  Log file         : {per_log_path.resolve()}",
            "", "",
        ])
        log_lines.extend(_format_summary_section(
            n_samples, scorer_type, scorer_used, scorer_name,
            pb_counts, gnina_scores, has_gnina, best_idx,
            kept_physical, total_samples, physical_only))
        log_lines.append("[ POSE RANKING ]")
        log_lines.extend(_format_pose_ranking_lines(ranked_samples, has_gnina))

        if max(pb_counts) < 4:
            log_lines.extend([
                "WARNING: No poses passed all PoseBusters checks (pb_4/4).",
                "         Inspect poses or regenerate with more samples/box adjustments.",
                "", "",
            ])

        log_lines.extend([
            "[ END ]",
            f"  Run finished at    : {end_time_local.isoformat()}Z",
            f"  Total runtime      : {_format_runtime(runtime_local)}",
            f"  Workdir preserved  : {run_workdir_abs}",
            "============================================================",
        ])
        with open(per_log_path, "w") as log_file:
            log_file.write("\n".join(log_lines))

    end_time = datetime.now(timezone.utc)
    runtime = (end_time - start_time).total_seconds()
    _write_json(
        run_workdir / "run_timing.json",
        {
            "mode": "batch",
            "run_name": run_name,
            "prepare_input_sec": prepare_input_sec,
            "esm_sec": esm_sec,
            "inference_sec": inference_sec,
            "scoring_sec": scoring_sec,
            "total_sec": runtime,
            "pipeline_timings": pipeline_timings or {},
            "ligand_count": len(molecule_uids),
        },
    )

    log_path = (log or run_workdir / f"{run_name}.log").resolve()
    command_line = "uv run matcha " + " ".join(sys.argv[1:]) if ".venv/bin/matcha" in sys.argv[0] else " ".join(sys.argv)
    log_lines = [
        banner.rstrip("\n"),
        "MATCHA DOCKING ENGINE  v2.0.0",
        "============================================================",
        "", "",
        "[ RUN INFO ]",
        f"  Start time       : {start_time.isoformat()}Z",
        f"  Command          : {command_line}",
        f"  Workdir          : {run_workdir_abs}",
        f"  Runtime          : {_format_runtime(runtime)}",
        "", "",
        "[ INPUT FILES ]",
        f"  Receptor         : {receptor_abs}",
        f"  Ligands          : {ligand_dir_abs} ({len(molecule_uids)} molecules)",
        f"  Output dir       : {run_workdir_abs}",
        f"  Scorer           : {scorer_type}" + (f" ({scorer_name})" if scorer_used else ""),
        "", "",
        "[ PROCESSING SUMMARY ]",
        f"  Samples per molecule : {n_samples}",
        f"  Total molecules      : {len(molecule_uids)}",
        "", "",
        "[ RESULTS ]",
    ]
    for mol_uid in molecule_uids:
        metrics_key = mol_uid if mol_uid in metrics else f"{mol_uid}_mol0"
        if metrics_key not in metrics:
            log_lines.append(f"  {mol_uid}: No results")
            continue
        mdata = metrics[metrics_key]
        pb_counts, gnina_scores, has_gnina, best_idx = _get_sample_stats(mdata["sample_metrics"])
        line = f"  {mol_uid}: pb={pb_counts[best_idx]}/4"
        if has_gnina:
            line += f", affinity={gnina_scores[best_idx]:.2f}"
        log_lines.append(line)

    log_lines.extend([
        "",
        "[ END ]",
        f"  Run finished at    : {end_time.isoformat()}Z",
        f"  Total runtime      : {_format_runtime(runtime)}",
        f"  Workdir preserved  : {run_workdir_abs}",
        "============================================================",
    ])
    with open(log_path, "w") as log_file:
        log_file.write("\n".join(log_lines))

    console.print("")
    console.print("[ BATCH SUMMARY ]")
    console.print(f"  Processed {len(molecule_uids)} molecules")
    console.print(f"  Results saved to : {run_workdir}")
    console.print(f"  Log file         : {log_path}")
    console.print(f"  Runtime          : {_format_runtime(runtime)}")
    console.print("")

    if not keep_workdir:
        _remove_tree_if_exists(work_dir)
        console.print(f"[bold green][matcha][/bold green] cleaned workdir {work_dir}")
    else:
        console.print(f"[bold green][matcha][/bold green] keeping workdir at {work_dir}")


def main() -> None:
    if len(sys.argv) == 1:
        _print_usage_and_exit()
    typer.run(run_matcha)


if __name__ == "__main__":
    main()
