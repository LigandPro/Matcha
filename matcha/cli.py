"""


Example:
    uv run match -r receptor.pdb -l ligand.sdf -o out.sdf --gpu 0
"""

import copy
import os
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import typer
from huggingface_hub import snapshot_download
import numpy as np
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
    "seed": 42,
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
    receptor: Path, ligand: Path, dataset_dir: Path, uid: str, original_receptor: Optional[Path] = None
) -> None:
    sample_dir = dataset_dir / uid
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Keep original filenames alongside uid-based ones for Matcha pipeline
    receptor_dest = sample_dir / f"{uid}_protein.pdb"
    ligand_dest = sample_dir / f"{uid}_ligand.sdf"

    if receptor.suffix.lower() != ".pdb":
        raise typer.BadParameter("Receptor must be a .pdb file.")
    shutil.copyfile(receptor, receptor_dest)
    _normalize_ligand(ligand, ligand_dest)

    # Also save original files with original names
    if original_receptor is not None and original_receptor != receptor:
        receptor_orig = sample_dir / original_receptor.name
        if receptor_orig != receptor_dest:
            shutil.copyfile(original_receptor, receptor_orig)

    ligand_orig = sample_dir / ligand.name
    if ligand_orig != ligand_dest:
        _normalize_ligand(ligand, ligand_orig)


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
    If no extension is provided, treat it as a directory when it doesn't exist yet; otherwise default to .sdf.
    """
    out = out.expanduser()
    if out.exists() and out.is_dir():
        return out / f"{run_name}.sdf"
    # If path has no suffix and doesn't exist yet, treat it as a folder
    if out.suffix == "":
        return out / f"{run_name}.sdf"
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


def _parse_csv_floats(val: str, expected_len: int, name: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in val.split(",") if p.strip() != ""]
    if len(parts) != expected_len:
        raise typer.BadParameter(f"{name} must have {expected_len} comma-separated values")
    try:
        floats = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise typer.BadParameter(f"{name} must contain numeric values") from exc
    return floats


def _autobox_from_ligand(ligand: Path, padding: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    mol = Chem.MolFromMolFile(str(ligand), removeHs=False, sanitize=False) if ligand.suffix.lower() in {".mol", ".mol2"} else None
    if mol is None and ligand.suffix.lower() == ".pdb":
        mol = Chem.MolFromPDBFile(str(ligand), removeHs=False, sanitize=False)
    if mol is None:
        suppl = Chem.SDMolSupplier(str(ligand), removeHs=False, sanitize=False)
        mol = suppl[0] if suppl and len(suppl) > 0 else None
    if mol is None or mol.GetNumConformers() == 0:
        raise typer.BadParameter(f"Failed to read ligand for autobox: {ligand}")
    coords = mol.GetConformer().GetPositions()
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2.0
    size = (maxs - mins) + 2 * float(padding)
    return tuple(center.tolist()), tuple(size.tolist())


def _autobox_from_protein(protein: Path, padding: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    coords = []
    with protein.open() as fin:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except ValueError:
                    continue
    if not coords:
        raise typer.BadParameter(f"No atom coordinates found in protein {protein}")
    arr = np.array(coords, dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    center = (mins + maxs) / 2.0
    size = (maxs - mins) + 2 * float(padding)
    return tuple(center.tolist()), tuple(size.tolist())


def _crop_protein_to_box(protein: Path, dest: Path, center: Tuple[float, float, float], size: Tuple[float, float, float]) -> None:
    cx, cy, cz = center
    sx, sy, sz = size
    half = (sx / 2.0, sy / 2.0, sz / 2.0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with protein.open() as fin, dest.open("w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                if (
                    abs(x - cx) <= half[0]
                    and abs(y - cy) <= half[1]
                    and abs(z - cz) <= half[2]
                ):
                    fout.write(line)
                    kept += 1
            elif line.startswith("END"):
                continue
        fout.write("END\n")
    if kept == 0:
        raise typer.BadParameter("Cropping box removed all protein atoms; adjust box center/size or padding.")

def _print_usage_and_exit() -> None:
    usage = """
Matcha docking - AI-powered molecular docking

Required:
  -r, --receptor PATH        Protein structure (.pdb)
  -l, --ligand PATH          Ligand (.sdf/.mol/.mol2/.pdb)
  -o, --out PATH             Output SDF path or directory

Search Space (optional - for focused docking):
  Manual box:
      --center-x, --center-y, --center-z FLOAT  Box center coordinates (Å)
      --size-x, --size-y, --size-z FLOAT        Box size in each dimension (Å)
  OR
  Autobox from reference ligand:
      --autobox-ligand PATH   Reference ligand for autobox (extracts box from its coords)
      --autobox-add FLOAT     Padding around reference ligand (Å, default: 4)

  If no box specified: blind docking on entire protein

Common:
      --checkpoints PATH     Folder for pipeline checkpoints (auto-download if missing)
      --run-name TEXT        Run name (default: matcha_cli_run)
      --n-samples INT        Samples per ligand (default: 20)
      --n-confs INT          Ligand conformers to generate (default: min(10, n-samples))
  -g, --gpu INT              CUDA device index
      --overwrite            Overwrite existing run folder
      --keep-workdir         Keep intermediates instead of cleaning up

Optional:
      --workdir PATH         Working directory (default: next to output)
      --log PATH             Log file path (default: <out>/<run-name>.log)
      --config PATH          Custom config to merge with defaults

Examples:
  # Blind docking on entire protein
  uv run matcha -r prot.pdb -l lig.sdf -o results/ --gpu 0

  # Autobox from reference ligand (default padding 4 Å)
  uv run matcha -r prot.pdb -l lig.sdf -o results/ \
    --autobox-ligand ref_ligand.sdf --gpu 0

  # Autobox with custom padding
  uv run matcha -r prot.pdb -l lig.sdf -o results/ \
    --autobox-ligand ref_ligand.sdf --autobox-add 6 --gpu 0

  # Manual box specification
  uv run matcha -r prot.pdb -l lig.sdf -o out.sdf \
    --center-x 10 --center-y 20 --center-z 30 \
    --size-x 20 --size-y 20 --size-z 20 --gpu 0
"""
    console.print(usage.rstrip())
    sys.exit(1)


def run_match(
    receptor: Optional[Path] = typer.Option(None, "-r", "--receptor", help="Protein structure (.pdb)."),
    ligand: Optional[Path] = typer.Option(None, "-l", "--ligand", help="Ligand with 3D coords (.sdf/.mol/.mol2/.pdb)."),
    out: Optional[Path] = typer.Option(
        None,
        "-o",
        "--out",
        help="Output path (SDF file or directory). All run artifacts will be stored alongside this location.",
    ),
    checkpoints: Optional[Path] = typer.Option(None, "--checkpoints", help="Folder containing the Matcha 'pipeline' checkpoints (optional)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional base config to merge with defaults."),
    workdir: Optional[Path] = typer.Option(
        None,
        "--workdir",
        help="Working dir (defaults to the folder of --out). Intermediates/logs stay here.",
    ),
    center_x: Optional[float] = typer.Option(None, "--center-x", "--center_x", help="X coordinate of box center (Å)"),
    center_y: Optional[float] = typer.Option(None, "--center-y", "--center_y", help="Y coordinate of box center (Å)"),
    center_z: Optional[float] = typer.Option(None, "--center-z", "--center_z", help="Z coordinate of box center (Å)"),
    size_x: Optional[float] = typer.Option(None, "--size-x", "--size_x", help="Box size in X dimension (Å)"),
    size_y: Optional[float] = typer.Option(None, "--size-y", "--size_y", help="Box size in Y dimension (Å)"),
    size_z: Optional[float] = typer.Option(None, "--size-z", "--size_z", help="Box size in Z dimension (Å)"),
    autobox_ligand: Optional[Path] = typer.Option(
        None,
        "--autobox-ligand",
        "--autobox_ligand",
        help="Reference ligand file for autobox (.sdf/.mol/.pdb)",
    ),
    autobox_add: Optional[float] = typer.Option(
        None,
        "--autobox-add",
        "--autobox_add",
        help="Padding to add around autobox ligand (Å, default: 4)",
    ),
    run_name: str = typer.Option("matcha_cli_run", "--run-name", help="Name for this docking run."),
    n_samples: int = typer.Option(20, "--n-samples", help="Number of samples to generate per ligand."),
    n_confs: Optional[int] = typer.Option(None, "--n-confs", help="Number of ligand conformers to generate (default min(10, n-samples))."),
    gpu: Optional[int] = typer.Option(
        None, "--gpu", "-g", "-gpu", help="CUDA device index (sets CUDA_VISIBLE_DEVICES)."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Remove existing run folder if present."),
    keep_workdir: bool = typer.Option(True, "--keep-workdir/--no-keep-workdir", help="Keep working data (default: True)."),
    log: Optional[Path] = typer.Option(None, "--log", help="Path to log file (defaults to <out>/<run-name>.log)."),
) -> None:
    if receptor is None or ligand is None or out is None:
        _print_usage_and_exit()

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    checkpoints = checkpoints or Path("checkpoints")
    checkpoints = _ensure_checkpoints(checkpoints)

    resolved_out = _resolve_output_path(out, run_name)
    output_dir = resolved_out.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = log or output_dir / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _print_console(msg: str) -> None:
        """Print only to console"""
        console.print(msg)

    start_time = datetime.utcnow()
    base_workdir = workdir or output_dir
    run_workdir = base_workdir / run_name if base_workdir != resolved_out else base_workdir
    if run_workdir.exists():
        if overwrite:
            shutil.rmtree(run_workdir)
        else:
            raise typer.BadParameter(
                f"Working directory {run_workdir} already exists. Use --overwrite or change --run-name."
            )
    run_workdir.mkdir(parents=True, exist_ok=True)

    banner = r"""
███╗   ███╗ █████╗ ████████╗ ██████╗██╗  ██╗ █████╗
████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║  ██║██╔══██╗
██╔████╔██║███████║   ██║   ██║     ███████║███████║
██║╚██╔╝██║██╔══██║   ██║   ██║     ██╔══██║██╔══██║
██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗██║  ██║██║  ██║
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝
"""
    console.print(banner.rstrip("\n"))
    console.print("MATCHA DOCKING ENGINE  v0.1.0")
    console.print("=" * 60)
    console.print("")

    # Search space (box) configuration - OPTIONAL
    # If not specified, will run on entire protein (blind docking)
    box_center_val: Optional[Tuple[float, float, float]] = None
    box_size_val: Optional[Tuple[float, float, float]] = None

    # Check which method was specified
    manual_box_specified = any([center_x, center_y, center_z, size_x, size_y, size_z])
    autobox_specified = autobox_ligand is not None

    if manual_box_specified and autobox_specified:
        raise typer.BadParameter("Cannot use both manual box and autobox. Choose one method.")

    # Manual box
    if manual_box_specified:
        if not all([center_x is not None, center_y is not None, center_z is not None,
                   size_x is not None, size_y is not None, size_z is not None]):
            raise typer.BadParameter(
                "Manual box requires ALL six parameters:\n"
                "  --center-x, --center-y, --center-z, --size-x, --size-y, --size-z"
            )
        box_center_val = (center_x, center_y, center_z)
        box_size_val = (size_x, size_y, size_z)
        if any(v <= 0 for v in box_size_val):
            raise typer.BadParameter("Box size values must be > 0")
        _print_console(f"[bold green][matcha][/bold green] manual box -> center {box_center_val}, size {box_size_val}")

    # Autobox from reference ligand
    elif autobox_specified:
        if not autobox_ligand.exists():
            raise typer.BadParameter(f"Autobox ligand file not found: {autobox_ligand}")
        padding = autobox_add if autobox_add is not None else 4.0
        if padding <= 0:
            raise typer.BadParameter("--autobox-add padding must be > 0")
        box_center_val, box_size_val = _autobox_from_ligand(autobox_ligand, padding)
        _print_console(f"[bold green][matcha][/bold green] autobox from reference ligand {autobox_ligand.name} with padding {padding:.1f} Å")
        _print_console(f"[bold green][matcha][/bold green] box -> center {box_center_val}, size {box_size_val}")

    # No box specified - blind docking on entire protein
    else:
        _print_console(f"[bold yellow][matcha][/bold yellow] No box specified - running blind docking on entire protein")
        _print_console(f"[bold yellow][matcha][/bold yellow] For faster/focused docking, consider using --autobox-ligand or manual box")

    receptor_for_run = receptor
    original_receptor = receptor
    if box_center_val is not None and box_size_val is not None:
        boxed_receptor = run_workdir / f"{run_name}_boxed_receptor.pdb"
        _crop_protein_to_box(receptor, boxed_receptor, box_center_val, box_size_val)
        receptor_for_run = boxed_receptor
        _print_console(f"[bold green][matcha][/bold green] cropped receptor saved to {boxed_receptor}")

    dataset_dir = run_workdir / "datasets" / "any_conf"
    _prepare_singleton_dataset(receptor_for_run, ligand, dataset_dir, run_name, original_receptor=original_receptor)

    base_conf = _load_base_conf(config)
    conf = _build_conf(base_conf, run_workdir, checkpoints)

    if n_confs is not None and n_confs < 1:
        raise typer.BadParameter("--n-confs must be >= 1")
    if n_confs is not None and n_samples > n_confs:
        n_samples = n_confs

    _print_console(f"[bold green][matcha][/bold green] workdir: {run_workdir}")
    _print_console(f"[bold green][matcha][/bold green] checkpoints: {checkpoints}")
    _print_console(f"[bold green][matcha][/bold green] samples per ligand: {n_samples}")
    if n_confs is not None:
        _print_console(f"[bold green][matcha][/bold green] ligand conformers: {n_confs}")
        conf.n_confs_override = int(n_confs)
    if gpu is not None:
        _print_console(f"[bold green][matcha][/bold green] using CUDA device #{gpu}")

    compute_sequences(conf)
    compute_esm_embeddings(conf)
    run_inference_pipeline(copy.deepcopy(conf), run_name, n_samples)
    compute_fast_filters(conf, run_name, n_samples)
    save_best_pred_to_sdf(conf, run_name)

    preds_root = Path(conf.inference_results_folder) / run_name
    fast_metrics_path = preds_root / f"{conf.test_dataset_types[0]}_final_preds_fast_metrics.npy"
    pred_sdf = preds_root / conf.test_dataset_types[0] / "sdf_predictions" / f"{run_name}.sdf"
    if not pred_sdf.exists():
        raise typer.Exit(code=1)

    def _rank_samples(sample_metrics: List[dict]) -> List[Tuple[int, dict]]:
        ranked_indices = sorted(
            range(len(sample_metrics)),
            key=lambda i: (
                float(sample_metrics[i].get("error_estimate_0", float("inf"))),
                -int(sample_metrics[i].get("posebusters_filters_passed_count_fast", 0)),
            ),
        )
        return [(rank, sample_metrics[i]) for rank, i in enumerate(ranked_indices, start=1)]

    def _save_all_poses(metrics_path: Path, out_path: Path) -> List[Tuple[int, dict]]:
        data = np.load(metrics_path, allow_pickle=True).item()
        if len(data) == 0:
            return []
        uid, sample_data = next(iter(data.items()))
        orig_mol = sample_data["orig_mol"]
        writer = Chem.SDWriter(str(out_path))
        ranked = _rank_samples(sample_data["sample_metrics"])
        for rank, sample in ranked:
            mol = copy.deepcopy(orig_mol)
            conf = Chem.Conformer(orig_mol.GetNumAtoms())
            for idx, (x, y, z) in enumerate(sample["pred_pos"]):
                conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            mol.SetProp("_Name", f"{uid}_rank{rank}")
            if "error_estimate_0" in sample:
                mol.SetDoubleProp("error_estimate_0", float(sample["error_estimate_0"]))
            pb_count = sample.get("posebusters_filters_passed_count_fast")
            if pb_count is not None:
                mol.SetIntProp("posebusters_pass_fast", int(pb_count))
            pb_flags = sample.get("posebusters_filters_fast")
            if pb_flags is not None:
                keys = ["not_too_far_away", "no_internal_clash", "no_clashes", "no_volume_clash", "is_buried_fraction"]
                for key_idx, key in enumerate(keys):
                    mol.SetProp(f"pb_{key}", str(pb_flags[key_idx]))
            writer.write(mol)
        writer.close()
        return ranked

    # Copy best pose to user-visible location
    shutil.copyfile(pred_sdf, resolved_out)
    _print_console(f"[bold green][matcha][/bold green] saved top-ranked pose to {resolved_out}")

    # Save all poses with properties
    all_pose_sdf = output_dir / f"{run_name}_poses.sdf"
    try:
        ranked_samples = _save_all_poses(fast_metrics_path, all_pose_sdf)
        _print_console(f"[bold green][matcha][/bold green] saved all poses (with conformers) to {all_pose_sdf}")
    except Exception as exc:  # pragma: no cover - best-effort logging
        _print_console(f"[bold yellow][matcha][/bold yellow] failed to save all poses: {exc}")
        ranked_samples = []

# Summaries
    metrics = np.load(fast_metrics_path, allow_pickle=True).item()
    uid, mdata = next(iter(metrics.items()))
    errs = [float(s.get("error_estimate_0", float("inf"))) for s in mdata["sample_metrics"]]
    pb_counts = [int(s.get("posebusters_filters_passed_count_fast", 0)) for s in mdata["sample_metrics"]]
    best_idx = int(np.argmin(errs))

    end_time = datetime.utcnow()
    runtime = (end_time - start_time).total_seconds()
    # Simplify command line display
    if sys.argv[0].endswith('/.venv/bin/matcha'):
        command_line = "uv run matcha " + " ".join(sys.argv[1:])
    else:
        command_line = " ".join(sys.argv)
    host = socket.gethostname()
    seed_value = int(conf.get("seed", DEFAULT_CONF["seed"]))
    cuda_device = gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "auto")

    # Convert paths to absolute for logging
    receptor_abs = receptor.resolve()
    ligand_abs = ligand.resolve()

    # Determine box mode for logging
    if autobox_ligand is not None:
        padding = autobox_add if autobox_add is not None else 4.0
        box_mode = f"autobox from reference ligand ({autobox_ligand.name}), padding {padding:.1f} Å"
    elif manual_box_specified:
        box_mode = "manual"
    else:
        box_mode = "blind docking (entire protein)"

    # Calculate box volume
    box_volume = None
    if box_size_val is not None:
        box_volume = box_size_val[0] * box_size_val[1] * box_size_val[2]

    log_lines = [
        banner.rstrip("\n"),
        "MATCHA DOCKING ENGINE  v0.1.0",
        "============================================================",
        "",
        "",
        "[ RUN INFO ]",
        f"  Start time       : {start_time.isoformat()}Z",
        f"  Host             : {host}",
        f"  Command          : {command_line}",
        f"  Workdir          : {run_workdir}",
        f"  Random seed      : {seed_value}",
        f"  CUDA device      : {cuda_device}",
        "",
        "",
    ]

    # Only show AUTODOCKING BOX section if box is specified
    if box_center_val is not None and box_size_val is not None:
        log_lines.extend([
            "[ AUTODOCKING BOX ]",
            f"  Mode             : {box_mode}",
            f"  Center (Å)       : ({box_center_val[0]:.3f}, {box_center_val[1]:.3f}, {box_center_val[2]:.3f})",
            f"  Size (Å)         : ({box_size_val[0]:.3f}, {box_size_val[1]:.3f}, {box_size_val[2]:.3f})",
            f"  Volume (Å^3)     : {box_volume:.1f}",
            "",
            "",
        ])
    else:
        log_lines.extend([
            "[ DOCKING MODE ]",
            f"  Mode             : {box_mode}",
            "",
            "",
        ])

    log_lines.extend([
        "[ INPUT / OUTPUT FILES ]",
        f"  Receptor         : {receptor_abs}",
        f"  Ligand           : {ligand_abs}",
        f"  Best pose SDF    : {resolved_out}",
        f"  All poses SDF    : {all_pose_sdf}",
        f"  Log file         : {log_path}",
        "",
        "",
        "",
        "[ SUMMARY ]",
        f"  Samples per ligand     : {n_samples}",
        f"  error_estimate_0 (Å)   : min={min(errs):.3f}, mean={float(np.mean(errs)):.3f}, max={max(errs):.3f}",
        f"  posebusters_fast checks: min={min(pb_counts)}/4, max={max(pb_counts)}/4",
        f"  Best sample            : rank={best_idx+1}, error_estimate_0={errs[best_idx]:.3f}, posebusters_pass_fast={pb_counts[best_idx]}/4",
        "",
        "",
        "  Note: error_estimate_0 is a lower-is-better RMSD-like metric (Å).",
        "        Negative values indicate high model confidence, positive values indicate uncertainty.",
        "        Lower values = higher confidence in the predicted pose.",
        "",
        "",
        "  PoseBusters checks (4 boolean tests):",
        "    1. not_too_far_away   : ligand is close to protein (distance check)",
        "    2. no_internal_clash  : no bad bonds/angles in ligand geometry",
        "    3. no_clashes         : no inter-molecular clashes (ligand-protein)",
        "    4. no_volume_clash    : no vdW volume overlaps",
        "  Additional metric:",
        "    - buried_fraction     : fraction of ligand buried in protein (shown separately)",
        "",
        "[ POSE RANKING ]",
        "  mode  error_est_0(Å)  pb_4/4  not_far  no_int_clash  no_clash  no_vol_clash  buried_frac",
        "  --------------------------------------------------------------------------------------------",
    ])
    for mode, sample in ranked_samples:
        err = float(sample.get("error_estimate_0", float("inf")))
        pb_count = int(sample.get("posebusters_filters_passed_count_fast", 0))
        pb_flags = sample.get("posebusters_filters_fast", [])

        # Extract individual flags
        not_far = "✓" if len(pb_flags) > 0 and pb_flags[0] else "✗"
        no_int_clash = "✓" if len(pb_flags) > 1 and pb_flags[1] else "✗"
        no_clash = "✓" if len(pb_flags) > 2 and pb_flags[2] else "✗"
        no_vol_clash = "✓" if len(pb_flags) > 3 and pb_flags[3] else "✗"
        buried_frac = f"{pb_flags[4]:.2f}" if len(pb_flags) > 4 else "n/a"

        log_lines.append(f"    {mode:<3}     {err:>7.3f}       {pb_count}/4      {not_far:^4}     {no_int_clash:^8}     {no_clash:^5}     {no_vol_clash:^8}      {buried_frac:>6}")

    log_lines.extend([
        "",
        "Legend: ✓ = passed, ✗ = failed",
        "",
        "",
    ])

    # Add warning if any error_estimate_0 is positive
    if any(e > 0 for e in errs):
        log_lines.extend([
            "WARNING: Some poses have positive error_est_0 values.",
            "         This indicates reduced model confidence in those predictions.",
            "         Consider generating more samples or inspecting poses manually.",
            "",
            "",
        ])

    log_lines.extend([
        "[ END ]",
        f"  Run finished at    : {end_time.isoformat()}Z",
        f"  Total runtime      : {runtime:.1f} s",
        f"  Workdir preserved  : {run_workdir}",
        "============================================================",
    ])

    # Write full report to log file
    with open(log_path, "w") as log_file:
        for line in log_lines:
            log_file.write(line + "\n")

    # Print summary to console
    console.print("")
    console.print("[ SUMMARY ]")
    console.print(f"  Samples per ligand     : {n_samples}")
    console.print(f"  error_estimate_0 (Å)   : min={min(errs):.3f}, mean={float(np.mean(errs)):.3f}, max={max(errs):.3f}")
    console.print(f"  posebusters_fast checks: min={min(pb_counts)}/4, max={max(pb_counts)}/4")
    console.print(f"  Best sample            : rank={best_idx+1}, error_estimate_0={errs[best_idx]:.3f}, posebusters_pass_fast={pb_counts[best_idx]}/4")
    console.print("")
    console.print(f"  Best pose SDF    : {resolved_out}")
    console.print(f"  All poses SDF    : {all_pose_sdf}")
    console.print(f"  Log file         : {log_path}")
    console.print(f"  Runtime          : {runtime:.1f} s")
    console.print("")

    if keep_workdir:
        console.print(f"[bold green][matcha][/bold green] keeping workdir at {run_workdir}")
    else:
        shutil.rmtree(run_workdir, ignore_errors=True)
        console.print(f"[bold green][matcha][/bold green] cleaned workdir {run_workdir}")


def main() -> None:
    if len(sys.argv) == 1:
        _print_usage_and_exit()
    typer.run(run_match)


if __name__ == "__main__":
    main()
