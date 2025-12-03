"""
Example:
    uv run matcha -r receptor.pdb -l ligand.sdf -o out.sdf --gpu 0
"""

import copy
import os
import shutil
import socket
import sys
from datetime import datetime, timezone
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

    receptor_dest = sample_dir / f"{uid}_protein.pdb"
    ligand_dest = sample_dir / f"{uid}_ligand.sdf"

    if receptor.suffix.lower() != ".pdb":
        raise typer.BadParameter("Receptor must be a .pdb file.")
    shutil.copyfile(receptor, receptor_dest)
    _normalize_ligand(ligand, ligand_dest)

    if original_receptor is not None and original_receptor != receptor:
        receptor_orig = sample_dir / original_receptor.name
        if receptor_orig != receptor_dest:
            shutil.copyfile(original_receptor, receptor_orig)

    ligand_orig = sample_dir / ligand.name
    if ligand_orig != ligand_dest:
        _normalize_ligand(ligand, ligand_orig)


def _split_multi_sdf(sdf_path: Path) -> List[Tuple[str, Chem.Mol]]:
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


def _prepare_batch_dataset(protein: Path, molecules: List[Tuple[str, Chem.Mol]], dataset_dir: Path) -> List[str]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    uids = []
    for name, mol in molecules:
        uid = name
        sample_dir = dataset_dir / uid
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(protein, sample_dir / f"{uid}_protein.pdb")
        writer = Chem.SDWriter(str(sample_dir / f"{uid}_ligand.sdf"))
        writer.write(mol)
        writer.close()
        uids.append(uid)
    return uids


def _create_batch_pocket_centers(pocket_center: Tuple[float, float, float], molecule_uids: List[str], n_samples: int, output_path: Path) -> None:
    pocket_centers = {}
    ligand_center = np.array(pocket_center)
    protein_center = np.zeros(3)
    for uid in molecule_uids:
        for conf_idx in range(n_samples):
            pocket_centers[f"{uid}_mol0_conf{conf_idx}"] = [{"tr_pred_init": ligand_center, "full_protein_center": protein_center}]
    np.save(output_path, [pocket_centers])


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


def _ensure_checkpoints(path: Path) -> Path:
    path = path.expanduser()
    stage1 = path / "pipeline" / "stage1" / "model.safetensors"
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
    mol = Chem.MolFromMolFile(str(ligand), removeHs=False, sanitize=False) if ligand.suffix.lower() in {".mol", ".mol2"} else None
    if mol is None and ligand.suffix.lower() == ".pdb":
        mol = Chem.MolFromPDBFile(str(ligand), removeHs=False, sanitize=False)
    if mol is None:
        suppl = Chem.SDMolSupplier(str(ligand), removeHs=False, sanitize=False)
        mol = suppl[0] if suppl and len(suppl) > 0 else None
    if mol is None or mol.GetNumConformers() == 0:
        raise typer.BadParameter(f"Failed to read ligand for autobox: {ligand}")
    mol = Chem.RemoveAllHs(mol, sanitize=False)
    return tuple(mol.GetConformer().GetPositions().mean(axis=0).tolist())


def _print_usage_and_exit() -> None:
    usage = """
Matcha docking - AI-powered molecular docking

Required (choose single or batch):
  SINGLE: -r, --receptor PATH         Protein (.pdb)
          -l, --ligand PATH          Ligand (.sdf/.mol/.mol2/.pdb)
  BATCH : -r, --receptor PATH         Protein (.pdb)
          --ligand-dir PATH          File/dir with multiple ligands (.sdf/.mol)
  Output: -o, --out PATH              Output directory (root for runs)

Search Space (optional):
  Manual box:   --center-x/--center-y/--center-z FLOAT  (Å)
  Autobox:      --autobox-ligand PATH   Reference ligand; box center taken from its coords
  If nothing specified: blind docking on whole protein

Common:
  --checkpoints PATH      Checkpoints folder (auto-download if missing)
  --run-name TEXT         Run name (default: matcha_cli_run)
  --n-samples INT         Poses per ligand (default: 20)
  --n-confs INT           Ligand conformers (default: min(10, n-samples))
  -g, --gpu INT           CUDA device index
  --overwrite             Remove existing run folder if present
  --keep-workdir          Keep intermediates (work/) instead of cleaning
  --physical-only         Save only PoseBusters-passing poses (pb_4/4); default off

Performance / paths:
  --docking-batch-limit INT   Tokens per docking batch (default: 15000)
  --scoring-batch-size INT    Batch size for scoring (default: 4)
  --workdir PATH              Working dir (defaults to <out>); intermediates live here
  --log PATH                  Log path (default: <out>/<run-name>.log)
  --config PATH               Extra config merged with defaults

Examples:
  uv run matcha -r prot.pdb -l lig.sdf -o results/ --gpu 0
  uv run matcha -r prot.pdb --ligand-dir ligs.sdf -o results/ --run-name batch --gpu 0
  uv run matcha -r prot.pdb -l lig.sdf --autobox-ligand ref.sdf -o results/
"""
    console.print(usage.rstrip())
    sys.exit(1)


def run_matcha(
    receptor: Optional[Path] = typer.Option(None, "-r", "--receptor", help="Protein structure (.pdb)."),
    ligand: Optional[Path] = typer.Option(None, "-l", "--ligand", help="Ligand with 3D coords (.sdf/.mol/.mol2/.pdb)."),
    ligand_dir: Optional[Path] = typer.Option(None, "--ligand-dir", help="File/dir with multiple ligands (.sdf/.mol)."),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output directory."),
    checkpoints: Optional[Path] = typer.Option(None, "--checkpoints", help="Folder containing Matcha checkpoints (optional)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional base config to merge with defaults."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Working dir (defaults to --out)."),
    center_x: Optional[float] = typer.Option(None, "--center-x", "--center_x", help="X coordinate of box center (Å)"),
    center_y: Optional[float] = typer.Option(None, "--center-y", "--center_y", help="Y coordinate of box center (Å)"),
    center_z: Optional[float] = typer.Option(None, "--center-z", "--center_z", help="Z coordinate of box center (Å)"),
    autobox_ligand: Optional[Path] = typer.Option(None, "--autobox-ligand", "--autobox_ligand", help="Reference ligand file for autobox (.sdf/.mol/.pdb)"),
    run_name: str = typer.Option("matcha_cli_run", "--run-name", help="Name for this docking run."),
    n_samples: int = typer.Option(20, "--n-samples", help="Number of samples (poses) to generate per ligand."),
    n_confs: Optional[int] = typer.Option(None, "--n-confs", help="Number of ligand conformers to generate with RDKit (default min(10, n-samples))."),
    gpu: Optional[int] = typer.Option(None, "--gpu", "-g", "-gpu", help="CUDA device index."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Remove existing run folder if present."),
    keep_workdir: bool = typer.Option(False, "--keep-workdir/--no-keep-workdir", help="Keep working data (default: False)."),
    log: Optional[Path] = typer.Option(None, "--log", help="Path to log file (defaults to <out>/<run-name>.log)."),
    docking_batch_limit: int = typer.Option(15000, "--docking-batch-limit", help="Number of tokens per batch for docking models."),
    scoring_batch_size: int = typer.Option(4, "--scoring-batch-size", help="Batch size for scoring."),
    physical_only: bool = typer.Option(False, "--physical-only/--keep-all-poses", help="Keep only PoseBusters-passing poses in outputs (default: False)."),
) -> None:
    if out is None:
        _print_usage_and_exit()
    batch_mode = ligand_dir is not None
    if (ligand is None) == (ligand_dir is None):
        _print_usage_and_exit()

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    checkpoints = _ensure_checkpoints(checkpoints or Path("checkpoints"))

    start_time = datetime.now(timezone.utc)
    base_workdir = workdir or out
    run_workdir = base_workdir / run_name
    if run_workdir.exists():
        if overwrite:
            shutil.rmtree(run_workdir)
        else:
            raise typer.BadParameter(
                f"Working directory {run_workdir} already exists. Use --overwrite or change --run-name."
            )
    run_workdir.mkdir(parents=True, exist_ok=True)

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
    console.print("MATCHA DOCKING ENGINE  v0.1.0")
    console.print("=" * 60)
    console.print("")

    # Box handling
    manual_box_specified = any([center_x, center_y, center_z])
    autobox_specified = autobox_ligand is not None
    box_center_val: Optional[Tuple[float, float, float]] = None

    if manual_box_specified and autobox_specified:
        raise typer.BadParameter("Cannot use both manual box and autobox. Choose one method.")
    if manual_box_specified:
        if not all([center_x is not None, center_y is not None, center_z is not None]):
            raise typer.BadParameter("Manual box requires --center-x/--center-y/--center-z")
        box_center_val = (center_x, center_y, center_z)
        console.print(f"[bold green][matcha][/bold green] manual center: {box_center_val}")
    elif autobox_specified:
        if not autobox_ligand.exists():
            raise typer.BadParameter(f"Autobox ligand file not found: {autobox_ligand}")
        box_center_val = _autobox_from_ligand(autobox_ligand)
        console.print(f"[bold green][matcha][/bold green] autobox from reference ligand {autobox_ligand.name}")
        console.print(f"[bold green][matcha][/bold green] center: {box_center_val}")
    else:
        console.print(f"[bold yellow][matcha][/bold yellow] No box specified - running blind docking on entire protein")

    receptor_for_run = receptor
    pocket_centers_filename = None

    dataset_dir = work_dir / "datasets" / "any_conf"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    molecule_uids: List[str] = []

    if batch_mode:
        console.print(f"[bold cyan][matcha][/bold cyan] Batch mode: processing multiple molecules from {ligand_dir.name}")
        if ligand_dir.is_file():
            molecules = _split_multi_sdf(ligand_dir)
        elif ligand_dir.is_dir():
            molecules = []
            for sdf_file in ligand_dir.glob("*.sdf"):
                molecules.extend(_split_multi_sdf(sdf_file))
        else:
            raise typer.BadParameter("--ligand-dir must be a file or directory")
        if not molecules:
            raise typer.BadParameter(f"No molecules found in {ligand_dir}")
        console.print(f"[bold green][matcha][/bold green] Found {len(molecules)} molecules to process")
        molecule_uids = _prepare_batch_dataset(receptor, molecules, dataset_dir)
        if box_center_val is not None:
            pocket_centers_filename = work_dir / 'stage1_any_conf.npy'
            _create_batch_pocket_centers(box_center_val, molecule_uids, n_samples, pocket_centers_filename)
    else:
        if box_center_val is not None:
            pocket_centers_filename = work_dir / 'stage1_any_conf.npy'
            def _create_pocket_centers_file(box_center_val: Tuple[float, float, float], n_samples: int, complex_name: str, pocket_centers_filename: Path) -> None:
                pocket_centers = {}
                ligand_center = np.array(box_center_val)
                protein_center = np.zeros(3)
                for i in range(n_samples):
                    pocket_centers[f'{complex_name}_mol0_conf{i}'] = [{'tr_pred_init': ligand_center, 'full_protein_center': protein_center}]
                np.save(pocket_centers_filename, [pocket_centers])
            _create_pocket_centers_file(box_center_val, n_samples, complex_name=run_name, pocket_centers_filename=pocket_centers_filename)
        _prepare_singleton_dataset(receptor_for_run, ligand, dataset_dir, run_name, original_receptor=receptor)
        molecule_uids = [run_name]

    base_conf = _load_base_conf(config)
    conf = _build_conf(base_conf, work_dir, checkpoints)
    if n_confs is not None and n_confs < 1:
        raise typer.BadParameter("--n-confs must be >= 1")
    console.print(f"[bold green][matcha][/bold green] workdir: {run_workdir}")
    console.print(f"[bold green][matcha][/bold green] checkpoints: {checkpoints}")
    console.print(f"[bold green][matcha][/bold green] samples per ligand: {n_samples}")
    if gpu is not None:
        console.print(f"[bold green][matcha][/bold green] using CUDA device #{gpu}")

    compute_sequences(conf)
    compute_esm_embeddings(conf)
    run_inference_pipeline(copy.deepcopy(conf), run_name, n_samples, pocket_centers_filename,
                           docking_batch_limit=docking_batch_limit, scoring_batch_size=scoring_batch_size)
    compute_fast_filters(conf, run_name, n_samples)
    save_best_pred_to_sdf(conf, run_name)

    preds_root = Path(conf.inference_results_folder) / run_name
    dataset_name = conf.test_dataset_types[0]
    fast_metrics_path = preds_root / f"{dataset_name}_final_preds_fast_metrics.npy"
    metrics = np.load(fast_metrics_path, allow_pickle=True).item()

    def _rank_samples(sample_metrics: List[dict]) -> List[Tuple[int, dict]]:
        ranked_indices = sorted(
            range(len(sample_metrics)),
            key=lambda i: (
                -int(sample_metrics[i].get("posebusters_filters_passed_count_fast", 0)),
                float(sample_metrics[i].get("error_estimate_0", float("inf"))),
            ),
        )
        return [(rank, sample_metrics[i]) for rank, i in enumerate(ranked_indices, start=1)]

    def _pose_is_physical(sample: dict) -> bool:
        pb_flags = sample.get("posebusters_filters_fast")
        if pb_flags is not None and len(pb_flags) >= 4:
            return all(bool(pb_flags[i]) for i in range(4))
        pb_count = sample.get("posebusters_filters_passed_count_fast")
        if pb_count is not None:
            return int(pb_count) == 4
        return True

    def _save_all_poses_for_uid(metrics_data, uid, out_path, filter_non_physical: bool = True):
        if uid not in metrics_data:
            return [], 0, 0
        sample_data = metrics_data[uid]
        orig_mol = sample_data["orig_mol"]
        ranked = _rank_samples(sample_data["sample_metrics"])

        # Optionally filter out poses failing PoseBusters fast checks
        ranked_filtered = [(r, s) for r, s in ranked if not filter_non_physical or _pose_is_physical(s)]
        ranked_to_use = ranked_filtered if ranked_filtered else ranked

        writer = Chem.SDWriter(str(out_path))
        for rank, sample in ranked_to_use:
            mol = copy.deepcopy(orig_mol)
            conf = Chem.Conformer(orig_mol.GetNumAtoms())
            for idx, (x, y, z) in enumerate(sample["pred_pos"]):
                conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            mol.SetProp("_Name", f"{uid}_rank{rank}")
            writer.write(mol)
        writer.close()
        return ranked_to_use, len(ranked_filtered), len(ranked)
    
    # single mode output
    if not batch_mode:
        resolved_out = run_workdir / f"{run_name}_best.sdf"
        all_poses_dest = run_workdir / f"{run_name}_poses.sdf"

        uid, mdata = next(iter(metrics.items()))
        errs = np.array([float(s.get("error_estimate_0", float("inf"))) for s in mdata["sample_metrics"]])
        pb_counts = np.array([int(s.get("posebusters_filters_passed_count_fast", 0)) for s in mdata["sample_metrics"]])
        def _get_best_sample_idx(errs, pb_counts):
            best_pb_count = max(pb_counts)
            pb_count_indices = np.arange(len(pb_counts))[pb_counts == best_pb_count]
            scores = errs[pb_count_indices]
            best_score_idx = np.argmin(scores)
            return pb_count_indices[best_score_idx]
        best_idx = _get_best_sample_idx(errs, pb_counts)

        pred_sdf_src = preds_root / dataset_name / "sdf_predictions" / f"{run_name}.sdf"
        shutil.copyfile(pred_sdf_src, resolved_out)
        ranked_samples, kept_physical, total_samples = _save_all_poses_for_uid(metrics, uid, all_poses_dest, filter_non_physical=physical_only)

        end_time = datetime.now(timezone.utc)
        runtime = (end_time - start_time).total_seconds()

        log_path = (log or run_workdir / f"{run_name}.log").resolve()
        receptor_abs = receptor.resolve()
        ligand_abs = ligand.resolve()
        resolved_out_abs = resolved_out.resolve()
        all_poses_abs = all_poses_dest.resolve()
        command_line = "uv run matcha " + " ".join(sys.argv[1:]) if ".venv/bin/matcha" in sys.argv[0] else " ".join(sys.argv)
        log_lines = [
            banner.rstrip("\n"),
            "MATCHA DOCKING ENGINE  v0.1.0",
            "============================================================",
            "",
            "",
            "[ RUN INFO ]",
            f"  Start time       : {start_time.isoformat()}Z",
            f"  Command          : {command_line}",
            f"  Workdir          : {run_workdir.resolve()}",
            f"  Runtime          : {_format_runtime(runtime)}",
            "",
            "",
        ]

        if box_center_val is not None:
            log_lines.extend([
                "[ AUTODOCKING BOX ]",
                f"  Mode             : {'manual center' if not autobox_specified else f'autobox from {autobox_ligand.name}'}",
                f"  Center (Å)       : ({box_center_val[0]:.3f}, {box_center_val[1]:.3f}, {box_center_val[2]:.3f})",
                "",
                "",
            ])
        else:
            log_lines.extend([
                "[ DOCKING MODE ]",
                "  Mode             : blind docking (entire protein)",
                "",
                "",
            ])

        log_lines.extend([
            "[ INPUT / OUTPUT FILES ]",
            f"  Receptor         : {receptor_abs}",
            f"  Ligand           : {ligand_abs}",
            f"  Best pose SDF    : {resolved_out_abs}",
            f"  All poses SDF    : {all_poses_abs}",
            f"  Log file         : {log_path}",
            "",
            "",
            "[ SUMMARY ]",
            f"  Samples per ligand     : {n_samples}",
            f"  error_estimate_0 (Å)   : min={min(errs):.3f}, mean={float(np.mean(errs)):.3f}, max={max(errs):.3f}",
            f"  posebusters_fast checks: min={min(pb_counts)}/4, max={max(pb_counts)}/4",
            f"  Best sample            : rank={best_idx+1}, error_estimate_0={errs[best_idx]:.3f}, posebusters_pass_fast={pb_counts[best_idx]}/4",
            f"  Filtered poses (pb_4/4): kept {kept_physical}/{total_samples}" + ("  [WARNING: none passed, keeping originals]" if physical_only and kept_physical==0 else ""),
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

            not_far = "✓" if len(pb_flags) > 0 and pb_flags[0] else "✗"
            no_int_clash = "✓" if len(pb_flags) > 1 and pb_flags[1] else "✗"
            no_clash = "✓" if len(pb_flags) > 2 and pb_flags[2] else "✗"
            no_vol_clash = "✓" if len(pb_flags) > 3 and pb_flags[3] else "✗"
            buried_frac = f"{pb_flags[4]:.2f}" if len(pb_flags) > 4 else "n/a"

            log_lines.append(
                f"  {mode:<3}     {err:>7.3f}       {pb_count}/4      {not_far:^4}     {no_int_clash:^8}     {no_clash:^5}     {no_vol_clash:^8}      {buried_frac:>6}"
            )

        log_lines.extend([
            "",
            "Legend: ✓ = passed, ✗ = failed",
            "",
            "",
        ])

        if any(e > 0 for e in errs):
            log_lines.extend([
                "WARNING: Some poses have positive error_est_0 values.",
                "         This indicates reduced model confidence in those predictions.",
                "         Consider generating more samples or inspecting poses manually.",
                "",
                "",
            ])
        if max(pb_counts) < 4:
            log_lines.extend([
                "WARNING: No poses passed all PoseBusters checks (pb_4/4).",
                "         Inspect poses or regenerate with more samples/box adjustments.",
                "",
                "",
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

        if not keep_workdir:
            shutil.rmtree(work_dir, ignore_errors=True)
            console.print(f"[bold green][matcha][/bold green] cleaned workdir {work_dir}")
        else:
            console.print(f"[bold green][matcha][/bold green] keeping workdir at {work_dir}")
        return

    # batch mode output
    sdf_preds_dir = preds_root / dataset_name / "sdf_predictions"
    molecule_uids = sorted([p.name for p in (work_dir / "datasets" / "any_conf").iterdir() if p.is_dir()])
    receptor_abs = receptor.resolve()
    ligand_dir_abs = ligand_dir.resolve()
    run_workdir_abs = run_workdir.resolve()

    for mol_uid in molecule_uids:
        metrics_key = mol_uid if mol_uid in metrics else f"{mol_uid}_mol0"
        if metrics_key not in metrics:
            console.print(f"[yellow]Warning: No results for {mol_uid}[/yellow]")
            continue
        mdata = metrics[metrics_key]
        errs = np.array([float(s.get("error_estimate_0", float("inf"))) for s in mdata["sample_metrics"]])
        pb_counts = np.array([int(s.get("posebusters_filters_passed_count_fast", 0)) for s in mdata["sample_metrics"]])
        def _get_best_sample_idx(errs, pb_counts):
            best_pb_count = max(pb_counts)
            pb_count_indices = np.arange(len(pb_counts))[pb_counts == best_pb_count]
            scores = errs[pb_count_indices]
            best_score_idx = np.argmin(scores)
            return pb_count_indices[best_score_idx]
        best_idx = _get_best_sample_idx(errs, pb_counts)

        pred_sdf_src = sdf_preds_dir / f"{mol_uid}.sdf"
        best_dest = best_dir / f"{mol_uid}.sdf"
        all_dest = all_dir / f"{mol_uid}_poses.sdf"

        if pred_sdf_src.exists():
            shutil.copyfile(pred_sdf_src, best_dest)
        ranked_samples, kept_physical, total_samples = _save_all_poses_for_uid(
            metrics, metrics_key, all_dest, filter_non_physical=physical_only
        )

        # Per-ligand detailed log (single-style) inside batch run
        ligand_input = (run_workdir / "work" / "datasets" / "any_conf" / mol_uid / f"{mol_uid}_ligand.sdf").resolve()
        per_log_dir = run_workdir / "logs"
        per_log_dir.mkdir(parents=True, exist_ok=True)
        per_log_path = per_log_dir / f"{mol_uid}.log"

        end_time_local = datetime.now(timezone.utc)
        runtime_local = (end_time_local - start_time).total_seconds()

        log_lines = [
            banner.rstrip("\n"),
            "MATCHA DOCKING ENGINE  v0.1.0",
            "============================================================",
            "",
            "",
            "[ RUN INFO ]",
            f"  Start time       : {start_time.isoformat()}Z",
            f"  Command          : batch run {run_name} (ligand {mol_uid})",
            f"  Workdir          : {run_workdir_abs}",
            f"  Runtime          : {_format_runtime(runtime_local)}",
            "",
            "",
        ]

        if box_center_val is not None:
            log_lines.extend([
                "[ AUTODOCKING BOX ]",
                f"  Mode             : {'manual center' if not autobox_specified else f'autobox from {autobox_ligand.name}'}",
                f"  Center (Å)       : ({box_center_val[0]:.3f}, {box_center_val[1]:.3f}, {box_center_val[2]:.3f})",
                "",
                "",
            ])
        else:
            log_lines.extend([
                "[ DOCKING MODE ]",
                "  Mode             : blind docking (entire protein)",
                "",
                "",
            ])

        log_lines.extend([
            "[ INPUT / OUTPUT FILES ]",
            f"  Receptor         : {receptor_abs}",
            f"  Ligand           : {ligand_input}",
            f"  Best pose SDF    : {best_dest.resolve()}",
            f"  All poses SDF    : {all_dest.resolve()}",
            f"  Log file         : {per_log_path.resolve()}",
            "",
            "",
            "[ SUMMARY ]",
            f"  Samples per ligand     : {n_samples}",
            f"  error_estimate_0 (Å)   : min={min(errs):.3f}, mean={float(np.mean(errs)):.3f}, max={max(errs):.3f}",
            f"  posebusters_fast checks: min={min(pb_counts)}/4, max={max(pb_counts)}/4",
            f"  Best sample            : rank={best_idx+1}, error_estimate_0={errs[best_idx]:.3f}, posebusters_pass_fast={pb_counts[best_idx]}/4",
            f"  Filtered poses (pb_4/4): kept {kept_physical}/{total_samples}" + ("  [WARNING: none passed, keeping originals]" if physical_only and kept_physical==0 else ""),
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
            pb_count_val = int(sample.get("posebusters_filters_passed_count_fast", 0))
            pb_flags = sample.get("posebusters_filters_fast", [])

            not_far = "✓" if len(pb_flags) > 0 and pb_flags[0] else "✗"
            no_int_clash = "✓" if len(pb_flags) > 1 and pb_flags[1] else "✗"
            no_clash = "✓" if len(pb_flags) > 2 and pb_flags[2] else "✗"
            no_vol_clash = "✓" if len(pb_flags) > 3 and pb_flags[3] else "✗"
            buried_frac = f"{pb_flags[4]:.2f}" if len(pb_flags) > 4 else "n/a"

            log_lines.append(
                f"  {mode:<3}     {err:>7.3f}       {pb_count_val}/4      {not_far:^4}     {no_int_clash:^8}     {no_clash:^5}     {no_vol_clash:^8}      {buried_frac:>6}"
            )

        log_lines.extend([
            "",
            "Legend: ✓ = passed, ✗ = failed",
            "",
            "",
        ])

        if any(e > 0 for e in errs):
            log_lines.extend([
                "WARNING: Some poses have positive error_est_0 values.",
                "         This indicates reduced model confidence in those predictions.",
                "         Consider generating more samples or inspecting poses manually.",
                "",
                "",
            ])
        if max(pb_counts) < 4:
            log_lines.extend([
                "WARNING: No poses passed all PoseBusters checks (pb_4/4).",
                "         Inspect poses or regenerate with more samples/box adjustments.",
                "",
                "",
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

    log_path = (log or run_workdir / f"{run_name}.log").resolve()
    command_line = "uv run matcha " + " ".join(sys.argv[1:]) if ".venv/bin/matcha" in sys.argv[0] else " ".join(sys.argv)
    receptor_abs = receptor.resolve()
    ligand_dir_abs = ligand_dir.resolve()
    run_workdir_abs = run_workdir.resolve()
    log_lines = [
        banner.rstrip("\n"),
        "MATCHA DOCKING ENGINE  v0.1.0",
        "============================================================",
        "",
        "",
        "[ RUN INFO ]",
        f"  Start time       : {start_time.isoformat()}Z",
        f"  Command          : {command_line}",
        f"  Workdir          : {run_workdir_abs}",
        f"  Runtime          : {_format_runtime(runtime)}",
        "",
        "",
        "[ INPUT FILES ]",
        f"  Receptor         : {receptor_abs}",
        f"  Ligands          : {ligand_dir_abs} ({len(molecule_uids)} molecules)",
        f"  Output dir       : {run_workdir_abs}",
        "",
        "",
        "[ PROCESSING SUMMARY ]",
        f"  Samples per molecule : {n_samples}",
        f"  Total molecules      : {len(molecule_uids)}",
        "",
        "",
        "[ RESULTS ]",
    ]
    for mol_uid in molecule_uids:
        metrics_key = mol_uid if mol_uid in metrics else f"{mol_uid}_mol0"
        if metrics_key not in metrics:
            log_lines.append(f"  {mol_uid}: No results")
            continue
        mdata = metrics[metrics_key]
        errs = np.array([float(s.get("error_estimate_0", float("inf"))) for s in mdata["sample_metrics"]])
        pb_counts = np.array([int(s.get("posebusters_filters_passed_count_fast", 0)) for s in mdata["sample_metrics"]])
        best_idx = _get_best_sample_idx(errs, pb_counts)
        log_lines.append(f"  {mol_uid}: Best err={errs[best_idx]:.3f}, pb={pb_counts[best_idx]}/4")

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

    if keep_workdir:
        console.print(f"[bold green][matcha][/bold green] keeping workdir at {work_dir}")
    else:
        shutil.rmtree(work_dir, ignore_errors=True)
        console.print(f"[bold green][matcha][/bold green] cleaned workdir {work_dir}")


def main() -> None:
    if len(sys.argv) == 1:
        _print_usage_and_exit()
    typer.run(run_matcha)


if __name__ == "__main__":
    main()
