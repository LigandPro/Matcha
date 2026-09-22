import subprocess
from pathlib import Path


def resolve_real_dataset_name(dataset_name: str) -> str:
    """Map benchmark dataset folder name to preprocessed receptor layout prefix.

    >>> resolve_real_dataset_name("dockgen_full")
    'dockgen'
    >>> resolve_real_dataset_name("pdbbind_conf")
    'pdbbind'
    """
    if dataset_name.startswith("dockgen"):
        return "dockgen"
    if "_conf" in dataset_name:
        return dataset_name.split("_conf")[0]
    return dataset_name


def resolve_gnina_receptor_path(
    *,
    dataset_name: str,
    uid: str,
    receptors_folder: Path,
) -> Path:
    """Resolve receptor PDB under ``preprocessed_receptors_base``."""
    real_ds_name = resolve_real_dataset_name(dataset_name)
    return receptors_folder / f"{real_ds_name}_{uid}" / f"{real_ds_name}_{uid}_protein.pdb"


def count_molecules_in_sdf(sdf_path: Path) -> int:
    """Count records via SDF $$$$ delimiters (same heuristic as the old shell script)."""
    count = 0
    with open(sdf_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == "$$$$":
                count += 1
    return 1 if count == 0 else count


def build_gnina_command(
    gnina_script: Path,
    receptor_path: Path,
    ligand_path: Path,
    output_path: Path,
    minimize: bool,
    device: str,
) -> list[str]:
    """Build GNINA argv (minimize or score_only; CNN device)."""
    cmd: list[str] = [
        str(gnina_script),
        "--receptor",
        str(receptor_path),
        "--ligand",
        str(ligand_path),
        "-o",
        str(output_path),
    ]
    if minimize:
        cmd.extend(["--minimize"])
    else:
        cmd.extend(["--score_only"])
    cmd.extend(["--device", device])
    return cmd


def run_gnina_subprocess(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one GNINA process."""
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
