import base64
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import SDWriter
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from tqdm import tqdm

from matcha.scoring.base import PoseScorer
from matcha.utils.log import get_logger

logger = get_logger(__name__)

_GNINA_API_URL = "https://api.github.com/repos/gnina/gnina/releases/latest"
GNINA_CACHE_DIR = Path.home() / ".matcha" / "bin"


def _resolve_gnina_download() -> tuple[str, str]:
    """Resolve the latest GNINA release URL via GitHub API.

    Returns:
        (version, download_url) tuple.

    Raises:
        RuntimeError: If the GitHub API is unreachable or returns no suitable asset.
    """
    try:
        req = urllib.request.Request(_GNINA_API_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
        version = data["tag_name"].lstrip("v")
        # Pick the default build (not cuda12.8)
        for asset in data["assets"]:
            if "cuda" not in asset["name"]:
                return version, asset["browser_download_url"]
    except Exception as exc:
        raise RuntimeError(
            "Failed to query GitHub for the latest GNINA release.\n"
            "Check your internet connection or install GNINA manually:\n"
            "  https://github.com/gnina/gnina/releases"
        ) from exc
    raise RuntimeError("No suitable GNINA binary found in the latest GitHub release.")


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a rich progress bar."""
    response = urllib.request.urlopen(url)  # noqa: S310
    total = int(response.headers.get("Content-Length", 0))

    with (
        Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress,
        open(dest, "wb") as f,
    ):
        task = progress.add_task("Downloading GNINA", total=total)
        while chunk := response.read(8192):
            f.write(chunk)
            progress.advance(task, len(chunk))


def _gnina_env() -> dict:
    """Return env dict with LD_LIBRARY_PATH pointing to CUDA/cuDNN libs.

    GNINA binaries are typically dynamically linked against CUDA + cuDNN.
    On some systems these libraries are available only via Python wheels
    (e.g. PyTorch + nvidia-*-cu12 packages). This helper makes GNINA runnable
    without requiring a system-wide CUDA installation.
    """
    env = os.environ.copy()
    extra_paths: list[str] = []

    # 1. Current env's torch
    try:
        import torch
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.exists():
            extra_paths.append(str(torch_lib))
    except ImportError:
        pass

    # 2. CUDA/cuDNN libs shipped as Python wheels (nvidia-*-cu12).
    try:
        import nvidia  # type: ignore

        roots: list[Path] = []
        # `nvidia` is typically a namespace package, so __file__ may be None.
        if getattr(nvidia, "__path__", None):
            roots.extend([Path(p) for p in nvidia.__path__])
        if getattr(nvidia, "__file__", None):
            roots.append(Path(nvidia.__file__).resolve().parent)

        for nvidia_root in roots:
            for sub in (
                "cudnn",
                "cublas",
                "cuda_runtime",
                "cusparse",
                "cusolver",
                "cufft",
                "nvjitlink",
                "nvtx",
                "cuda_nvrtc",
                "cuda_cupti",
            ):
                lib_dir = nvidia_root / sub / "lib"
                if lib_dir.exists():
                    extra_paths.append(str(lib_dir))
    except Exception:
        pass

    # 3. Conda environments (fallback)
    for base in (
        Path.home() / "miniforge",
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path("/opt/conda"),
    ):
        candidates = sorted(base.glob("**/libcudnn.so*"))
        if candidates:
            extra_paths.append(str(candidates[0].parent))
            break

    if extra_paths:
        deduped = list(dict.fromkeys(extra_paths))
        existing = env.get("LD_LIBRARY_PATH", "")
        new_path = ":".join(deduped)
        env["LD_LIBRARY_PATH"] = f"{new_path}:{existing}" if existing else new_path

    return env


def _is_working_gnina(path: str) -> bool:
    """Check if a gnina binary actually works (not a stub)."""
    try:
        result = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=10,
            env=_gnina_env(),
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError, subprocess.SubprocessError):
        return False


def ensure_gnina() -> str:
    """Find GNINA in PATH / cache, or auto-download it. Returns path to binary."""
    # 1. Already in PATH (verify it actually works, not a stub)
    system_gnina = shutil.which("gnina")
    if system_gnina and _is_working_gnina(system_gnina):
        return system_gnina

    # 2. Previously downloaded
    cached = GNINA_CACHE_DIR / "gnina"
    if cached.exists() and cached.stat().st_size > 1_000_000 and _is_working_gnina(str(cached)):
        return str(cached)

    # 3. Platform check
    if sys.platform != "linux":
        raise RuntimeError(
            "GNINA is only available for Linux.\n"
            "Use --scorer none to skip scoring."
        )

    # 4. Ask user confirmation
    from rich.console import Console

    console = Console()
    console.print(
        "[bold yellow]GNINA not found.[/] "
        "Required for pose scoring (~1.4 GB download)."
    )
    try:
        answer = input("Download GNINA automatically? [Y/n] ").strip().lower()
    except EOFError:
        answer = "n"
    if answer and answer != "y":
        raise RuntimeError(
            "GNINA download declined. Install manually:\n"
            "  conda install -c conda-forge gnina\n"
            "or download from https://github.com/gnina/gnina/releases\n"
            "Alternatively, use --scorer none to skip scoring."
        )

    # 5. Resolve latest version and download
    version, url = _resolve_gnina_download()
    GNINA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = cached.with_suffix(".downloading")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        _download_with_progress(url, tmp_path)
        tmp_path.rename(cached)
        cached.chmod(0o755)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    logger.info(f"GNINA {version} installed to {cached}")
    return str(cached)


def _extract_gnina_score(mol, score_type="CNNscore", use_minimized=True):
    """Extract gnina score from molecule properties.

    Args:
        mol: RDKit molecule object.
        score_type: Type of score ("CNNscore", "CNNaffinity", or "Affinity").
        use_minimized: Whether to look for minimized property names first.

    Returns:
        Score value or None if not found.
    """
    if use_minimized:
        possible_names = [
            f"minimized{score_type}" if score_type != "Affinity" else "minimizedAffinity",
            score_type,
        ]
    else:
        possible_names = [
            score_type,
            f"minimized{score_type}" if score_type != "Affinity" else "minimizedAffinity",
        ]

    for prop_name in possible_names:
        if mol.HasProp(prop_name):
            try:
                return float(mol.GetProp(prop_name))
            except (ValueError, TypeError):
                continue

    # Fallback: search all properties for score-like names
    props = mol.GetPropsAsDict(includePrivate=True, includeComputed=True)
    for key, value in props.items():
        if score_type.lower() in key.lower():
            try:
                return float(value)
            except (ValueError, TypeError):
                continue

    return None


def _normalize_uid(uid: str) -> str:
    # Keep in sync with select_top_poses() uid normalization.
    for suffix in ("_minimized", "_poses", "_predictions"):
        if uid.endswith(suffix):
            return uid[: -len(suffix)]
    return uid


_MATCHA_NAME_PREFIX = "MATCHA"


def _encode_matcha_name(uid: str, pose_idx: int, orig_name: str) -> str:
    uid_b64 = base64.urlsafe_b64encode(uid.encode("utf-8")).decode("ascii")
    orig_b64 = base64.urlsafe_b64encode(orig_name.encode("utf-8")).decode("ascii")
    return f"{_MATCHA_NAME_PREFIX}|{uid_b64}|{pose_idx}|{orig_b64}"


def _decode_matcha_name(name: str) -> tuple[str, int, str]:
    """Decode marker from molecule _Name.

    Supports:
      - New: MATCHA|<uid_b64>|<pose_idx>|<orig_name_b64>
      - Legacy: <uid>::<pose_idx>
    """
    if name.startswith(f"{_MATCHA_NAME_PREFIX}|"):
        parts = name.split("|", 3)
        if len(parts) != 4:
            raise ValueError(f"Invalid MATCHA _Name format: {name!r}")
        _, uid_b64, idx_s, orig_b64 = parts
        uid = base64.urlsafe_b64decode(uid_b64.encode("ascii")).decode("utf-8")
        pose_idx = int(idx_s)
        orig_name = base64.urlsafe_b64decode(orig_b64.encode("ascii")).decode("utf-8")
        return uid, pose_idx, orig_name

    if "::" in name:
        uid, idx_s = name.rsplit("::", 1)
        return uid, int(idx_s), ""

    raise ValueError(f"Missing expected _Name markers in: {name!r}")


def _find_top_scored_molecule(sdf_path, score_type="CNNscore", use_minimized=True,
                              filters_data=None, uid=None, n_samples=40):
    """Read SDF file and find the molecule with the best gnina score.

    Args:
        sdf_path: Path to SDF file containing multiple poses.
        score_type: Type of score to use for ranking.
        use_minimized: Whether to look for minimized property names.
        filters_data: Optional dict with filter results {uid: {filter_field: [values]}}.
        uid: UID for this molecule (required if using filters).
        n_samples: Number of samples per stage.

    Returns:
        Tuple of (best_mol, best_score, best_idx) or None.
    """
    try:
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    except OSError:
        return None

    if use_minimized:
        prop_name = f"minimized{score_type}" if score_type != "Affinity" else "minimizedAffinity"
    else:
        prop_name = score_type

    keep_limit = 3 * int(n_samples)

    mols = []
    scores = []
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        if i >= keep_limit:
            break

        # Try direct property access first
        if mol.HasProp(prop_name):
            try:
                score = float(mol.GetProp(prop_name))
                mols.append(mol)
                scores.append(score)
                continue
            except (ValueError, TypeError):
                pass

        # Fall back to search
        score = _extract_gnina_score(mol, score_type, use_minimized)
        if score is not None:
            mols.append(mol)
            scores.append(score)

    if len(mols) == 0:
        return None

    scores = np.array(scores)

    # Apply filters if provided
    valid_indices = np.arange(len(mols))
    if filters_data is not None and uid is not None and uid in filters_data:
        filter_counts = np.array(
            filters_data[uid]['posebusters_filters_passed_count_fast']
        )
        # Trim to keep_mask length if needed
        if len(filter_counts) > keep_limit:
            filter_counts = filter_counts[:keep_limit]

        if len(filter_counts) != len(mols):
            logger.warning(
                f"Filter count mismatch for {uid}: "
                f"{len(filter_counts)} filters vs {len(mols)} molecules"
            )
        else:
            max_filters_passed = np.max(filter_counts)
            valid_mask = filter_counts >= max_filters_passed
            valid_indices = valid_indices[valid_mask]

            if len(valid_indices) == 0:
                logger.warning(f"No molecules pass filters for {uid}, using all")
                valid_indices = np.arange(len(mols))

    valid_scores = scores[valid_indices]

    # For Affinity, lower is better; for CNNscore/CNNaffinity, higher is better
    if score_type in ("Affinity", "minimizedAffinity"):
        best_valid_idx = np.argmin(valid_scores)
    else:
        best_valid_idx = np.argmax(valid_scores)

    best_idx = valid_indices[best_valid_idx]
    return mols[best_idx], float(scores[best_idx]), best_idx


class GninaScorer(PoseScorer):
    """Scorer using the GNINA molecular docking program."""

    def __init__(self, gnina_path=None, minimize=True, score_type="Affinity",
                 cnn_scoring="none"):
        if gnina_path is not None:
            self._gnina_path = str(gnina_path)
        else:
            self._gnina_path = ensure_gnina()
        self.minimize = minimize
        self.score_type = score_type
        self.cnn_scoring = cnn_scoring

    @property
    def name(self) -> str:
        return "gnina"

    @property
    def gnina_path(self) -> str:
        return self._gnina_path

    def score_poses(self, receptor_path, sdf_input_dir, sdf_output_dir, device=0):
        sdf_input_dir = Path(sdf_input_dir)
        sdf_output_dir = Path(sdf_output_dir)
        sdf_output_dir.mkdir(parents=True, exist_ok=True)

        sdf_files = sorted(sdf_input_dir.glob("*.sdf"))
        if not sdf_files:
            logger.warning(f"No SDF files found in {sdf_input_dir}")
            return

        logger.info(f"Scoring {len(sdf_files)} SDF files with gnina "
                     f"(minimize={self.minimize})")

        for sdf_file in tqdm(sdf_files, desc="GNINA scoring"):
            output_sdf = sdf_output_dir / sdf_file.name
            cmd = [
                self.gnina_path,
                "--receptor", str(receptor_path),
                "--ligand", str(sdf_file),
                "--device", str(device),
                "--cnn_scoring", self.cnn_scoring,
            ]
            if self.minimize:
                cmd.append("--minimize")
            else:
                cmd.append("--score_only")
            cmd.extend(["-o", str(output_sdf)])

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True,
                               env=_gnina_env())
            except subprocess.CalledProcessError as e:
                logger.error(f"gnina failed for {sdf_file.name}: {e.stderr}")
            except FileNotFoundError:
                raise RuntimeError(
                    f"gnina binary not found at {self.gnina_path}. "
                    "Please install gnina or use --scorer none."
                )

    def score_poses_combined(
        self,
        receptor_path,
        sdf_input_dir,
        sdf_output_dir,
        best_output_dir,
        filters_path=None,
        n_samples=40,
        device=0,
    ):
        """Score all per-ligand SDFs with a single GNINA invocation.

        This reduces process startup overhead by:
          1) concatenating all ligand pose SDFs into one combined SDF,
          2) running GNINA once,
          3) splitting results back into per-uid scored SDFs and selecting best poses.
        """
        sdf_input_dir = Path(sdf_input_dir)
        sdf_output_dir = Path(sdf_output_dir)
        best_output_dir = Path(best_output_dir)
        sdf_output_dir.mkdir(parents=True, exist_ok=True)
        best_output_dir.mkdir(parents=True, exist_ok=True)

        sdf_files = sorted(sdf_input_dir.glob("*.sdf"))
        if not sdf_files:
            logger.warning(f"No SDF files found in {sdf_input_dir}")
            return

        # Avoid collisions with possible ligand uids and parallel runs.
        pid = os.getpid()
        combined_in = sdf_output_dir / f".matcha_combined_input.{pid}.sdf"
        combined_out = sdf_output_dir / f".matcha_combined_scored.{pid}.sdf"

        logger.info(f"Building combined SDF with {len(sdf_files)} ligands: {combined_in}")
        writer = SDWriter(str(combined_in))
        writer.SetKekulize(False)
        for sdf_file in tqdm(sdf_files, desc="Preparing combined SDF"):
            uid = sdf_file.stem
            try:
                supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
            except OSError as exc:
                logger.warning(f"Skipping invalid SDF input for GNINA: {sdf_file} ({exc})")
                continue
            for idx, mol in enumerate(supplier):
                if mol is None:
                    continue
                orig_name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                mol.SetProp("_Name", _encode_matcha_name(uid, idx, orig_name))
                writer.write(mol)
        writer.close()

        cmd = [
            self.gnina_path,
            "--receptor", str(receptor_path),
            "--ligand", str(combined_in),
            "--device", str(device),
            "--cnn_scoring", self.cnn_scoring,
        ]
        if self.minimize:
            cmd.append("--minimize")
        else:
            cmd.append("--score_only")
        cmd.extend(["-o", str(combined_out)])

        logger.info(f"Running combined GNINA scoring (minimize={self.minimize})")
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, env=_gnina_env())
        except subprocess.CalledProcessError as e:
            logger.error(f"gnina failed (combined): {e.stderr}")
            raise RuntimeError("GNINA scoring failed") from e
        except FileNotFoundError:
            raise RuntimeError(
                f"gnina binary not found at {self.gnina_path}. "
                "Please install gnina or use --scorer none."
            )

        filters_data = None
        if filters_path is not None:
            filters_path = Path(filters_path)
            if filters_path.exists():
                logger.info(f"Loading filters from {filters_path}")
                with open(filters_path) as f:
                    filters_data = json.load(f)
            else:
                logger.warning(f"Filters file not found: {filters_path}")

        def _score_better(a: float, b: float) -> bool:
            if self.score_type in ("Affinity", "minimizedAffinity"):
                return a < b
            return a > b

        keep_limit = 3 * int(n_samples)
        supplier = Chem.SDMolSupplier(str(combined_out), removeHs=False, sanitize=False)

        def _write_uid_outputs(uid_raw: str, entries: list[tuple[int, Chem.Mol, str]]) -> None:
            uid_norm = _normalize_uid(uid_raw)

            # Determine filter threshold (match select_top_poses behavior).
            counts = []
            max_filters = None
            if filters_data is not None and uid_norm in filters_data:
                counts = filters_data[uid_norm].get("posebusters_filters_passed_count_fast", [])
                if counts:
                    max_filters = max(counts[: min(len(counts), keep_limit)])

            entries.sort(key=lambda t: t[0])

            # Write scored per-uid SDF (all poses), preserving original names.
            scored_path = sdf_output_dir / f"{uid_raw}.sdf"
            w = SDWriter(str(scored_path))
            w.SetKekulize(False)
            try:
                for _, mol, orig_name in entries:
                    mol.SetProp("_Name", orig_name)
                    w.write(mol)
            finally:
                w.close()

            # Select and write best pose (first keep_limit, filtered by PB counts).
            best_mol_local = None
            best_score_local = None
            best_orig_name = ""
            for pose_idx, mol, orig_name in entries:
                if pose_idx >= keep_limit:
                    continue
                if max_filters is not None:
                    if pose_idx >= len(counts):
                        continue
                    if int(counts[pose_idx]) < int(max_filters):
                        continue
                score = _extract_gnina_score(mol, score_type=self.score_type, use_minimized=self.minimize)
                if score is None:
                    continue
                if best_score_local is None or _score_better(score, best_score_local):
                    best_score_local = score
                    best_mol_local = mol
                    best_orig_name = orig_name

            if best_mol_local is not None:
                best_mol_local.SetProp("_Name", best_orig_name)
                best_path = best_output_dir / f"{uid_norm}.sdf"
                bw = SDWriter(str(best_path))
                try:
                    bw.SetKekulize(False)
                    bw.write(best_mol_local)
                finally:
                    bw.close()

        logger.info("Splitting combined GNINA output and selecting best poses")
        current_uid = None
        current_entries: list[tuple[int, Chem.Mol, str]] = []
        seen_uids = set()
        non_contiguous = False

        for mol in tqdm(supplier, desc="Processing combined GNINA output"):
            if mol is None:
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            try:
                uid, pose_idx, orig_name = _decode_matcha_name(name)
            except Exception as e:
                raise RuntimeError(
                    "Combined GNINA output is missing expected _Name markers. "
                    "Use --gnina-batch-mode per-ligand."
                ) from e

            if current_uid is None:
                current_uid = uid
                seen_uids.add(uid)

            if uid != current_uid:
                if uid in seen_uids:
                    non_contiguous = True
                    break
                _write_uid_outputs(current_uid, current_entries)
                current_uid = uid
                seen_uids.add(uid)
                current_entries = []

            current_entries.append((pose_idx, mol, orig_name))

        if not non_contiguous and current_uid is not None:
            _write_uid_outputs(current_uid, current_entries)

        if non_contiguous:
            logger.warning(
                "Non-contiguous uids detected in combined GNINA output; "
                "re-grouping output by uid (slower)."
            )
            uid_to_entries = defaultdict(list)
            supplier2 = Chem.SDMolSupplier(str(combined_out), removeHs=False, sanitize=False)
            for mol in supplier2:
                if mol is None:
                    continue
                name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                uid, pose_idx, orig_name = _decode_matcha_name(name)
                uid_to_entries[uid].append((pose_idx, mol, orig_name))
            for uid, entries in uid_to_entries.items():
                _write_uid_outputs(uid, entries)

    def select_top_poses(self, sdf_dir, output_dir, filters_path=None, n_samples=20):
        sdf_dir = Path(sdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sdf_files = sorted(sdf_dir.glob("*.sdf"))
        if not sdf_files:
            logger.warning(f"No SDF files found in {sdf_dir}")
            return

        # Load filters if provided
        filters_data = None
        if filters_path is not None:
            filters_path = Path(filters_path)
            if filters_path.exists():
                logger.info(f"Loading filters from {filters_path}")
                with open(filters_path) as f:
                    filters_data = json.load(f)
            else:
                logger.warning(f"Filters file not found: {filters_path}")

        use_minimized = self.minimize
        successful = 0
        failed = 0

        for sdf_file in tqdm(sdf_files, desc="Selecting top poses"):
            uid = _normalize_uid(sdf_file.stem)

            result = _find_top_scored_molecule(
                sdf_file,
                score_type=self.score_type,
                use_minimized=use_minimized,
                filters_data=filters_data,
                uid=uid,
                n_samples=n_samples,
            )

            if result is None:
                logger.warning(f"No valid scored molecules in {sdf_file.name}")
                failed += 1
                continue

            best_mol, best_score, best_idx = result
            writer = SDWriter(str(output_dir / f"{uid}.sdf"))
            writer.SetKekulize(False)
            writer.write(best_mol)
            writer.close()
            successful += 1

        logger.info(f"Top pose selection: {successful} successful, {failed} failed")


class CustomScriptScorer(PoseScorer):
    """Scorer using a custom external script."""

    def __init__(self, script_path):
        self._script_path = str(script_path)
        if not Path(self._script_path).exists():
            raise FileNotFoundError(f"Scorer script not found: {script_path}")

    @property
    def name(self) -> str:
        return f"custom:{Path(self._script_path).name}"

    def score_poses(self, receptor_path, sdf_input_dir, sdf_output_dir, device=0):
        sdf_input_dir = Path(sdf_input_dir)
        sdf_output_dir = Path(sdf_output_dir)
        sdf_output_dir.mkdir(parents=True, exist_ok=True)

        sdf_files = sorted(sdf_input_dir.glob("*.sdf"))
        if not sdf_files:
            logger.warning(f"No SDF files found in {sdf_input_dir}")
            return

        logger.info(f"Scoring {len(sdf_files)} SDF files with {self.name}")

        for sdf_file in tqdm(sdf_files, desc=f"Scoring ({self.name})"):
            output_sdf = sdf_output_dir / sdf_file.name
            cmd = [
                self._script_path,
                "--receptor", str(receptor_path),
                "--ligand-sdf", str(sdf_file),
                "--output-sdf", str(output_sdf),
                "--device", str(device),
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Custom scorer failed for {sdf_file.name}: {e.stderr}")

    def select_top_poses(self, sdf_dir, output_dir, filters_path=None, n_samples=40):
        """Select top poses - uses same logic as GninaScorer (reads SDF properties)."""
        sdf_dir = Path(sdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sdf_files = sorted(sdf_dir.glob("*.sdf"))
        if not sdf_files:
            logger.warning(f"No SDF files found in {sdf_dir}")
            return

        filters_data = None
        if filters_path is not None:
            filters_path = Path(filters_path)
            if filters_path.exists():
                with open(filters_path) as f:
                    filters_data = json.load(f)

        successful = 0
        failed = 0

        for sdf_file in tqdm(sdf_files, desc="Selecting top poses"):
            uid = _normalize_uid(sdf_file.stem)

            result = _find_top_scored_molecule(
                sdf_file,
                score_type="CNNscore",
                use_minimized=False,
                filters_data=filters_data,
                uid=uid,
                n_samples=n_samples,
            )

            if result is None:
                failed += 1
                continue

            best_mol, _, _ = result
            writer = SDWriter(str(output_dir / f"{uid}.sdf"))
            writer.SetKekulize(False)
            writer.write(best_mol)
            writer.close()
            successful += 1

        logger.info(f"Top pose selection: {successful} successful, {failed} failed")


def create_scorer(scorer_type, scorer_path=None, minimize=True):
    """Factory function to create a PoseScorer instance.

    Args:
        scorer_type: "gnina" or "custom".
        scorer_path: Path to gnina binary or custom script.
        minimize: Whether to minimize poses (gnina only).

    Returns:
        PoseScorer instance.
    """
    if scorer_type == "gnina":
        return GninaScorer(gnina_path=scorer_path, minimize=minimize)
    if scorer_type == "custom":
        if scorer_path is None:
            raise ValueError("--scorer-path is required for custom scorer")
        return CustomScriptScorer(script_path=scorer_path)
    raise ValueError(f"Unknown scorer type: {scorer_type}")
