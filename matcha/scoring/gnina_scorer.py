import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import SDWriter
from tqdm import tqdm

from matcha.scoring.base import PoseScorer
from matcha.utils.log import get_logger

logger = get_logger(__name__)


def _extract_gnina_score(mol, score_type="CNNscore", use_minimized=True):
    """Extract gnina score from molecule properties.

    Args:
        mol: RDKit molecule object.
        score_type: Type of score ("CNNscore", "CNNaffinity", or "Affinity").
        use_minimized: Whether to look for minimized property names first.

    Returns:
        Score value or None if not found.
    """
    minimized_name = f"minimized{score_type}"
    if use_minimized:
        possible_names = [minimized_name, score_type]
    else:
        possible_names = [score_type, minimized_name]

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
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)

    prop_name = f"minimized{score_type}" if use_minimized else score_type

    keep_mask = np.arange(3 * n_samples)

    mols = []
    scores = []
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        if i not in keep_mask:
            continue

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

    if not mols:
        return None

    scores = np.array(scores)

    # Apply filters if provided
    valid_indices = np.arange(len(mols))
    if filters_data is not None and uid is not None and uid in filters_data:
        filter_counts = np.array(
            filters_data[uid]['posebusters_filters_passed_count_fast']
        )
        # Trim to keep_mask length if needed
        if len(filter_counts) > len(keep_mask):
            filter_counts = filter_counts[keep_mask]

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
        self._gnina_path = str(gnina_path) if gnina_path is not None else shutil.which("gnina")
        self.minimize = minimize
        self.score_type = score_type
        self.cnn_scoring = cnn_scoring

    @property
    def name(self) -> str:
        return "gnina"

    @property
    def gnina_path(self) -> str:
        if self._gnina_path is None:
            raise RuntimeError(
                "gnina not found in PATH. Install via:\n"
                "  conda install -c conda-forge gnina\n"
                "or download from https://github.com/gnina/gnina/releases\n"
                "Alternatively, use --scorer none to skip scoring."
            )
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
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"gnina failed for {sdf_file.name}: {e.stderr}")
            except FileNotFoundError:
                raise RuntimeError(
                    f"gnina binary not found at {self.gnina_path}. "
                    "Please install gnina or use --scorer none."
                )

    def select_top_poses(self, sdf_dir, output_dir, filters_path=None, n_samples=40):
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
            uid = sdf_file.stem
            # Remove common suffixes
            for suffix in ("_minimized", "_poses", "_predictions"):
                if uid.endswith(suffix):
                    uid = uid[:-len(suffix)]
                    break

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
            uid = sdf_file.stem

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
    elif scorer_type == "custom":
        if scorer_path is None:
            raise ValueError("--scorer-path is required for custom scorer")
        return CustomScriptScorer(script_path=scorer_path)
    else:
        raise ValueError(f"Unknown scorer type: {scorer_type}")
