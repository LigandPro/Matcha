from abc import ABC, abstractmethod


class PoseScorer(ABC):
    """Abstract base class for pose scoring."""

    @abstractmethod
    def score_poses(self, receptor_path, sdf_input_dir, sdf_output_dir, device=0):
        """Score all SDF files in input_dir, write scored SDF files to output_dir.

        Args:
            receptor_path: Path to receptor PDB file.
            sdf_input_dir: Directory containing input SDF files (multi-conformer).
            sdf_output_dir: Directory to write scored SDF files.
            device: GPU device index.
        """

    @abstractmethod
    def select_top_poses(self, sdf_dir, output_dir, filters_path=None, n_samples=40):
        """Select best pose from each SDF based on scores and optional filters.

        Args:
            sdf_dir: Directory containing scored SDF files.
            output_dir: Directory to write best-pose SDF files (one per ligand).
            filters_path: Optional path to filters_results.json with PB filter counts.
            n_samples: Number of samples per stage (used for keep_mask calculation).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Scorer name for logging."""
