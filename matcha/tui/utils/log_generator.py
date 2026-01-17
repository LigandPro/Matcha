"""
Log file generator for docking runs.
Provides reusable methods for generating structured log files.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime


BANNER = r"""
███╗   ███╗ █████╗ ████████╗ ██████╗██╗  ██╗ █████╗
████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║  ██║██╔══██╗
██╔████╔██║███████║   ██║   ██║     ███████║███████║
██║╚██╔╝██║██╔══██║   ██║   ██║     ██╔══██║██╔══██║
██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗██║  ██║██║  ██║
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝
"""


def _format_runtime(seconds: float) -> str:
    """Format runtime in human-readable format."""
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


def _get_best_sample_idx(errs, pb_counts) -> int:
    """
    Get index of best sample based on PoseBusters count and error estimate.

    Args:
        errs: Array of error estimates
        pb_counts: Array of PoseBusters check counts

    Returns:
        Index of best sample
    """
    best_pb_count = max(pb_counts)
    pb_count_indices = [i for i, v in enumerate(pb_counts) if v == best_pb_count]
    best_score_idx = min(pb_count_indices, key=lambda i: errs[i])
    return best_score_idx


class LogFileGenerator:
    """Utility class for generating structured log files."""

    @staticmethod
    def write_banner(log_lines: List[str], title: str = "MATCHA DOCKING ENGINE  v0.1.0", width: int = 80) -> None:
        """
        Write banner section to log lines.

        Args:
            log_lines: List to append log lines to
            title: Title to display after banner
            width: Width of separator line
        """
        log_lines.extend([
            BANNER.rstrip("\n"),
            title,
            "=" * width,
            "",
            "",
        ])

    @staticmethod
    def write_run_info(
        log_lines: List[str],
        start_time: datetime,
        command: str,
        workdir: Path,
        runtime: float,
    ) -> None:
        """
        Write run information section.

        Args:
            log_lines: List to append log lines to
            start_time: Job start datetime
            command: Command line used
            workdir: Working directory path
            runtime: Runtime in seconds
        """
        log_lines.extend([
            "[ RUN INFO ]",
            f"  Start time       : {start_time.isoformat()}Z",
            f"  Command          : {command}",
            f"  Workdir          : {workdir.resolve()}",
            f"  Runtime          : {_format_runtime(runtime)}",
            "",
            "",
        ])

    @staticmethod
    def write_box_info(
        log_lines: List[str],
        box_mode: str,
        center: Optional[Tuple[float, float, float]] = None,
        autobox_ligand: Optional[str] = None,
    ) -> None:
        """
        Write autodocking box information.

        Args:
            log_lines: List to append log lines to
            box_mode: Box mode ('manual', 'autobox', or 'blind')
            center: Center coordinates for manual/autobox mode
            autobox_ligand: Ligand file used for autobox
        """
        if box_mode == "manual":
            log_lines.extend([
                "[ AUTODOCKING BOX ]",
                "  Mode             : manual center",
                f"  Center (Å)       : ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})",
                "",
                "",
            ])
        elif box_mode == "autobox" and autobox_ligand:
            log_lines.extend([
                "[ AUTODOCKING BOX ]",
                f"  Mode             : autobox from {Path(autobox_ligand).name}",
                "",
                "",
            ])
            if center is not None:
                log_lines.insert(-2, f"  Center (Å)       : ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        else:
            log_lines.extend([
                "[ DOCKING MODE ]",
                "  Mode             : blind docking (entire protein)",
                "",
                "",
            ])

    @staticmethod
    def write_input_output_files(
        log_lines: List[str],
        receptor: Path,
        ligand_input: str,
        best_dest: Path,
        all_dest: Path,
        log_path: Path,
    ) -> None:
        """
        Write input/output files section.

        Args:
            log_lines: List to append log lines to
            receptor: Receptor file path
            ligand_input: Ligand input description
            best_dest: Best pose output path
            all_dest: All poses output path
            log_path: Log file path
        """
        log_lines.extend([
            "[ INPUT / OUTPUT FILES ]",
            f"  Receptor         : {receptor.resolve()}",
            f"  Ligand           : {ligand_input}",
            f"  Best pose SDF    : {best_dest.resolve()}",
            f"  All poses SDF    : {all_dest.resolve()}",
            f"  Log file         : {log_path.resolve()}",
            "",
            "",
        ])

    @staticmethod
    def write_summary(
        log_lines: List[str],
        n_samples: int,
        errs: List[float],
        pb_counts: List[int],
        best_idx: int,
        physical_only: bool,
        kept_physical: int,
        total_samples: int,
    ) -> None:
        """
        Write summary section with metrics.

        Args:
            log_lines: List to append log lines to
            n_samples: Number of samples per ligand
            errs: List of error estimates
            pb_counts: List of PoseBusters check counts
            best_idx: Index of best sample
            physical_only: Whether filtering for physical poses only
            kept_physical: Number of poses kept after filtering
            total_samples: Total number of samples
        """
        import numpy as np

        warning_msg = ""
        if physical_only and kept_physical == 0:
            warning_msg = "  [WARNING: none passed, keeping originals]"

        log_lines.extend([
            "[ SUMMARY ]",
            f"  Samples per ligand     : {n_samples}",
            f"  error_estimate_0 (Å)   : min={min(errs):.3f}, mean={float(np.mean(errs)):.3f}, max={max(errs):.3f}",
            f"  posebusters_fast checks: min={min(pb_counts)}/4, max={max(pb_counts)}/4",
            f"  Best sample            : rank={best_idx + 1}, error_estimate_0={errs[best_idx]:.3f}, posebusters_pass_fast={pb_counts[best_idx]}/4",
            f"  Filtered poses (pb_4/4): kept {kept_physical}/{total_samples}{warning_msg}",
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
        ])

    @staticmethod
    def write_pose_table(
        log_lines: List[str],
        ranked_samples: List[Tuple[str, Dict[str, Any]]],
    ) -> None:
        """
        Write pose ranking table.

        Args:
            log_lines: List to append log lines to
            ranked_samples: List of (mode, sample_data) tuples sorted by rank
        """
        log_lines.extend([
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

    @staticmethod
    def write_warnings(
        log_lines: List[str],
        errs: List[float],
        pb_counts: List[int],
    ) -> None:
        """
        Write warning messages based on metrics.

        Args:
            log_lines: List to append log lines to
            errs: List of error estimates
            pb_counts: List of PoseBusters check counts
        """
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

    @staticmethod
    def write_end(
        log_lines: List[str],
        end_time: datetime,
        runtime: float,
        workdir: Path,
    ) -> None:
        """
        Write end section with final information.

        Args:
            log_lines: List to append log lines to
            end_time: Job end datetime
            runtime: Total runtime in seconds
            workdir: Working directory path
        """
        log_lines.extend([
            "[ END ]",
            f"  Run finished at    : {end_time.isoformat()}Z",
            f"  Total runtime      : {_format_runtime(runtime)}",
            f"  Workdir preserved  : {workdir.resolve()}",
            "============================================================",
        ])

    @staticmethod
    def write_batch_input_files(
        log_lines: List[str],
        receptor: Path,
        ligand_dir: Path,
        output_dir: Path,
        num_molecules: int,
    ) -> None:
        """
        Write input files section for batch mode.

        Args:
            log_lines: List to append log lines to
            receptor: Receptor file path
            ligand_dir: Ligand directory path
            output_dir: Output directory path
            num_molecules: Number of molecules being processed
        """
        log_lines.extend([
            "[ INPUT FILES ]",
            f"  Receptor         : {receptor.resolve()}",
            f"  Ligands          : {ligand_dir.resolve()} ({num_molecules} molecules)",
            f"  Output dir       : {output_dir.resolve()}",
            "",
            "",
        ])

    @staticmethod
    def write_batch_summary(
        log_lines: List[str],
        n_samples: int,
        num_molecules: int,
    ) -> None:
        """
        Write processing summary for batch mode.

        Args:
            log_lines: List to append log lines to
            n_samples: Samples per molecule
            num_molecules: Total number of molecules
        """
        log_lines.extend([
            "[ PROCESSING SUMMARY ]",
            f"  Samples per molecule : {n_samples}",
            f"  Total molecules      : {num_molecules}",
            "",
            "",
            "[ RESULTS ]",
        ])

    @staticmethod
    def write_batch_results(
        log_lines: List[str],
        molecule_uids: List[str],
        metrics: Dict[str, Any],
    ) -> None:
        """
        Write results section for batch mode.

        Args:
            log_lines: List to append log lines to
            molecule_uids: List of molecule UIDs
            metrics: Dictionary of metrics per molecule
        """
        import numpy as np

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

        log_lines.append("")
