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


def _get_best_sample_idx(pb_counts, gnina_scores=None) -> int:
    """Get index of best sample based on PB count and optional GNINA affinity.

    Args:
        pb_counts: Array of PoseBusters check counts
        gnina_scores: Optional array of GNINA affinity scores (lower is better)

    Returns:
        Index of best sample
    """
    import numpy as np
    pb_counts = np.asarray(pb_counts)
    best_pb_count = max(pb_counts)
    pb_count_indices = np.arange(len(pb_counts))[pb_counts == best_pb_count]
    if gnina_scores is not None:
        gnina_scores = np.asarray(gnina_scores)
        scores = gnina_scores[pb_count_indices]
        best_score_idx = np.argmin(scores)
    else:
        best_score_idx = 0
    return pb_count_indices[best_score_idx]


class LogFileGenerator:
    """Utility class for generating structured log files."""

    @staticmethod
    def write_banner(log_lines: List[str], title: str = "MATCHA DOCKING ENGINE  v2.0.0", width: int = 80) -> None:
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
        pb_counts: List[int],
        best_idx: int,
        physical_only: bool,
        kept_physical: int,
        total_samples: int,
        scorer_type: str = "none",
        scorer_name: str = "",
        gnina_scores: Optional[List[float]] = None,
    ) -> None:
        import numpy as np

        warning_msg = ""
        if physical_only and kept_physical == 0:
            warning_msg = "  [WARNING: none passed, keeping originals]"

        affinity_str = ""
        if gnina_scores is not None:
            affinity_str = f", affinity={gnina_scores[best_idx]:.2f}"

        lines = [
            "[ SUMMARY ]",
            f"  Samples per ligand     : {n_samples}",
            f"  Scorer                 : {scorer_type}" + (f" ({scorer_name})" if scorer_name else ""),
        ]
        if gnina_scores is not None:
            lines.append(f"  GNINA Affinity (kcal/mol): min={min(gnina_scores):.2f}, mean={float(np.mean(gnina_scores)):.2f}, max={max(gnina_scores):.2f}")
        lines.extend([
            f"  PoseBusters checks     : min={min(pb_counts)}/4, max={max(pb_counts)}/4",
            f"  Best sample            : rank={best_idx + 1}, pb={pb_counts[best_idx]}/4{affinity_str}",
            f"  Filtered poses (pb_4/4): kept {kept_physical}/{total_samples}{warning_msg}",
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
        log_lines.extend(lines)

    @staticmethod
    def write_pose_table(
        log_lines: List[str],
        ranked_samples: List[Tuple[str, Dict[str, Any]]],
        has_gnina: bool = False,
    ) -> None:
        if has_gnina:
            log_lines.extend([
                "[ POSE RANKING ]",
                "  rank  affinity  pb  checks  buried_frac",
                "  -----------------------------------------",
            ])
        else:
            log_lines.extend([
                "[ POSE RANKING ]",
                "  rank  pb  checks  buried_frac",
                "  ------------------------------",
            ])

        for mode, sample in ranked_samples:
            pb_count_val = int(sample.get("posebusters_filters_passed_count_fast", 0))
            pb_flags = sample.get("posebusters_filters_fast", [])
            checks = "".join(
                "✓" if len(pb_flags) > j and pb_flags[j] else "✗"
                for j in range(4)
            )
            buried_frac = f"{pb_flags[4]:.2f}" if len(pb_flags) > 4 else " n/a"
            if has_gnina:
                aff = f"{sample.get('gnina_score', float('inf')):>8.2f}"
                log_lines.append(f"  {mode:<4}  {aff}  {pb_count_val}/4  {checks}   {buried_frac:>6}")
            else:
                log_lines.append(f"  {mode:<4}  {pb_count_val}/4  {checks}   {buried_frac:>6}")

        log_lines.extend([
            "",
            "Legend: checks = not_far | no_int_clash | no_clash | no_vol_clash (✓=pass ✗=fail)",
            "",
            "",
        ])

    @staticmethod
    def write_warnings(
        log_lines: List[str],
        pb_counts: List[int],
    ) -> None:
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
        import numpy as np

        for mol_uid in molecule_uids:
            metrics_key = mol_uid if mol_uid in metrics else f"{mol_uid}_mol0"
            if metrics_key not in metrics:
                log_lines.append(f"  {mol_uid}: No results")
                continue
            mdata = metrics[metrics_key]
            pb_counts = np.array([int(s.get("posebusters_filters_passed_count_fast", 0)) for s in mdata["sample_metrics"]])
            has_gnina = any("gnina_score" in s for s in mdata["sample_metrics"])
            gnina_scores = None
            if has_gnina:
                gnina_scores = np.array([float(s.get("gnina_score", float("inf"))) for s in mdata["sample_metrics"]])
            best_idx = _get_best_sample_idx(pb_counts, gnina_scores)
            line = f"  {mol_uid}: pb={pb_counts[best_idx]}/4"
            if has_gnina:
                line += f", affinity={gnina_scores[best_idx]:.2f}"
            log_lines.append(line)

        log_lines.append("")
