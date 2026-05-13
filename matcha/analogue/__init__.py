"""Template-aware analogue docking helpers for Matcha.

The analogue package implements the lightweight, RDKit-based front-end used to
turn a bound template ligand plus a congeneric ligand series into aligned seed
poses and FEP/RBFE-friendly export files.  The neural Matcha refinement stage can
then consume those seed poses through the existing predicted-transform path.
"""

from .mcs import MCSMapping, find_robust_mcs
from .workflow import AnalogueWorkflowConfig, AnalogueWorkflowResult, run_analogue_workflow

__all__ = [
    "MCSMapping",
    "find_robust_mcs",
    "AnalogueWorkflowConfig",
    "AnalogueWorkflowResult",
    "run_analogue_workflow",
]
