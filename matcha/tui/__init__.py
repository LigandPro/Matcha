"""
Matcha TUI - Terminal User Interface for molecular docking.

This module provides a JSON-RPC backend for the Node.js/Ink frontend.
"""

from matcha.tui.backend import run_backend

__all__ = ["run_backend", "main"]


def main() -> None:
    """Entry point for matcha-tui-backend command."""
    run_backend()
