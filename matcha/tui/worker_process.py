"""
Worker process entrypoint for Matcha TUI docking jobs.

This module is launched by the JSON-RPC backend as a separate OS process per job.
It reads a single JSON object from stdin (the docking config), runs the docking
pipeline, and prints newline-delimited JSON progress events to stdout.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from dataclasses import asdict
from typing import Any, Callable

from matcha.tui.protocol import ProgressEvent


class WorkerJob:
    """Minimal job object compatible with matcha.tui.docking_worker.run_docking()."""

    def __init__(self, job_id: str, config: dict[str, Any], progress_callback: Callable[[ProgressEvent], None]):
        self.job_id = job_id
        self.config = config
        self.progress_callback = progress_callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def is_cancelled(self) -> bool:
        return self.cancelled


def _read_config_from_stdin() -> dict[str, Any]:
    data = sys.stdin.read()
    if not data.strip():
        raise ValueError("Missing docking config on stdin")
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise TypeError("Docking config must be a JSON object")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Matcha TUI docking worker process")
    parser.add_argument("--job-id", required=True, help="Job identifier")
    args = parser.parse_args(argv)

    job_id: str = args.job_id
    config = _read_config_from_stdin()

    def emit(event: ProgressEvent) -> None:
        event_dict = event.to_dict()
        event_dict["job_id"] = job_id
        sys.stdout.write(json.dumps(event_dict) + "\n")
        sys.stdout.flush()

    job = WorkerJob(job_id=job_id, config=config, progress_callback=emit)

    def handle_term(_signum: int, _frame: Any) -> None:
        job.cancel()

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)

    try:
        from matcha.tui.docking_worker import run_docking

        run_docking(job)
        return 0
    except Exception as e:
        emit(ProgressEvent(type="error", message=str(e)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

