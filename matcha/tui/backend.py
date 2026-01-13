"""
JSON-RPC server for Matcha TUI.

Communicates with the Node.js/Ink frontend via stdin/stdout.
"""

import json
import os
import sys
import signal
import threading
from pathlib import Path
from typing import Any, Callable, Optional
from datetime import datetime

from matcha.tui.protocol import (
    FileInfo,
    ValidationResult,
    RunInfo,
    PoseInfo,
    ProgressEvent,
    PipelineStage,
    make_response,
    make_error,
    make_notification,
    ErrorCode,
)

# Global state
_current_job: Optional["DockingJob"] = None
_shutdown_event = threading.Event()


class DockingJob:
    """Represents a running docking job."""

    def __init__(self, job_id: str, config: dict, progress_callback: Callable):
        self.job_id = job_id
        self.config = config
        self.progress_callback = progress_callback
        self.cancelled = False
        self.thread: Optional[threading.Thread] = None

    def cancel(self) -> None:
        self.cancelled = True

    def is_cancelled(self) -> bool:
        return self.cancelled


class RPCHandler:
    """Handles JSON-RPC requests."""

    def __init__(self):
        self.methods: dict[str, Callable] = {
            "ping": self.ping,
            "list_files": self.list_files,
            "validate_receptor": self.validate_receptor,
            "validate_ligand": self.validate_ligand,
            "get_ligand_info": self.get_ligand_info,
            "start_docking": self.start_docking,
            "get_progress": self.get_progress,
            "cancel_job": self.cancel_job,
            "list_runs": self.list_runs,
            "get_run_details": self.get_run_details,
            "get_poses": self.get_poses,
            "check_gpu": self.check_gpu,
            "check_checkpoints": self.check_checkpoints,
            "shutdown": self.shutdown,
        }

    def handle(self, request: dict) -> Optional[dict]:
        """Handle a JSON-RPC request and return response."""
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method not in self.methods:
            return make_error(req_id, ErrorCode.METHOD_NOT_FOUND, f"Method not found: {method}")

        try:
            result = self.methods[method](**params) if params else self.methods[method]()
            return make_response(req_id, result)
        except TypeError as e:
            return make_error(req_id, ErrorCode.INVALID_PARAMS, str(e))
        except Exception as e:
            return make_error(req_id, ErrorCode.INTERNAL_ERROR, str(e))

    def ping(self) -> dict:
        """Health check."""
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    def list_files(
        self,
        path: str,
        extensions: Optional[list[str]] = None,
        show_hidden: bool = False,
    ) -> list[dict]:
        """List files in a directory."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not p.is_dir():
            raise ValueError(f"Not a directory: {path}")

        files: list[FileInfo] = []
        try:
            for item in sorted(p.iterdir()):
                if not show_hidden and item.name.startswith("."):
                    continue

                ext = item.suffix.lower()
                if item.is_file() and extensions:
                    if ext not in extensions and ext.lstrip(".") not in extensions:
                        continue

                files.append(
                    FileInfo(
                        name=item.name,
                        path=str(item),
                        is_dir=item.is_dir(),
                        size=item.stat().st_size if item.is_file() else 0,
                        extension=ext,
                    )
                )
        except PermissionError:
            pass

        # Sort: directories first, then files
        files.sort(key=lambda f: (not f.is_dir, f.name.lower()))
        return [f.to_dict() for f in files]

    def validate_receptor(self, path: str) -> dict:
        """Validate a receptor PDB file."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return ValidationResult(False, f"File not found: {path}").to_dict()
        if p.suffix.lower() != ".pdb":
            return ValidationResult(False, "Receptor must be a .pdb file").to_dict()

        try:
            with open(p) as f:
                content = f.read()

            atom_count = content.count("\nATOM ")
            hetatm_count = content.count("\nHETATM ")
            chains = set()
            for line in content.split("\n"):
                if line.startswith("ATOM ") or line.startswith("HETATM"):
                    if len(line) > 21:
                        chains.add(line[21])

            return ValidationResult(
                True,
                f"Valid PDB: {atom_count} atoms, {hetatm_count} hetatoms, {len(chains)} chains",
                {
                    "atom_count": atom_count,
                    "hetatm_count": hetatm_count,
                    "chains": list(chains),
                },
            ).to_dict()
        except Exception as e:
            return ValidationResult(False, f"Error reading PDB: {e}").to_dict()

    def validate_ligand(self, path: str) -> dict:
        """Validate a ligand file."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return ValidationResult(False, f"File not found: {path}").to_dict()

        ext = p.suffix.lower()
        if ext not in {".sdf", ".mol", ".mol2", ".pdb"}:
            return ValidationResult(
                False, f"Unsupported format: {ext}. Use .sdf, .mol, .mol2, or .pdb"
            ).to_dict()

        try:
            from rdkit import Chem

            if ext == ".sdf":
                suppl = Chem.SDMolSupplier(str(p), removeHs=False, sanitize=False)
                mols = [m for m in suppl if m is not None]
                if not mols:
                    return ValidationResult(False, "No valid molecules in SDF").to_dict()
                mol = mols[0]
                n_mols = len(mols)
            elif ext in {".mol", ".mol2"}:
                mol = Chem.MolFromMolFile(str(p), removeHs=False, sanitize=False)
                n_mols = 1
            else:  # .pdb
                mol = Chem.MolFromPDBFile(str(p), removeHs=False, sanitize=False)
                n_mols = 1

            if mol is None:
                return ValidationResult(False, "Failed to parse molecule").to_dict()

            return ValidationResult(
                True,
                f"Valid ligand: {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds",
                {
                    "atom_count": mol.GetNumAtoms(),
                    "bond_count": mol.GetNumBonds(),
                    "n_molecules": n_mols,
                    "format": ext,
                },
            ).to_dict()
        except ImportError:
            return ValidationResult(
                True, "Ligand file exists (RDKit not available for validation)"
            ).to_dict()
        except Exception as e:
            return ValidationResult(False, f"Error: {e}").to_dict()

    def get_ligand_info(self, path: str) -> dict:
        """Get detailed ligand information for batch mode."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"error": f"File not found: {path}"}

        try:
            from rdkit import Chem

            suppl = Chem.SDMolSupplier(str(p), removeHs=False, sanitize=False)
            molecules = []
            for i, mol in enumerate(suppl):
                if mol is not None:
                    name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
                    molecules.append(
                        {
                            "index": i,
                            "name": name,
                            "atoms": mol.GetNumAtoms(),
                            "bonds": mol.GetNumBonds(),
                        }
                    )
            return {"count": len(molecules), "molecules": molecules}
        except Exception as e:
            return {"error": str(e)}

    def start_docking(self, config: dict) -> dict:
        """Start a docking job."""
        global _current_job

        if _current_job is not None and _current_job.thread and _current_job.thread.is_alive():
            return {"error": "A job is already running", "job_id": _current_job.job_id}

        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        def progress_callback(event: ProgressEvent):
            send_notification("progress", event.to_dict())

        _current_job = DockingJob(job_id, config, progress_callback)

        # Start docking in background thread
        from matcha.tui.docking_worker import run_docking

        _current_job.thread = threading.Thread(
            target=run_docking, args=(_current_job,), daemon=True
        )
        _current_job.thread.start()

        return {"job_id": job_id, "status": "started"}

    def get_progress(self) -> dict:
        """Get current job progress."""
        global _current_job
        if _current_job is None:
            return {"status": "no_job"}

        return {
            "job_id": _current_job.job_id,
            "cancelled": _current_job.cancelled,
            "running": _current_job.thread.is_alive() if _current_job.thread else False,
        }

    def cancel_job(self) -> dict:
        """Cancel the current job."""
        global _current_job
        if _current_job is None:
            return {"status": "no_job"}

        _current_job.cancel()
        return {"status": "cancelled", "job_id": _current_job.job_id}

    def list_runs(self, output_dir: str) -> list[dict]:
        """List previous docking runs in a directory."""
        p = Path(output_dir).expanduser().resolve()
        if not p.exists():
            return []

        runs: list[RunInfo] = []
        for item in p.iterdir():
            if item.is_dir():
                log_files = list(item.glob("*.log")) + list(item.glob("**/*.log"))
                if log_files or list(item.glob("*_best.sdf")):
                    stat = item.stat()
                    runs.append(
                        RunInfo(
                            name=item.name,
                            path=str(item),
                            date=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            status="completed" if list(item.glob("*_best.sdf")) else "unknown",
                        )
                    )

        runs.sort(key=lambda r: r.date, reverse=True)
        return [r.to_dict() for r in runs]

    def get_run_details(self, run_path: str) -> dict:
        """Get details of a completed run."""
        p = Path(run_path).expanduser().resolve()
        if not p.exists():
            return {"error": f"Run not found: {run_path}"}

        details = {
            "name": p.name,
            "path": str(p),
            "files": {},
        }

        # Find output files
        for pattern, key in [
            ("*_best.sdf", "best_pose"),
            ("*_poses.sdf", "all_poses"),
            ("*.log", "log"),
        ]:
            matches = list(p.glob(pattern))
            if matches:
                details["files"][key] = str(matches[0])

        # Try to parse log for metrics
        log_file = details["files"].get("log")
        if log_file:
            try:
                with open(log_file) as f:
                    content = f.read()
                    # Extract basic info from log
                    if "n-samples" in content.lower():
                        details["has_log"] = True
            except Exception:
                pass

        return details

    def get_poses(self, run_path: str) -> list[dict]:
        """Get pose results from a completed run."""
        import numpy as np

        p = Path(run_path).expanduser().resolve()

        # Look for metrics file
        metrics_files = list(p.glob("**/*_final_preds_fast_metrics.npy"))
        if not metrics_files:
            return []

        try:
            data = np.load(metrics_files[0], allow_pickle=True).item()
            poses: list[PoseInfo] = []

            for uid, samples in data.items():
                for i, sample in enumerate(samples):
                    pb_filters = sample.get("posebusters_filters_fast", [False] * 5)
                    poses.append(
                        PoseInfo(
                            rank=i + 1,
                            error_estimate=float(sample.get("error_estimate_0", 0)),
                            pb_count=int(sample.get("posebusters_filters_passed_count_fast", 0)),
                            not_too_far_away=bool(pb_filters[0]) if len(pb_filters) > 0 else False,
                            no_internal_clash=bool(pb_filters[1]) if len(pb_filters) > 1 else False,
                            no_clashes=bool(pb_filters[2]) if len(pb_filters) > 2 else False,
                            no_volume_clash=bool(pb_filters[3]) if len(pb_filters) > 3 else False,
                            buried_fraction=float(pb_filters[4]) if len(pb_filters) > 4 else 0.0,
                        )
                    )
                break  # Only first UID for now

            # Sort by ranking
            poses.sort(key=lambda p: (-p.pb_count, p.error_estimate))
            for i, pose in enumerate(poses):
                pose.rank = i + 1

            return [p.to_dict() for p in poses]
        except Exception as e:
            return [{"error": str(e)}]

    def check_gpu(self) -> dict:
        """Check GPU availability."""
        try:
            import torch

            if torch.cuda.is_available():
                return {
                    "available": True,
                    "count": torch.cuda.device_count(),
                    "devices": [
                        {
                            "index": i,
                            "name": torch.cuda.get_device_name(i),
                            "memory": torch.cuda.get_device_properties(i).total_memory,
                        }
                        for i in range(torch.cuda.device_count())
                    ],
                }
            return {"available": False, "message": "No CUDA devices found"}
        except ImportError:
            return {"available": False, "message": "PyTorch not installed"}
        except Exception as e:
            return {"available": False, "message": str(e)}

    def check_checkpoints(self, path: Optional[str] = None) -> dict:
        """Check if checkpoints are available."""
        from pathlib import Path

        if path:
            p = Path(path).expanduser().resolve()
            if p.exists() and (p / "pipeline").exists():
                stages = list((p / "pipeline").iterdir())
                return {"available": True, "path": str(p), "stages": len(stages)}

        # Check default locations
        default_paths = [
            Path.home() / ".cache" / "matcha" / "checkpoints",
            Path("./checkpoints"),
        ]
        for dp in default_paths:
            if dp.exists() and (dp / "pipeline").exists():
                stages = list((dp / "pipeline").iterdir())
                return {"available": True, "path": str(dp), "stages": len(stages)}

        return {"available": False, "message": "Checkpoints not found"}

    def shutdown(self) -> dict:
        """Shutdown the backend."""
        _shutdown_event.set()
        return {"status": "shutting_down"}


# Global handler instance
_handler = RPCHandler()


def send_notification(method: str, params: dict) -> None:
    """Send a notification to the frontend."""
    msg = make_notification(method, params)
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def handle_signal(signum: int, frame: Any) -> None:
    """Handle shutdown signals."""
    global _current_job
    if _current_job:
        _current_job.cancel()
    _shutdown_event.set()


def run_backend() -> None:
    """Run the JSON-RPC backend server."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Send ready notification
    send_notification("ready", {"version": "1.0.0"})

    # Main loop - read requests from stdin
    while not _shutdown_event.is_set():
        try:
            line = sys.stdin.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                response = make_error(None, ErrorCode.PARSE_ERROR, str(e))
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                continue

            response = _handler.handle(request)
            if response:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

        except KeyboardInterrupt:
            break
        except Exception as e:
            # Log errors but keep running
            send_notification("error", {"message": str(e)})

    # Cleanup
    global _current_job
    if _current_job:
        _current_job.cancel()


if __name__ == "__main__":
    run_backend()
