"""
JSON-RPC server for Matcha TUI.

Communicates with the Node.js/Ink frontend via stdin/stdout.
"""

import json
import locale
import os
import sys
import signal
import threading
import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Dict, TypedDict, List, Union
from datetime import datetime
from collections import deque

from matcha.tui.protocol import (
    FileInfo,
    ValidationResult,
    RunInfo,
    PoseInfo,
    ProgressEvent,
    DebugEvent,
    PipelineStage,
    make_response,
    make_error,
    make_notification,
    ErrorCode,
)
from matcha.tui.utils import extract_pb_filters

# Force English locale for consistent output
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANGUAGE'] = 'en_US:en'

try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except Exception:
    # Fallback if locale not installed on system
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except Exception:
        pass  # Use system default as last resort


# TypedDict definitions for return types
class GpuDeviceInfo(TypedDict):
    """Information about a single GPU device."""
    index: int
    name: str
    memory: int


class GpuInfo(TypedDict, total=False):
    """GPU availability information returned by check_gpu()."""
    available: bool
    count: int
    devices: List[GpuDeviceInfo]
    message: str


class RunDetails(TypedDict, total=False):
    """Details of a completed docking run returned by get_run_details()."""
    name: str
    path: str
    files: Dict[str, str]
    is_batch: bool
    has_log: bool
    receptor: str
    ligand: str
    error: str


class DeleteResult(TypedDict, total=False):
    """Result of deleting a run directory returned by delete_run()."""
    success: bool
    message: str
    error: str


class JobManager:
    """Thread-safe manager for docking jobs."""

    def __init__(self):
        self._jobs: Dict[str, "DockingJob"] = {}
        self._job_queue: deque = deque()
        self._running_job_id: Optional[str] = None
        self._lock = threading.Lock()

    def add_job(self, job_id: str, job: "DockingJob") -> None:
        """Add a job to the queue.

        Args:
            job_id: Unique job identifier
            job: DockingJob instance
        """
        with self._lock:
            self._jobs[job_id] = job
            self._job_queue.append(job_id)
            emit_debug("info", "backend", f"Job {job_id} added to queue (position: {len(self._job_queue)})")

    def start_next(self) -> None:
        """Start the next job from the queue (must be called WITHOUT holding lock)."""
        with self._lock:
            if len(self._job_queue) == 0:
                return

            job_id = self._job_queue.popleft()
            job = self._jobs.get(job_id)

            if not job:
                return

            self._running_job_id = job_id
            emit_debug("info", "backend", f"Starting job from queue: {job_id}")

        # Start thread OUTSIDE lock to avoid potential deadlocks
        job.thread = threading.Thread(
            target=self._run_job, args=(job_id,), daemon=False
        )
        job.thread.start()

    def _run_job(self, job_id: str) -> None:
        """Run a docking job in a separate thread."""
        try:
            job = self.get_job(job_id)
            if not job:
                emit_debug("error", "backend", f"Job not found: {job_id}")
                return

            emit_debug("info", "backend", f"Executing job: {job_id}")

            # Import here to avoid circular dependency
            from matcha.tui.docking_worker import run_docking

            run_docking(job)

        except Exception as e:
            emit_debug("error", "backend", f"Job {job_id} failed: {e}")
            # Send error notification
            send_notification("job_error", {"job_id": job_id, "error": str(e)})
        finally:
            # Clean up and start next job
            self.cleanup_job(job_id)

    def cleanup_job(self, job_id: str) -> None:
        """Clean up a completed job and start next in queue.

        Args:
            job_id: Job ID to clean up
        """
        # Perform cleanup inside lock
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]

            if self._running_job_id == job_id:
                self._running_job_id = None

        # Start next job OUTSIDE lock to prevent deadlock
        self.start_next()

    def get_job(self, job_id: str) -> Optional["DockingJob"]:
        """Get a job by ID (thread-safe).

        Args:
            job_id: Job identifier

        Returns:
            DockingJob instance or None if not found
        """
        with self._lock:
            return self._jobs.get(job_id)

    def get_running_job_id(self) -> Optional[str]:
        """Get the currently running job ID (thread-safe).

        Returns:
            Job ID or None
        """
        with self._lock:
            return self._running_job_id

    def cancel_job(self, job_id: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """Cancel a specific job or the running job.

        Args:
            job_id: Job ID to cancel, or None for running job

        Returns:
            Tuple of (success, actual_job_id)
        """
        with self._lock:
            target_id = job_id or self._running_job_id

            if not target_id or target_id not in self._jobs:
                return (False, None)

            job = self._jobs[target_id]
            emit_debug("info", "backend", f"Cancelling job: {target_id}")
            job.cancel()
            return (True, target_id)

    def list_all_jobs(self) -> list[dict]:
        """List all active jobs (running and queued).

        Returns:
            List of job info dictionaries
        """
        with self._lock:
            jobs_list = []

            # Add running job
            if self._running_job_id and self._running_job_id in self._jobs:
                job = self._jobs[self._running_job_id]
                jobs_list.append({
                    "job_id": self._running_job_id,
                    "status": "running",
                    "config": job.config,
                    "cancelled": job.cancelled,
                })

            # Add queued jobs
            for job_id in self._job_queue:
                if job_id in self._jobs:
                    job = self._jobs[job_id]
                    jobs_list.append({
                        "job_id": job_id,
                        "status": "queued",
                        "config": job.config,
                        "cancelled": job.cancelled,
                    })

            return jobs_list

    def cancel_all_jobs(self) -> None:
        """Cancel all jobs (for shutdown)."""
        with self._lock:
            for job in self._jobs.values():
                job.cancel()


# Global job manager instance
_job_manager = JobManager()
_shutdown_event = threading.Event()


def emit_debug(level: str, component: str, message: str, data: dict = None) -> None:
    """Emit a debug event to the frontend.

    Args:
        level: 'debug', 'info', 'warn', or 'error'
        component: Component name (e.g., 'backend', 'worker', 'rpc')
        message: Debug message
        data: Optional additional data
    """
    if not os.environ.get('MATCHA_DEBUG'):
        return

    event = DebugEvent(
        level=level,
        component=component,
        message=message,
        timestamp=datetime.now().isoformat(),
        data=data
    )

    notification = make_notification("debug", event.to_dict())
    print(json.dumps(notification), flush=True)


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
            "list_jobs": self.list_jobs,
            "list_runs": self.list_runs,
            "get_run_details": self.get_run_details,
            "get_poses": self.get_poses,
            "delete_run": self.delete_run,
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
        emit_debug("debug", "backend", f"Validating receptor: {path}")
        p = Path(path).expanduser().resolve()
        if not p.exists():
            emit_debug("warn", "backend", f"Receptor file not found: {path}")
            return ValidationResult(False, f"File not found: {path}").to_dict()
        if p.suffix.lower() != ".pdb":
            emit_debug("warn", "backend", f"Invalid receptor extension: {p.suffix}")
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

            emit_debug("info", "backend", f"Receptor validated: {atom_count} atoms, {len(chains)} chains")
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
            emit_debug("error", "backend", f"Error reading receptor PDB: {e}")
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
        """Start a docking job (adds to queue)."""
        emit_debug("info", "backend", "Starting docking job", {"config": config})

        # Generate unique job ID with milliseconds
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        def progress_callback(event: ProgressEvent):
            # Add job_id to progress event
            event_dict = event.to_dict()
            event_dict["job_id"] = job_id
            send_notification("progress", event_dict)

        # Create and add job to manager
        job = DockingJob(job_id, config, progress_callback)
        _job_manager.add_job(job_id, job)

        # Start immediately if no job is running
        if _job_manager.get_running_job_id() is None:
            _job_manager.start_next()

        return {"job_id": job_id, "status": "queued"}

    def get_progress(self, job_id: Optional[str] = None) -> dict:
        """Get progress of a specific job or the running job."""
        # If no job_id specified, use running job
        target_id = job_id or _job_manager.get_running_job_id()

        if not target_id:
            return {"status": "no_job"}

        job = _job_manager.get_job(target_id)
        if not job:
            return {"status": "no_job"}

        return {
            "job_id": job.job_id,
            "cancelled": job.cancelled,
            "running": job.thread.is_alive() if job.thread else False,
        }

    def cancel_job(self, job_id: Optional[str] = None) -> dict:
        """Cancel a specific job or the running job."""
        success, target_id = _job_manager.cancel_job(job_id)

        if not success:
            emit_debug("warn", "backend", "No job to cancel")
            return {"status": "no_job"}

        return {"status": "cancelled", "job_id": target_id}

    def list_jobs(self) -> dict:
        """List all active jobs (running and queued)."""
        return {"jobs": _job_manager.list_all_jobs()}

    def list_runs(self, output_dir: str) -> List[Dict[str, Any]]:
        """List previous docking runs in a directory."""
        p = Path(output_dir).expanduser().resolve()
        if not p.exists():
            return []

        runs: list[RunInfo] = []
        run_with_times: list[tuple[RunInfo, float]] = []

        for item in p.iterdir():
            if item.is_dir():
                log_files = list(item.glob("*.log")) + list(item.glob("**/*.log"))
                has_single_best = bool(list(item.glob("*_best.sdf")))
                batch_best_dir = item / "best_poses"
                has_batch_best = batch_best_dir.is_dir() and any(batch_best_dir.glob("*.sdf"))
                if log_files or has_single_best or has_batch_best:
                    stat = item.stat()
                    mtime = stat.st_mtime

                    # Parse receptor and ligand from log if available
                    receptor = None
                    ligand = None
                    if log_files:
                        try:
                            with open(log_files[0]) as f:
                                for line in f:
                                    if 'Receptor' in line and ':' in line:
                                        receptor = line.split(':', 1)[1].strip()
                                    elif 'Ligand' in line and ':' in line and 'ligand' in line.lower():
                                        ligand = line.split(':', 1)[1].strip()
                        except Exception as e:
                            emit_debug("warn", "backend", f"Failed to parse log file for run info: {e}")

                    run_info = RunInfo(
                        name=item.name,
                        path=str(item),
                        date=datetime.fromtimestamp(mtime).isoformat(),
                        status="completed" if (has_single_best or has_batch_best) else "unknown",
                        receptor=receptor,
                        ligand=ligand,
                    )
                    run_with_times.append((run_info, mtime))

        # Sort by modification time descending (newest first)
        run_with_times.sort(key=lambda x: x[1], reverse=True)
        runs = [r for r, _ in run_with_times]

        return [r.to_dict() for r in runs]

    def get_run_details(self, run_path: str) -> RunDetails:
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
        ]:
            matches = list(p.glob(pattern))
            if matches:
                details["files"][key] = str(matches[0])

        # Prefer summary log in run root if present
        root_log = p / f"{p.name}.log"
        if root_log.exists():
            details["files"]["log"] = str(root_log)
        else:
            matches = list(p.glob("*.log"))
            if matches:
                details["files"]["log"] = str(matches[0])

        batch_best_dir = p / "best_poses"
        batch_all_dir = p / "all_poses"
        batch_logs_dir = p / "logs"
        if batch_best_dir.is_dir():
            details["files"]["best_poses"] = str(batch_best_dir)
            details["is_batch"] = True
        if batch_all_dir.is_dir():
            details["files"]["all_poses_dir"] = str(batch_all_dir)
            details["is_batch"] = True
        if batch_logs_dir.is_dir():
            details["files"]["logs_dir"] = str(batch_logs_dir)
            details["is_batch"] = True

        # Try to parse log for metrics and input files
        log_file = details["files"].get("log")
        if log_file:
            try:
                with open(log_file) as f:
                    content = f.read()
                    # Extract basic info from log
                    if "n-samples" in content.lower():
                        details["has_log"] = True

                    # Extract receptor and ligand paths from log
                    for line in content.split('\n'):
                        if 'Receptor' in line and ':' in line:
                            receptor_path = line.split(':', 1)[1].strip()
                            details["receptor"] = receptor_path
                        elif 'Ligand' in line and ':' in line and 'ligand' in line.lower():
                            ligand_path = line.split(':', 1)[1].strip()
                            details["ligand"] = ligand_path
            except Exception as e:
                emit_debug("warn", "backend", f"Failed to parse log file for run details: {e}")

        return details

    def get_poses(self, run_path: str, ligand_name: str | None = None) -> list[dict]:
        """Get pose results from a completed run.

        Args:
            run_path: Path to the run directory
            ligand_name: Optional ligand name for batch mode
        """
        import numpy as np

        p = Path(run_path).expanduser().resolve()

        # Determine search pattern based on ligand_name
        if ligand_name:
            search_pattern = f"**/*{ligand_name}*_final_preds.npy"
        else:
            search_pattern = "**/*_final_preds.npy"

        # Look for metrics file
        metrics_files = list(p.glob(search_pattern))
        if not metrics_files:
            return []

        try:
            data = np.load(metrics_files[0], allow_pickle=True).item()
            poses: list[PoseInfo] = []

            for uid, samples_dict in data.items():
                if 'sample_metrics' not in samples_dict:
                    continue

                for i, sample in enumerate(samples_dict['sample_metrics']):
                    pb_data = extract_pb_filters(sample)
                    gnina_score = float(sample["gnina_score"]) if "gnina_score" in sample else None
                    poses.append(
                        PoseInfo(
                            rank=i + 1,
                            pb_count=int(sample.get("posebusters_filters_passed_count_fast", 0)),
                            gnina_score=gnina_score,
                            **pb_data,
                        )
                    )
                break  # Only first UID for now

            # Sort by ranking: highest PB count first, then lowest GNINA score
            poses.sort(key=lambda p: (-p.pb_count, p.gnina_score if p.gnina_score is not None else float("inf")))
            for i, pose in enumerate(poses):
                pose.rank = i + 1

            return [p.to_dict() for p in poses]
        except Exception as e:
            return [{"error": str(e)}]

    def check_gpu(self) -> GpuInfo:
        """Check GPU/accelerator availability (CUDA, MPS)."""
        try:
            import torch

            if torch.cuda.is_available():
                return {
                    "available": True,
                    "type": "cuda",
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
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return {
                    "available": True,
                    "type": "mps",
                    "count": 1,
                    "devices": [{"index": 0, "name": "Apple Metal (MPS)", "memory": 0}],
                    "message": "Apple Metal GPU available (GNINA scoring requires CUDA)",
                }
            return {"available": False, "message": "No CUDA or MPS devices found"}
        except ImportError:
            return {"available": False, "message": "PyTorch not installed"}
        except Exception as e:
            return {"available": False, "message": str(e)}

    def delete_run(self, run_path: str) -> DeleteResult:
        """Delete a docking run directory and all its contents.

        Args:
            run_path: Absolute path to the run directory

        Returns:
            dict with success status and message
        """
        try:
            run_path_obj = Path(run_path)

            # Verify path exists and is a directory
            if not run_path_obj.exists():
                return {"success": False, "error": f"Path does not exist: {run_path}"}

            if not run_path_obj.is_dir():
                return {"success": False, "error": f"Path is not a directory: {run_path}"}

            # Delete the directory and all contents
            shutil.rmtree(run_path)

            return {"success": True, "message": f"Successfully deleted run: {run_path_obj.name}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to delete run: {str(e)}"}

    def check_checkpoints(self, path: Optional[str] = None) -> dict:
        """Check if checkpoints are available."""
        from pathlib import Path

        if path:
            p = Path(path).expanduser().resolve()
            if p.exists() and (p / "matcha_pipeline").exists():
                stages = list((p / "matcha_pipeline").iterdir())
                return {"available": True, "path": str(p), "stages": len(stages)}

        # Check default locations
        default_paths = [
            Path.home() / ".cache" / "matcha" / "checkpoints",
            Path("./checkpoints"),
        ]
        for dp in default_paths:
            if dp.exists() and (dp / "matcha_pipeline").exists():
                stages = list((dp / "matcha_pipeline").iterdir())
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
    # Cancel all running and queued jobs
    _job_manager.cancel_all_jobs()
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

    # Cleanup - cancel all jobs
    _job_manager.cancel_all_jobs()


if __name__ == "__main__":
    run_backend()
