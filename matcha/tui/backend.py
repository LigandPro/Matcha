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
import subprocess
import time
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
    """Thread-safe manager for docking jobs.

    Runs each job in a separate OS process to allow safe parallelism and GPU pinning.
    """

    def __init__(self):
        self._jobs: Dict[str, "ManagedJob"] = {}
        self._job_queue: deque = deque()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._scheduler_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._all_gpus_cache: Optional[list[int]] = None

        # Keep finished jobs for a short time so the UI can still show them.
        self._retention_seconds = 600

    def start_scheduler(self) -> None:
        """Start the background scheduler thread (idempotent)."""
        with self._lock:
            if self._scheduler_thread and self._scheduler_thread.is_alive():
                return
            self._shutdown = False
            self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self._scheduler_thread.start()

    def shutdown(self) -> None:
        """Stop scheduling and terminate running jobs."""
        with self._lock:
            self._shutdown = True
            self._cv.notify_all()

        # Best-effort termination of all jobs.
        self.cancel_all_jobs()

    def add_job(self, job_id: str, job: "ManagedJob") -> None:
        """Add a job to the queue.

        Args:
            job_id: Unique job identifier
            job: ManagedJob instance
        """
        with self._lock:
            self._jobs[job_id] = job
            self._job_queue.append(job_id)
            emit_debug("info", "backend", f"Job {job_id} added to queue (position: {len(self._job_queue)})")
            self._cv.notify_all()

    def _scheduler_loop(self) -> None:
        """Periodically attempt to start queued jobs based on available GPUs."""
        while True:
            with self._lock:
                if self._shutdown:
                    return
                self._prune_finished_locked()
                # Attempt to start as many jobs as possible.
                self._schedule_once_locked()
                self._cv.wait(timeout=2.0)

    def get_job(self, job_id: str) -> Optional["ManagedJob"]:
        """Get a job by ID (thread-safe).

        Args:
            job_id: Job identifier

        Returns:
            ManagedJob instance or None if not found
        """
        with self._lock:
            return self._jobs.get(job_id)

    def _get_all_gpus(self) -> list[int]:
        """Return list of GPU indices, cached."""
        if self._all_gpus_cache is not None:
            return self._all_gpus_cache
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            indices = []
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    indices.append(int(line))
                except ValueError:
                    continue
            self._all_gpus_cache = sorted(set(indices))
            return self._all_gpus_cache
        except Exception:
            self._all_gpus_cache = []
            return []

    def _get_external_busy_gpus(self) -> set[int]:
        """Return set of GPU indices that have active compute processes (external view)."""
        all_gpus = self._get_all_gpus()
        if not all_gpus:
            return set()
        try:
            gpu_uuid_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            idx_by_uuid: dict[str, int] = {}
            for line in gpu_uuid_out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2:
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                idx_by_uuid[parts[1]] = idx

            compute_out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            busy: set[int] = set()
            for line in compute_out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if not parts or not parts[0]:
                    continue
                uuid = parts[0]
                if uuid in idx_by_uuid:
                    busy.add(idx_by_uuid[uuid])
            return busy
        except Exception:
            # If we cannot query processes, assume unknown. Fall back to internal scheduling only.
            emit_debug("warn", "backend", "Failed to query external GPU usage; falling back to internal-only scheduling")
            return set()

    def _internal_busy_gpus_locked(self) -> set[int]:
        busy: set[int] = set()
        for job in self._jobs.values():
            if job.status == "running" and job.assigned_gpu is not None:
                busy.add(job.assigned_gpu)
        return busy

    def _schedule_once_locked(self) -> None:
        if not self._job_queue:
            return

        all_gpus = self._get_all_gpus()
        internal_busy = self._internal_busy_gpus_locked()

        # If GPU discovery is unavailable, fall back to a single-slot scheduler (CPU or unknown device).
        if not all_gpus:
            any_running = any(j.status == "running" for j in self._jobs.values())
            if any_running:
                return
            # Start the first queued job.
            for job_id in list(self._job_queue):
                job = self._jobs.get(job_id)
                if not job or job.status != "queued":
                    continue
                self._start_job_locked(job)
                try:
                    self._job_queue.remove(job_id)
                except ValueError:
                    pass
                return

        external_busy = self._get_external_busy_gpus()
        auto_free = [g for g in all_gpus if g not in external_busy and g not in internal_busy]

        started_any = True
        while started_any:
            started_any = False
            for job_id in list(self._job_queue):
                job = self._jobs.get(job_id)
                if not job or job.status != "queued":
                    # Remove stale entries.
                    try:
                        self._job_queue.remove(job_id)
                    except ValueError:
                        pass
                    continue

                # Cancelled before start
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.end_time = datetime.now().isoformat()
                    try:
                        self._job_queue.remove(job_id)
                    except ValueError:
                        pass
                    send_notification("progress", {"job_id": job_id, "type": "cancelled", "message": "Job cancelled"})
                    started_any = True
                    break

                requested = job.requested_gpu
                if requested is not None:
                    if requested in internal_busy:
                        continue
                    if all_gpus and requested not in all_gpus:
                        job.status = "failed"
                        job.error = f"Requested GPU {requested} not found"
                        job.end_time = datetime.now().isoformat()
                        try:
                            self._job_queue.remove(job_id)
                        except ValueError:
                            pass
                        send_notification("progress", {"job_id": job_id, "type": "error", "message": job.error})
                        started_any = True
                        break

                    job.assigned_gpu = requested
                    job.external_gpu_busy = requested in external_busy
                    self._start_job_locked(job)
                    internal_busy.add(requested)
                    try:
                        self._job_queue.remove(job_id)
                    except ValueError:
                        pass
                    started_any = True
                    break

                # Auto scheduling
                if auto_free:
                    assigned = auto_free.pop(0)
                    job.assigned_gpu = assigned
                    self._start_job_locked(job)
                    internal_busy.add(assigned)
                    try:
                        self._job_queue.remove(job_id)
                    except ValueError:
                        pass
                    started_any = True
                    break

                # No resources left for auto jobs
                return

    def _start_job_locked(self, job: "ManagedJob") -> None:
        """Spawn the worker process and start reader threads (lock must be held)."""
        job.status = "running"
        job.start_time = datetime.now().isoformat()
        job.progress = {"stage": "init", "percent": 0}

        cmd = [sys.executable, "-m", "matcha.tui.worker_process", "--job-id", job.job_id]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if job.assigned_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(job.assigned_gpu)

        emit_debug("info", "backend", f"Spawning worker for job {job.job_id}", {"cmd": cmd, "gpu": job.assigned_gpu})

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=os.getcwd(),
            bufsize=1,
        )
        job.process = proc

        # Send config then close stdin.
        try:
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(job.config))
            proc.stdin.close()
        except Exception as e:
            emit_debug("error", "backend", f"Failed to send config to worker: {e}")

        def stdout_reader() -> None:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    emit_debug("warn", "backend", "Failed to parse worker event", {"line": line[:500]})
                    continue
                if not isinstance(event, dict):
                    continue
                event.setdefault("job_id", job.job_id)
                send_notification("progress", event)
                self._update_job_from_event(job.job_id, event)

        def stderr_reader() -> None:
            assert proc.stderr is not None
            for raw in proc.stderr:
                msg = raw.rstrip()
                if not msg:
                    continue
                # Keep stderr as debug-only to avoid spamming UI.
                emit_debug("warn", "worker-stderr", msg[:1000], {"job_id": job.job_id})

        job.stdout_thread = threading.Thread(target=stdout_reader, daemon=True)
        job.stderr_thread = threading.Thread(target=stderr_reader, daemon=True)
        job.stdout_thread.start()
        job.stderr_thread.start()

        def waiter() -> None:
            rc = proc.wait()
            with self._lock:
                self._finalize_after_exit_locked(job.job_id, rc)
                self._cv.notify_all()

        job.waiter_thread = threading.Thread(target=waiter, daemon=True)
        job.waiter_thread.start()

    def _update_job_from_event(self, job_id: str, event: dict) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            etype = event.get("type")
            if etype in {"stage_start", "stage_progress", "stage_done"}:
                stage = event.get("stage")
                if stage:
                    if etype == "stage_start":
                        job.progress = {"stage": stage, "percent": 0}
                    elif etype == "stage_progress":
                        job.progress = {"stage": stage, "percent": int(event.get("progress") or 0)}
                    elif etype == "stage_done":
                        job.progress = {"stage": stage, "percent": 100}
            elif etype == "job_done":
                job.status = "completed"
                job.end_time = datetime.now().isoformat()
            elif etype == "cancelled":
                job.status = "cancelled"
                job.end_time = datetime.now().isoformat()
            elif etype == "error":
                job.status = "failed"
                job.error = str(event.get("message") or "Unknown error")
                job.end_time = datetime.now().isoformat()

    def _finalize_after_exit_locked(self, job_id: str, returncode: int) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        # If already terminal, keep it.
        if job.status in {"completed", "failed", "cancelled"}:
            return
        if job.cancel_requested:
            job.status = "cancelled"
        elif returncode == 0:
            job.status = "completed"
        else:
            job.status = "failed"
            job.error = job.error or f"Worker exited with code {returncode}"
        job.end_time = datetime.now().isoformat()

    def _prune_finished_locked(self) -> None:
        now = time.time()
        to_delete: list[str] = []
        for jid, job in self._jobs.items():
            if job.status in {"completed", "failed", "cancelled"} and job.end_time:
                try:
                    end_ts = datetime.fromisoformat(job.end_time).timestamp()
                except Exception:
                    end_ts = now
                if now - end_ts > self._retention_seconds:
                    to_delete.append(jid)
        for jid in to_delete:
            self._jobs.pop(jid, None)

    def cancel_job(self, job_id: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """Cancel a specific job or the running job.

        Args:
            job_id: Job ID to cancel, or None for running job

        Returns:
            Tuple of (success, actual_job_id)
        """
        with self._lock:
            target_id = job_id
            if target_id is None:
                # Backward-compatible: cancel the first running job, else first queued job.
                running = [j.job_id for j in self._jobs.values() if j.status == "running"]
                queued = [j.job_id for j in self._jobs.values() if j.status == "queued"]
                target_id = (running[0] if running else (queued[0] if queued else None))

            if not target_id or target_id not in self._jobs:
                return (False, None)

            job = self._jobs[target_id]
            emit_debug("info", "backend", f"Cancelling job: {target_id}")
            job.cancel_requested = True

            # If queued, it will be finalized by scheduler soon.
            if job.status == "queued":
                try:
                    self._job_queue.remove(target_id)
                except ValueError:
                    pass
                job.status = "cancelled"
                job.end_time = datetime.now().isoformat()
                send_notification("progress", {"job_id": target_id, "type": "cancelled", "message": "Job cancelled"})
                self._cv.notify_all()
                return (True, target_id)

            # If running, terminate the worker process.
            if job.status == "running" and job.process and job.process.poll() is None:
                try:
                    job.process.terminate()
                except Exception:
                    pass
                self._cv.notify_all()
            return (True, target_id)

    def list_all_jobs(self) -> list[dict]:
        """List all active jobs (running and queued).

        Returns:
            List of job info dictionaries
        """
        with self._lock:
            jobs_list = []
            for job in self._jobs.values():
                jobs_list.append({
                    "job_id": job.job_id,
                    "status": job.status,
                    "config": job.config,
                    "cancelled": job.cancel_requested or job.status == "cancelled",
                    "requested_gpu": job.requested_gpu,
                    "assigned_gpu": job.assigned_gpu,
                    "external_gpu_busy": job.external_gpu_busy,
                    "progress": job.progress,
                    "error": job.error,
                    "start_time": job.start_time,
                    "end_time": job.end_time,
                })

            return jobs_list

    def cancel_all_jobs(self) -> None:
        """Cancel all jobs (for shutdown)."""
        with self._lock:
            for job in self._jobs.values():
                job.cancel_requested = True
                if job.process and job.process.poll() is None:
                    try:
                        job.process.terminate()
                    except Exception:
                        pass
            self._cv.notify_all()


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


class ManagedJob:
    """Represents a docking job managed by the backend."""

    def __init__(self, job_id: str, config: dict):
        self.job_id = job_id
        self.config = config
        self.requested_gpu: Optional[int] = config.get("gpu")
        self.assigned_gpu: Optional[int] = None
        self.external_gpu_busy: bool = False
        self.status: str = "queued"  # queued, running, completed, failed, cancelled
        self.cancel_requested: bool = False

        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.error: Optional[str] = None
        self.progress: dict[str, Any] = {"stage": "init", "percent": 0}

        self.process: Optional[subprocess.Popen[str]] = None
        self.stdout_thread: Optional[threading.Thread] = None
        self.stderr_thread: Optional[threading.Thread] = None
        self.waiter_thread: Optional[threading.Thread] = None


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

        job = ManagedJob(job_id, config)
        _job_manager.add_job(job_id, job)
        return {
            "job_id": job_id,
            "status": "queued",
            "requested_gpu": job.requested_gpu,
        }

    def get_progress(self, job_id: Optional[str] = None) -> dict:
        """Get progress of a specific job or the running job."""
        target_id = job_id
        if target_id is None:
            # Backward-compatible: return first running job progress if present.
            jobs = _job_manager.list_all_jobs()
            running = [j for j in jobs if j.get("status") == "running"]
            if running:
                target_id = running[0].get("job_id")

        if not target_id:
            return {"status": "no_job"}

        job = _job_manager.get_job(target_id)
        if not job:
            return {"status": "no_job"}

        return {
            "job_id": job.job_id,
            "status": job.status,
            "cancelled": job.cancel_requested or job.status == "cancelled",
            "running": job.status == "running",
            "requested_gpu": job.requested_gpu,
            "assigned_gpu": job.assigned_gpu,
            "progress": job.progress,
            "error": job.error,
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
        _job_manager.shutdown()
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
    _job_manager.shutdown()
    _shutdown_event.set()


def run_backend() -> None:
    """Run the JSON-RPC backend server."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Send ready notification
    send_notification("ready", {"version": "1.0.0"})

    # Start background scheduler for queued jobs
    _job_manager.start_scheduler()

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
