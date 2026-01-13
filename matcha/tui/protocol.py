"""
JSON-RPC protocol types and utilities for TUI communication.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Optional
import json


class PipelineStage(str, Enum):
    INIT = "init"
    CHECKPOINTS = "checkpoints"
    DATASET = "dataset"
    ESM = "esm"
    STAGE1 = "stage1"
    STAGE2 = "stage2"
    STAGE3 = "stage3"
    SCORING = "scoring"
    POSEBUSTERS = "posebusters"
    DONE = "done"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FileInfo:
    name: str
    path: str
    is_dir: bool
    size: int
    extension: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationResult:
    valid: bool
    message: str
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProgressEvent:
    type: str  # stage_start, stage_progress, stage_done, poses_update, job_done, error
    stage: Optional[str] = None
    name: Optional[str] = None
    progress: Optional[int] = None
    elapsed: Optional[float] = None
    message: Optional[str] = None
    poses: Optional[list] = None
    best_error: Optional[float] = None
    best_pb: Optional[int] = None
    output_path: Optional[str] = None
    current_ligand: Optional[str] = None
    ligand_index: Optional[int] = None
    total_ligands: Optional[int] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class PoseInfo:
    rank: int
    error_estimate: float
    pb_count: int
    not_too_far_away: bool
    no_internal_clash: bool
    no_clashes: bool
    no_volume_clash: bool
    buried_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunInfo:
    name: str
    path: str
    date: str
    status: str
    receptor: Optional[str] = None
    ligand: Optional[str] = None
    n_poses: Optional[int] = None
    best_error: Optional[float] = None
    best_pb: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def make_response(id: int, result: Any) -> dict:
    """Create a JSON-RPC response."""
    return {
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    }


def make_error(id: int, code: int, message: str, data: Any = None) -> dict:
    """Create a JSON-RPC error response."""
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": id,
        "error": error,
    }


def make_notification(method: str, params: dict) -> dict:
    """Create a JSON-RPC notification (no response expected)."""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
    }


# Error codes
class ErrorCode:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # Custom errors
    FILE_NOT_FOUND = -32000
    VALIDATION_ERROR = -32001
    JOB_ERROR = -32002
    CANCELLED = -32003
