from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import prody
from rdkit import Chem
from rdkit.Chem import AllChem

from matcha.webui import build_matcha_workspace


@dataclass
class JobState:
    id: str
    state: str
    message: str
    created_at: float
    updated_at: float
    root_dir: Path
    error: str | None = None
    result_path: Path | None = None
    log_path: Path | None = None
    process: subprocess.Popen[str] | None = None
    cancel_requested: bool = False


JOBS: dict[str, JobState] = {}
JOBS_LOCK = threading.Lock()
SAFE_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
DEFAULT_FIXTURE_ENTRY = os.environ.get("MATCHA_DEFAULT_FIXTURE_ENTRY", "3HTB").upper()
DEFAULT_FIXTURE_LIGAND = os.environ.get("MATCHA_DEFAULT_FIXTURE_LIGAND", "JZ4").upper()


def _jobs_root() -> Path:
    configured = os.environ.get("MATCHA_UI_RUNS_DIR")
    if configured:
        root = Path(configured).expanduser()
    else:
        root = Path.cwd() / ".matcha-ui-runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_fixture_paths() -> tuple[Path, Path]:
    receptor_override = os.environ.get("MATCHA_DEFAULT_RECEPTOR")
    ligand_override = os.environ.get("MATCHA_DEFAULT_LIGAND")
    if receptor_override and ligand_override:
        receptor_path = Path(receptor_override).expanduser()
        ligand_path = Path(ligand_override).expanduser()
        if receptor_path.exists() and ligand_path.exists():
            return receptor_path, ligand_path
        raise FileNotFoundError("Configured default Matcha fixture paths are not available on the server")

    cache_dir = _jobs_root() / "fixtures" / DEFAULT_FIXTURE_ENTRY
    cache_dir.mkdir(parents=True, exist_ok=True)
    receptor_path = cache_dir / f"{DEFAULT_FIXTURE_ENTRY}_protein.pdb"
    ligand_path = cache_dir / f"{DEFAULT_FIXTURE_ENTRY}_{DEFAULT_FIXTURE_LIGAND}_ideal.sdf"
    if receptor_path.exists() and ligand_path.exists():
        return receptor_path, ligand_path

    pdb_text = _download_text(f"https://files.rcsb.org/download/{DEFAULT_FIXTURE_ENTRY}.pdb")
    ligand_text = _download_text(f"https://files.rcsb.org/ligands/download/{DEFAULT_FIXTURE_LIGAND}_ideal.sdf")
    receptor_path.write_text(_extract_protein_pdb(pdb_text))
    ligand_path.write_text(ligand_text)
    return receptor_path, ligand_path


def _default_fixture_payload() -> dict[str, Any]:
    receptor_path, ligand_path = _default_fixture_paths()
    return {
        "receptorFilename": receptor_path.name,
        "receptorText": receptor_path.read_text(),
        "receptorSourcePath": str(receptor_path),
        "ligandFilename": ligand_path.name,
        "ligandText": ligand_path.read_text(),
        "ligandSourcePath": str(ligand_path),
    }


def _smiles_filename(name: str | None) -> str:
    raw = (name or "smiles_ligand").strip()
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return f"{sanitized or 'smiles_ligand'}.sdf"


def _smiles_preview_payload(smiles: str, name: str | None = None) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Failed to parse SMILES")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status != 0:
        params.useRandomCoords = True
        embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status != 0:
        raise ValueError("Failed to generate a 3D conformer from SMILES")
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=400)
    except Exception:
        pass
    mol.SetProp("_Name", (name or "smiles_ligand").strip() or "smiles_ligand")
    return {
        "filename": _smiles_filename(name),
        "text": f"{Chem.MolToMolBlock(mol)}\n$$$$\n",
        "sourcePath": f"SMILES: {smiles}",
        "smiles": smiles,
    }


def _download_text(url: str) -> str:
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def _extract_protein_pdb(pdb_text: str) -> str:
    kept_lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("HEADER", "TITLE ", "COMPND", "SOURCE", "KEYWDS", "EXPDTA", "AUTHOR", "REMARK", "CRYST1", "SEQRES", "ATOM  ", "TER   ", "END")):
            kept_lines.append(line)
    if not any(line.startswith("ATOM  ") for line in kept_lines):
        raise FileNotFoundError("Downloaded PDB does not contain protein ATOM records")
    if not kept_lines or kept_lines[-1] != "END":
        kept_lines.append("END")
    return "\n".join(kept_lines) + "\n"


def _read_log_tail(path: Path | None, *, max_bytes: int = 32_768) -> tuple[str, bool]:
    if path is None or not path.exists():
        return "", False
    data = path.read_bytes()
    truncated = len(data) > max_bytes
    if truncated:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="replace"), truncated


def _job_payload(job: JobState) -> dict[str, Any]:
    return {
        "jobId": job.id,
        "state": job.state,
        "message": job.message,
        "createdAt": job.created_at,
        "updatedAt": job.updated_at,
        "error": job.error,
        "resultReady": job.result_path is not None and job.result_path.exists(),
        "logReady": job.log_path is not None and job.log_path.exists(),
        "cancelReady": job.state in {"queued", "running", "cancelling"} and job.process is not None,
    }


def _update_job(job_id: str, **changes: Any) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def _sanitize_leaf_name(name: str, *, field: str) -> str:
    candidate = Path(name)
    sanitized = candidate.name
    if sanitized != name or sanitized in {"", ".", ".."}:
        raise ValueError(f"{field} must be a simple file name")
    return sanitized


def _sanitize_run_name(name: str) -> str:
    if not SAFE_RUN_NAME_RE.fullmatch(name):
        raise ValueError("runName may contain only letters, digits, dot, underscore, and dash")
    return name


def _write_inputs(job_root: Path, request: dict[str, Any]) -> tuple[Path, Path]:
    receptor_name = request["receptorFilename"]
    ligand_name = request["ligandFilename"]
    receptor_path = job_root / receptor_name
    ligand_path = job_root / ligand_name
    receptor_path.write_text(request["receptorText"])
    ligand_path.write_text(request["ligandText"])
    return receptor_path, ligand_path


def _write_optional_input(job_root: Path, filename: str | None, text: str | None) -> Path | None:
    if not filename or not text:
        return None
    path = job_root / _sanitize_leaf_name(filename, field="auxiliaryFilename")
    path.write_text(text)
    return path


def _protein_center_from_receptor(receptor_path: Path) -> tuple[float, float, float]:
    suffix = receptor_path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        structure = prody.parseMMCIF(str(receptor_path))
    else:
        structure = prody.parsePDB(str(receptor_path))
    if structure is None:
        raise ValueError(f"Failed to parse receptor structure: {receptor_path.name}")
    coords = structure.select("protein").getCoords() if structure.select("protein") is not None else structure.getCoords()
    if coords is None or len(coords) == 0:
        raise ValueError(f"Receptor contains no coordinates: {receptor_path.name}")
    center = np.mean(coords, axis=0)
    return float(center[0]), float(center[1]), float(center[2])


def _build_matcha_command(
    *,
    receptor_path: Path,
    ligand_path: Path,
    job_root: Path,
    run_parent: Path,
    run_name: str,
    params: dict[str, Any],
) -> list[str]:
    command = [
        "uv",
        "run",
        "matcha",
        "--receptor",
        str(receptor_path),
        "--ligand",
        str(ligand_path),
        "--out",
        str(run_parent),
        "--run-name",
        run_name,
        "--n-samples",
        str(int(params.get("nSamples", 20))),
        "--num-steps",
        str(int(params.get("numSteps", 10))),
        "--scorer",
        str(params.get("scorer", "none")),
        "--gnina-batch-mode",
        "combined",
        "--docking-batch-limit",
        "15000",
        "--num-dataloader-workers",
        "32",
        "--overwrite",
        "--keep-workdir",
        "--export-trace",
    ]
    device = params.get("device")
    device_value = device.strip() if isinstance(device, str) else ""
    command.extend(["--device", device_value or "cuda:0"])
    binding_site_mode = str(params.get("bindingSiteMode", "protein_center"))
    if binding_site_mode == "manual":
        center_x = params.get("centerX")
        center_y = params.get("centerY")
        center_z = params.get("centerZ")
        if None in {center_x, center_y, center_z}:
            raise ValueError("Manual binding site mode requires centerX/centerY/centerZ")
        command.extend(["--center-x", str(center_x), "--center-y", str(center_y), "--center-z", str(center_z)])
    elif binding_site_mode == "protein_center":
        center_x, center_y, center_z = _protein_center_from_receptor(receptor_path)
        command.extend(["--center-x", str(center_x), "--center-y", str(center_y), "--center-z", str(center_z)])
    elif binding_site_mode == "box_json":
        box_json_path = _write_optional_input(job_root, params.get("boxJsonFilename"), params.get("boxJsonText"))
        if box_json_path is None:
            raise ValueError("box_json binding site mode requires boxJsonFilename and boxJsonText")
        command.extend(["--box-json", str(box_json_path)])
    elif binding_site_mode == "autobox_ligand":
        autobox_ligand_path = _write_optional_input(job_root, params.get("autoboxLigandFilename"), params.get("autoboxLigandText"))
        if autobox_ligand_path is None:
            raise ValueError("autobox_ligand binding site mode requires autoboxLigandFilename and autoboxLigandText")
        command.extend(["--autobox-ligand", str(autobox_ligand_path)])
    elif binding_site_mode != "blind":
        raise ValueError(f"Unknown bindingSiteMode: {binding_site_mode}")
    command.append("--scorer-minimize" if bool(params.get("scorerMinimize", True)) else "--no-scorer-minimize")
    command.append("--physical-only" if bool(params.get("physicalOnly", False)) else "--keep-all-poses")
    return command


def _run_job(job_id: str, request: dict[str, Any]) -> None:
    try:
        with JOBS_LOCK:
            job = JOBS[job_id]
            job_root = job.root_dir

        receptor_path, ligand_path = _write_inputs(job_root, request)
        params = request.get("params", {})
        run_name = _sanitize_run_name(params.get("runName", "matcha-ui-run"))
        run_parent = job_root / "run"
        log_path = job_root / "matcha-live.log"
        _update_job(job_id, state="running", message="Running Matcha inference", log_path=log_path)
        env = os.environ.copy()
        env.setdefault("MATCHA_AUTO_DOWNLOAD_GNINA", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        command = _build_matcha_command(
            receptor_path=receptor_path,
            ligand_path=ligand_path,
            job_root=job_root,
            run_parent=run_parent,
            run_name=run_name,
            params=params,
        )
        with open(log_path, "w", buffering=1) as log_file:
            log_file.write(f"$ {' '.join(command)}\n\n")
            process = subprocess.Popen(
                command,
                cwd=_repo_root(),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _update_job(job_id, process=process)
            exit_code = process.wait()
        with JOBS_LOCK:
            job_after_wait = JOBS[job_id]
            cancel_requested = job_after_wait.cancel_requested
        _update_job(job_id, process=None)
        if cancel_requested:
            _update_job(job_id, state="cancelled", message="Matcha job was cancelled")
            return
        if exit_code != 0:
            tail, _ = _read_log_tail(log_path)
            raise RuntimeError(f"Matcha CLI exited with code {exit_code}\n{tail}")

        run_workdir = run_parent / run_name
        _update_job(job_id, state="running", message="Building visualization workspace")
        workspace = build_matcha_workspace(
            run_workdir=run_workdir,
            run_name=run_name,
            receptor_path=receptor_path,
            ligand_path=ligand_path,
            physical_only=bool(params.get("physicalOnly", False)),
        )
        result_path = job_root / "matcha-workspace.json"
        result_path.write_text(json.dumps(workspace))
        _update_job(job_id, state="completed", message="Matcha workspace is ready", result_path=result_path)
    except BaseException as exc:
        with JOBS_LOCK:
            job = JOBS[job_id]
            if job.cancel_requested:
                job.process = None
                job.state = "cancelled"
                job.message = "Matcha job was cancelled"
                job.error = None
                job.updated_at = time.time()
                return
        _update_job(job_id, state="failed", message="Matcha job failed", error=str(exc))


class MatchaHandler(BaseHTTPRequestHandler):
    server_version = "MatchaSpellHTTP/1.0"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok", "service": "matcha-ui-runner"})
            return

        if self.path == "/fixtures/default":
            try:
                self._send_json(HTTPStatus.OK, _default_fixture_payload())
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": str(exc)})
            return

        if self.path.startswith("/runs/"):
            parts = [part for part in self.path.split("/") if part]
            if len(parts) >= 2:
                job_id = parts[1]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                if job is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"Job {job_id} not found"})
                    return
                if len(parts) == 2:
                    self._send_json(HTTPStatus.OK, _job_payload(job))
                    return
                if len(parts) == 3 and parts[2] == "result":
                    if job.result_path is None or not job.result_path.exists():
                        self._send_json(HTTPStatus.CONFLICT, {"detail": f"Job {job_id} is not complete yet"})
                        return
                    self._send_json(HTTPStatus.OK, json.loads(job.result_path.read_text()))
                    return
                if len(parts) == 3 and parts[2] == "log":
                    text, truncated = _read_log_tail(job.log_path)
                    self._send_json(HTTPStatus.OK, {
                        "jobId": job.id,
                        "text": text,
                        "truncated": truncated,
                        "logPath": str(job.log_path) if job.log_path else None,
                    })
                    return

        self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"Unknown path {self.path}"})

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "Request body is required"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": f"Invalid JSON: {exc}"})
            return

        if self.path == "/smiles/preview":
            smiles = str(payload.get("smiles", "")).strip()
            if not smiles:
                self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "SMILES is required"})
                return
            try:
                self._send_json(HTTPStatus.OK, _smiles_preview_payload(smiles, payload.get("name")))
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": str(exc)})
            return

        if self.path != "/runs":
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"Unknown path {self.path}"})
            return

        required = ("receptorFilename", "receptorText", "ligandFilename", "ligandText")
        missing = [key for key in required if not payload.get(key)]
        if missing:
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": f"Missing required fields: {', '.join(missing)}"})
            return
        try:
            payload["receptorFilename"] = _sanitize_leaf_name(payload["receptorFilename"], field="receptorFilename")
            payload["ligandFilename"] = _sanitize_leaf_name(payload["ligandFilename"], field="ligandFilename")
            params = payload.setdefault("params", {})
            if "runName" in params and params["runName"]:
                params["runName"] = _sanitize_run_name(str(params["runName"]))
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
            return

        job_id = uuid.uuid4().hex[:12]
        job_root = _jobs_root() / job_id
        job_root.mkdir(parents=True, exist_ok=True)
        job = JobState(
            id=job_id,
            state="queued",
            message="Job accepted",
            created_at=time.time(),
            updated_at=time.time(),
            root_dir=job_root,
        )
        with JOBS_LOCK:
            JOBS[job_id] = job

        thread = threading.Thread(target=_run_job, args=(job_id, payload), daemon=True)
        thread.start()
        self._send_json(HTTPStatus.ACCEPTED, _job_payload(job))

    def do_DELETE(self) -> None:
        if not self.path.startswith("/runs/"):
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"Unknown path {self.path}"})
            return

        parts = [part for part in self.path.split("/") if part]
        if len(parts) != 2:
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"Unknown path {self.path}"})
            return

        job_id = parts[1]
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"Job {job_id} not found"})
                return
            if job.state in {"completed", "failed", "cancelled"}:
                self._send_json(HTTPStatus.CONFLICT, {"detail": f"Job {job_id} is already {job.state}"})
                return
            job.cancel_requested = True
            job.state = "cancelling"
            job.message = "Cancelling Matcha job"
            job.updated_at = time.time()
            process = job.process

        if process is not None:
            process.terminate()

        with JOBS_LOCK:
            payload = _job_payload(JOBS[job_id])
        self._send_json(HTTPStatus.OK, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Matcha docking jobs for the SPELL Mol* UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8899, help="Bind port")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MatchaHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
