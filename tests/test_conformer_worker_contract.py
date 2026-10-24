import json
import multiprocessing
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from matcha.utils.preprocessing import (
    _auto_setup_worker_once,
    _worker_setup_file_lock,
    generate_conformer_mols_batch,
)


def _arg_value(cmd: list[str], flag: str) -> str:
    idx = cmd.index(flag)
    return cmd[idx + 1]


def _hold_worker_lock(lock_path: str, shared_state, hold_seconds: float):
    with _worker_setup_file_lock(Path(lock_path)):
        shared_state.append(("enter", time.time()))
        time.sleep(hold_seconds)
        shared_state.append(("exit", time.time()))


def _write_mock_worker_outputs(cmd: list[str], confs_per_mol: int = 2):
    input_sdf = Path(_arg_value(cmd, "--input-sdf"))
    params_json = Path(_arg_value(cmd, "--params-json"))
    output_sdf = Path(_arg_value(cmd, "--output-sdf"))
    meta_json = Path(_arg_value(cmd, "--meta-json"))

    with open(params_json, "r", encoding="utf-8") as f:
        params = json.load(f)
    assert params["confs_per_mol"] == confs_per_mol

    supplier = Chem.SDMolSupplier(str(input_sdf), sanitize=True, removeHs=False)
    writer = Chem.SDWriter(str(output_sdf))
    writer.SetKekulize(False)
    input_count = 0
    output_count = 0
    for mol in supplier:
        assert mol is not None
        input_count += 1
        uid = mol.GetProp("_Name")
        base = Chem.Mol(mol)
        base = Chem.AddHs(base, addCoords=True)
        params_embed = AllChem.ETKDGv3()
        params_embed.randomSeed = 1
        AllChem.EmbedMultipleConfs(base, confs_per_mol, params_embed)
        for cid in range(min(confs_per_mol, base.GetNumConformers())):
            out = Chem.Mol(base)
            conf = base.GetConformer(cid)
            out.RemoveAllConformers()
            out.AddConformer(conf, assignId=True)
            out.SetProp("_Name", uid)
            out.SetProp("conf_id", str(cid))
            writer.write(out)
            output_count += 1
    writer.close()

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {"ok": True, "input_molecules": input_count, "written_conformers": output_count, "errors": []},
            f,
        )


def test_worker_backend_contract_success(monkeypatch):
    def fake_run(cmd, check, capture_output, text, timeout):
        _write_mock_worker_outputs(cmd, confs_per_mol=2)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setenv("MATCHA_CONFORMER_WORKER_CMD", "fake_worker")
    monkeypatch.setattr("matcha.utils.preprocessing.subprocess.run", fake_run)

    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
    output = generate_conformer_mols_batch(mols, confs_per_mol=2, backend="worker", chunk_size=2)

    assert len(output) == 2
    assert all(len(conf_list) == 2 for conf_list in output)
    assert all(mol.GetNumConformers() == 1 for conf_list in output for mol in conf_list)


def test_worker_backend_fail_fast(monkeypatch):
    def failing_run(cmd, check, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 1, "", "mock worker failure")

    monkeypatch.setenv("MATCHA_CONFORMER_WORKER_CMD", "fake_worker")
    monkeypatch.setattr("matcha.utils.preprocessing.subprocess.run", failing_run)

    mols = [Chem.MolFromSmiles("CCO")]
    with pytest.raises(RuntimeError, match="Worker conformer generation failed"):
        generate_conformer_mols_batch(mols, confs_per_mol=2, backend="worker", chunk_size=1)


def test_auto_backend_fallback_to_rdkit(monkeypatch):
    def failing_run(cmd, check, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 1, "", "worker unavailable")

    monkeypatch.setenv("MATCHA_CONFORMER_WORKER_CMD", "fake_worker")
    monkeypatch.setattr("matcha.utils.preprocessing.subprocess.run", failing_run)

    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCN")]
    output = generate_conformer_mols_batch(mols, confs_per_mol=2, backend="auto", chunk_size=2)

    assert len(output) == 2
    assert all(len(conf_list) >= 1 for conf_list in output)
    assert all(mol.GetNumConformers() == 1 for conf_list in output for mol in conf_list)


def test_auto_backend_falls_back_when_worker_returns_invalid_sdf(monkeypatch):
    def fake_run(cmd, check, capture_output, text, timeout):
        output_sdf = Path(_arg_value(cmd, "--output-sdf"))
        meta_json = Path(_arg_value(cmd, "--meta-json"))
        input_sdf = Path(_arg_value(cmd, "--input-sdf"))

        supplier = Chem.SDMolSupplier(str(input_sdf), sanitize=False, removeHs=False)
        writer = Chem.SDWriter(str(output_sdf))
        writer.SetKekulize(False)
        for mol in supplier:
            assert mol is not None
            uid = mol.GetProp("_Name")
            out = Chem.Mol(mol)
            conf = out.GetConformer(0)
            conf.SetAtomPosition(0, (float("nan"), 0.0, 0.0))
            out.SetProp("_Name", uid)
            out.SetProp("conf_id", "0")
            writer.write(out)
        writer.close()
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump({"ok": True, "errors": []}, f)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setenv("MATCHA_CONFORMER_WORKER_CMD", "fake_worker")
    monkeypatch.setattr("matcha.utils.preprocessing.subprocess.run", fake_run)

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"), addCoords=True)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    output = generate_conformer_mols_batch([mol], confs_per_mol=1, backend="auto", chunk_size=1)

    assert len(output) == 1
    assert len(output[0]) == 1
    out_mol = output[0][0]
    assert out_mol.GetNumConformers() == 1
    coords = out_mol.GetConformer(0).GetPositions()
    assert np.isfinite(coords).all()


def test_auto_setup_worker_once_reuses_successful_setup(monkeypatch, tmp_path):
    state = {"ready": False, "calls": 0}

    def fake_worker_command_configured():
        return state["ready"]

    def fake_run(cmd, cwd, check, capture_output, text, timeout):
        state["calls"] += 1
        time.sleep(0.1)
        state["ready"] = True
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("matcha.utils.preprocessing._WORKER_AUTO_SETUP_ATTEMPTED", False)
    monkeypatch.setattr("matcha.utils.preprocessing._WORKER_AUTO_SETUP_SUCCEEDED", False)
    monkeypatch.setattr("matcha.utils.preprocessing._worker_command_configured", fake_worker_command_configured)
    monkeypatch.setattr("matcha.utils.preprocessing.subprocess.run", fake_run)
    monkeypatch.setattr("matcha.utils.preprocessing._env_flag", lambda *args, **kwargs: True)
    monkeypatch.setattr("matcha.utils.preprocessing.shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setenv("MATCHA_CONFORMER_AUTO_SETUP_WORKER", "1")

    repo_root = tmp_path / "repo"
    setup_script = repo_root / "scripts" / "setup_nvmolkit_worker.py"
    setup_script.parent.mkdir(parents=True, exist_ok=True)
    setup_script.write_text("print('ok')\n", encoding="utf-8")

    original_resolve = Path.resolve

    def fake_resolve(path_self, *args, **kwargs):
        if path_self.name == "preprocessing.py":
            return repo_root / "matcha" / "utils" / "preprocessing.py"
        return original_resolve(path_self, *args, **kwargs)

    monkeypatch.setattr("pathlib.Path.resolve", fake_resolve)

    results = []

    def run_setup():
        results.append(_auto_setup_worker_once())

    threads = [threading.Thread(target=run_setup) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [True, True]
    assert state["calls"] == 1


def test_worker_setup_file_lock_excludes_other_processes(tmp_path):
    lock_path = tmp_path / "worker.setup.lock"
    manager = multiprocessing.Manager()
    events = manager.list()

    processes = [
        multiprocessing.Process(target=_hold_worker_lock, args=(str(lock_path), events, 0.2)),
        multiprocessing.Process(target=_hold_worker_lock, args=(str(lock_path), events, 0.2)),
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0

    event_list = list(events)
    assert len(event_list) == 4

    intervals = []
    current_start = None
    for event_type, timestamp in event_list:
        if event_type == "enter":
            current_start = timestamp
        else:
            intervals.append((current_start, timestamp))
            current_start = None

    assert len(intervals) == 2
    intervals.sort()
    assert intervals[1][0] >= intervals[0][1]
