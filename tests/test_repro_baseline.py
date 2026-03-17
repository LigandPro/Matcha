import json
import subprocess
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from matcha.utils.repro_baseline import (
    collect_repro_snapshot,
    create_baseline_from_git_ref,
    extract_pose_scores,
    write_baseline_json,
)


def _write_scored_pose_sdf(path: Path, scores: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    writer = Chem.SDWriter(str(path))
    writer.SetKekulize(False)
    for score in scores:
        current = Chem.Mol(mol)
        current.SetProp("Affinity", str(score))
        writer.write(current)
    writer.close()


def _prepare_run_artifacts(run_dir: Path, run_name: str) -> dict:
    any_conf_dir = run_dir / "work" / "runs" / run_name / "any_conf"
    _write_scored_pose_sdf(any_conf_dir / "scored_sdf_predictions" / f"{run_name}.sdf", [-7.2, -6.8, -6.5])
    _write_scored_pose_sdf(any_conf_dir / "best_scored_predictions" / f"{run_name}.sdf", [-7.2])
    filters = {
        run_name: {
            "not_too_far_away": [True, True, True],
            "no_internal_clash": [True, False, True],
            "no_clashes": [True, True, True],
            "no_volume_clash": [True, False, True],
            "is_buried_fraction": [0.8, 0.5, 0.7],
            "posebusters_filters_passed_count_fast": [4, 2, 4],
        }
    }
    filters_path = any_conf_dir / "filters_results_minimized.json"
    filters_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filters_path, "w", encoding="utf-8") as f:
        json.dump(filters, f, indent=2)
    return {
        "scores": [-7.2, -6.8, -6.5],
        "best_score": -7.2,
        "filters": filters[run_name],
    }


def test_collect_repro_snapshot_reads_expected_format(tmp_path: Path):
    expected = _prepare_run_artifacts(tmp_path / "output" / "baseline_run", "baseline_run")

    snapshot = collect_repro_snapshot(tmp_path / "output" / "baseline_run", "baseline_run")

    assert snapshot == expected
    assert extract_pose_scores(
        tmp_path / "output" / "baseline_run" / "work" / "runs" / "baseline_run" / "any_conf"
        / "scored_sdf_predictions" / "baseline_run.sdf"
    ) == expected["scores"]


def test_create_baseline_from_git_ref_writes_json(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    output_json = tmp_path / "baseline.json"
    expected = {
        "scores": [-7.2, -6.8, -6.5],
        "best_score": -7.2,
        "filters": {
            "not_too_far_away": [True, True, True],
            "no_internal_clash": [True, False, True],
            "no_clashes": [True, True, True],
            "no_volume_clash": [True, False, True],
            "is_buried_fraction": [0.8, 0.5, 0.7],
            "posebusters_filters_passed_count_fast": [4, 2, 4],
        },
    }

    def fake_run(cmd, cwd=None, check=False, capture_output=False, text=False, env=None):
        if cmd[:4] == ["git", "worktree", "add", "--detach"]:
            worktree_dir = Path(cmd[4])
            worktree_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:4] == ["git", "worktree", "remove", "--force"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:3] == ["uv", "run", "matcha"]:
            run_name = cmd[cmd.index("--run-name") + 1]
            output_root = Path(cmd[cmd.index("-o") + 1])
            _prepare_run_artifacts(output_root / run_name, run_name)
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("matcha.utils.repro_baseline.subprocess.run", fake_run)

    snapshot = create_baseline_from_git_ref(
        repo_root=repo_root,
        git_ref="origin/main",
        receptor=Path("/tmp/receptor.pdb"),
        ligand=Path("/tmp/ligand.sdf"),
        scorer_path=Path("/tmp/scorer.sh"),
        output_json=output_json,
        run_name="baseline_run",
        cuda_visible_devices="7",
        extra_args=["--physical-only"],
    )

    assert snapshot == expected
    with open(output_json, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved == expected


def test_write_baseline_json_persists_expected_payload(tmp_path: Path):
    output_json = tmp_path / "nested" / "baseline.json"
    payload = {
        "scores": [1.0, 2.0],
        "best_score": 1.0,
        "filters": {
            "not_too_far_away": [True, False],
            "no_internal_clash": [True, True],
            "no_clashes": [True, True],
            "no_volume_clash": [True, False],
            "is_buried_fraction": [0.4, 0.2],
            "posebusters_filters_passed_count_fast": [4, 3],
        },
    }

    write_baseline_json(payload, output_json)

    with open(output_json, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved == payload
