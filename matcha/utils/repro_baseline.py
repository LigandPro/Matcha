from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem


SCORE_PROPERTIES = ("minimizedAffinity", "Affinity", "minimizedCNNscore", "CNNscore")


def extract_pose_scores(sdf_path: Path) -> list[float]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    scores: list[float] = []
    for mol in supplier:
        if mol is None:
            scores.append(float("nan"))
            continue
        value = None
        for prop in SCORE_PROPERTIES:
            if mol.HasProp(prop):
                try:
                    value = float(mol.GetProp(prop))
                    break
                except (TypeError, ValueError):
                    continue
        if value is None:
            raise AssertionError(f"No score property found in {sdf_path}")
        scores.append(value)
    return scores


def read_best_score(sdf_path: Path) -> float:
    values = extract_pose_scores(sdf_path)
    if not values:
        raise AssertionError(f"No poses in {sdf_path}")
    return values[0]


def collect_repro_snapshot(run_dir: Path, run_name: str) -> dict:
    any_conf_dir = run_dir / "work" / "runs" / run_name / "any_conf"
    filters_path = any_conf_dir / "filters_results_minimized.json"

    scored_path = None
    best_scored_path = None
    for scored_dir_name, best_dir_name in (
        ("scored_sdf_predictions", "best_scored_predictions"),
        ("minimized_sdf_predictions", "best_minimized_predictions"),
    ):
        candidate_scored = any_conf_dir / scored_dir_name / f"{run_name}.sdf"
        candidate_best = any_conf_dir / best_dir_name / f"{run_name}.sdf"
        if candidate_scored.exists() and candidate_best.exists():
            scored_path = candidate_scored
            best_scored_path = candidate_best
            break

    assert scored_path is not None, f"Missing scored SDF under {any_conf_dir}"
    assert best_scored_path is not None, f"Missing best scored SDF under {any_conf_dir}"
    assert filters_path.exists(), f"Missing filters JSON: {filters_path}"

    with open(filters_path, encoding="utf-8") as f:
        filters_all = json.load(f)

    uid = next(iter(filters_all.keys()))
    return {
        "scores": extract_pose_scores(scored_path),
        "best_score": read_best_score(best_scored_path),
        "filters": filters_all[uid],
    }


def load_baseline_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_baseline_json(snapshot: dict, output_json: Path) -> Path:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    return output_json


def build_matcha_command(
    *,
    receptor: Path,
    ligand: Path,
    output_root: Path,
    run_name: str,
    scorer_path: Path,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "matcha",
        "-r",
        str(receptor),
        "-l",
        str(ligand),
        "--run-name",
        run_name,
        "--overwrite",
        "-o",
        str(output_root),
        "--scorer-path",
        str(scorer_path),
        "--keep-workdir",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def create_baseline_from_git_ref(
    *,
    repo_root: Path,
    git_ref: str,
    receptor: Path,
    ligand: Path,
    scorer_path: Path,
    output_json: Path,
    run_name: str = "repro_baseline",
    cuda_visible_devices: str | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    repo_root = repo_root.resolve()
    output_json = output_json.resolve()

    if shutil.which("git") is None:
        raise RuntimeError("git is required to generate a baseline from a reference branch")

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    with tempfile.TemporaryDirectory(prefix="matcha_repro_ref_") as tmp_dir:
        worktree_dir = Path(tmp_dir) / "worktree"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree_dir), git_ref],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            output_root = worktree_dir / ".tmp_repro_baseline_output"
            cmd = build_matcha_command(
                receptor=receptor,
                ligand=ligand,
                output_root=output_root,
                run_name=run_name,
                scorer_path=scorer_path,
                extra_args=extra_args,
            )
            completed = subprocess.run(
                cmd,
                cwd=worktree_dir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Reference branch command failed.\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDOUT:\n{completed.stdout}\n"
                    f"STDERR:\n{completed.stderr}"
                )
            snapshot = collect_repro_snapshot(output_root / run_name, run_name)
            write_baseline_json(snapshot, output_json)
            return snapshot
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_dir)],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
