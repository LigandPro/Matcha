import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from matcha.cli import run_matcha


FIXTURE_ROOT = Path(__file__).resolve().parent / "data" / "complexes" / "1HVY_D16"
FIXTURE_RECEPTOR = FIXTURE_ROOT / "1HVY_D16_protein.pdb"
FIXTURE_LIGAND = FIXTURE_ROOT / "1HVY_D16_ligand_start_conf.sdf"


def _extract_pose_scores(sdf_path: Path) -> list[float]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    scores: list[float] = []
    for mol in supplier:
        if mol is None:
            scores.append(float("nan"))
            continue
        value = None
        for prop in ("minimizedAffinity", "Affinity", "minimizedCNNscore", "CNNscore"):
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


def _read_first_score(sdf_path: Path) -> float:
    values = _extract_pose_scores(sdf_path)
    if not values:
        raise AssertionError(f"No poses in {sdf_path}")
    return values[0]


def _collect_repro_snapshot(run_dir: Path, run_name: str) -> dict:
    any_conf_dir = run_dir / "work" / "runs" / run_name / "any_conf"
    scored_path = any_conf_dir / "scored_sdf_predictions" / f"{run_name}.sdf"
    best_scored_path = any_conf_dir / "best_scored_predictions" / f"{run_name}.sdf"
    filters_path = any_conf_dir / "filters_results_minimized.json"

    assert scored_path.exists(), f"Missing scored SDF: {scored_path}"
    assert best_scored_path.exists(), f"Missing best scored SDF: {best_scored_path}"
    assert filters_path.exists(), f"Missing filters JSON: {filters_path}"

    with open(filters_path, encoding="utf-8") as f:
        filters_all = json.load(f)

    uid = next(iter(filters_all.keys()))
    return {
        "scores": _extract_pose_scores(scored_path),
        "best_score": _read_first_score(best_scored_path),
        "filters": filters_all[uid],
    }


def _prepare_fake_checkpoints(root: Path) -> Path:
    checkpoints = root / "checkpoints"
    stage1 = checkpoints / "matcha_pipeline" / "stage1"
    stage1.mkdir(parents=True, exist_ok=True)
    (stage1 / "model.safetensors").write_text("fake", encoding="utf-8")
    return checkpoints


def _write_scored_copy(src_sdf: Path, dst_sdf: Path) -> None:
    dst_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(dst_sdf))
    writer.SetKekulize(False)
    supplier = Chem.SDMolSupplier(str(src_sdf), removeHs=False, sanitize=False)
    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        scored = Chem.Mol(mol)
        scored.SetProp("CNNscore", f"{100.0 - idx:.6f}")
        writer.write(scored)
    writer.close()


class _FakeScorer:
    name = "fake-custom"

    def score_poses(self, receptor_path, sdf_input_dir, sdf_output_dir, device=0):
        in_dir = Path(sdf_input_dir)
        out_dir = Path(sdf_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for sdf_file in sorted(in_dir.glob("*.sdf")):
            _write_scored_copy(sdf_file, out_dir / sdf_file.name)

    def select_top_poses(self, sdf_dir, output_dir, filters_path=None, n_samples=40):
        in_dir = Path(sdf_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filters = {}
        if filters_path is not None and Path(filters_path).exists():
            with open(filters_path, encoding="utf-8") as f:
                filters = json.load(f)

        for sdf_file in sorted(in_dir.glob("*.sdf")):
            uid = sdf_file.stem
            supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
            mols = [m for m in supplier if m is not None]
            if not mols:
                continue

            scores = [float(m.GetProp("CNNscore")) for m in mols]
            valid_indices = list(range(len(mols)))
            if uid in filters:
                counts = filters[uid].get("posebusters_filters_passed_count_fast", [])
                if counts:
                    best_count = max(int(c) for c in counts[: len(mols)])
                    valid_indices = [
                        i for i in range(min(len(mols), len(counts))) if int(counts[i]) == best_count
                    ]
            if not valid_indices:
                valid_indices = list(range(len(mols)))

            best_idx = max(valid_indices, key=lambda idx: scores[idx])
            writer = Chem.SDWriter(str(out_dir / f"{uid}.sdf"))
            writer.SetKekulize(False)
            writer.write(mols[best_idx])
            writer.close()


def _fake_run_v2_inference_pipeline(
    conf,
    run_name,
    n_preds_to_use,
    pocket_centers_filename=None,
    docking_batch_limit=15000,
    num_workers=0,
    progress_callback=None,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=False,
    compute_torsion_angles_pred=False,
):
    run_root = Path(conf.inference_results_folder) / run_name
    any_conf_dir = run_root / "any_conf"
    sdf_predictions_dir = any_conf_dir / "sdf_predictions"
    sdf_predictions_dir.mkdir(parents=True, exist_ok=True)

    ligand_path = Path(conf.any_data_dir) / run_name / f"{run_name}_ligand.sdf"
    supplier = Chem.SDMolSupplier(str(ligand_path), removeHs=False, sanitize=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise AssertionError(f"Failed to read prepared ligand for fake pipeline: {ligand_path}")

    base_pos = mol.GetConformer().GetPositions().astype(float)
    sample_metrics = []

    writer = Chem.SDWriter(str(sdf_predictions_dir / f"{run_name}.sdf"))
    writer.SetKekulize(False)

    for idx in range(n_preds_to_use):
        pred_pos = base_pos + np.array([idx * 0.01, 0.0, 0.0])
        sample_metrics.append({"pred_pos": pred_pos, "error_estimate_0": float(idx)})

        pose = Chem.Mol(mol)
        conf_new = Chem.Conformer(pose.GetNumAtoms())
        for atom_idx, (x, y, z) in enumerate(pred_pos):
            conf_new.SetAtomPosition(atom_idx, (float(x), float(y), float(z)))
        pose.RemoveAllConformers()
        pose.AddConformer(conf_new, assignId=True)
        pose.SetProp("_Name", f"{run_name}_pose_{idx}")
        writer.write(pose)

    writer.close()

    metrics = {
        f"{run_name}_mol0": {
            "orig_mol": Chem.Mol(mol),
            "sample_metrics": sample_metrics,
        }
    }
    np.save(run_root / "any_conf_final_preds_merged.npy", [metrics])

    filter_counts = [4 if idx % 2 == 0 else 3 for idx in range(n_preds_to_use)]
    filters_payload = {
        run_name: {
            "not_too_far_away": [True for _ in range(n_preds_to_use)],
            "no_internal_clash": [idx % 2 == 0 for idx in range(n_preds_to_use)],
            "no_clashes": [True for _ in range(n_preds_to_use)],
            "no_volume_clash": [idx % 2 == 0 for idx in range(n_preds_to_use)],
            "is_buried_fraction": [0.42 for _ in range(n_preds_to_use)],
            "posebusters_filters_passed_count_fast": filter_counts,
        }
    }
    with open(any_conf_dir / "filters_results.json", "w", encoding="utf-8") as f:
        json.dump(filters_payload, f, indent=2)
    with open(any_conf_dir / "filters_results_minimized.json", "w", encoding="utf-8") as f:
        json.dump(filters_payload, f, indent=2)

    return {
        "stage1_sec": 0.0,
        "stage2_sec": 0.0,
        "stage3_sec": 0.0,
        "sdf_save_sec": 0.0,
        "stage_by_dataset": {"any_conf": {}},
    }


def _run_local_cli_repro_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str) -> dict:
    checkpoints = _prepare_fake_checkpoints(tmp_path)
    output_root = tmp_path / "out"

    fake_scorer_script = tmp_path / "fake_scorer.sh"
    fake_scorer_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_scorer_script.chmod(0o755)

    monkeypatch.setattr("matcha.utils.esm_utils.compute_sequences", lambda conf: None)
    monkeypatch.setattr("matcha.utils.esm_utils.compute_esm_embeddings", lambda conf: None)
    monkeypatch.setattr("matcha.utils.inference_utils.run_v2_inference_pipeline", _fake_run_v2_inference_pipeline)
    monkeypatch.setattr("matcha.scoring.create_scorer", lambda *args, **kwargs: _FakeScorer())

    run_matcha(
        receptor=FIXTURE_RECEPTOR,
        ligand=FIXTURE_LIGAND,
        ligand_dir=None,
        out=output_root,
        checkpoints=checkpoints,
        config=None,
        workdir=None,
        center_x=None,
        center_y=None,
        center_z=None,
        autobox_ligand=None,
        box_json=None,
        run_name=run_name,
        n_samples=4,
        n_confs=None,
        device="cpu",
        gpus=None,
        overwrite=True,
        keep_workdir=True,
        log=None,
        docking_batch_limit=15000,
        recursive=True,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
        scorer_type="custom",
        scorer_path=fake_scorer_script,
        scorer_minimize=True,
        gnina_batch_mode="combined",
        physical_only=False,
        num_dataloader_workers=0,
    )

    return _collect_repro_snapshot(output_root / run_name, run_name)


def test_cli_reproducible_scores_and_filters_with_fixed_seed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    first = _run_local_cli_repro_case(tmp_path / "run1", monkeypatch, "repro_case")
    second = _run_local_cli_repro_case(tmp_path / "run2", monkeypatch, "repro_case")

    assert first == second


@pytest.mark.skipif(os.getenv("MATCHA_RUN_EXTERNAL_REPRO_TEST") != "1", reason="external repro test is opt-in")
def test_cli_external_reproducibility_with_real_assets(tmp_path: Path):
    receptor = Path(os.environ.get("MATCHA_REPRO_RECEPTOR", ""))
    ligand = Path(os.environ.get("MATCHA_REPRO_LIGAND", ""))
    scorer_path = Path(os.environ.get("MATCHA_REPRO_SCORER_PATH", ""))

    required = {
        "MATCHA_REPRO_RECEPTOR": receptor,
        "MATCHA_REPRO_LIGAND": ligand,
        "MATCHA_REPRO_SCORER_PATH": scorer_path,
    }
    missing = [name for name, path in required.items() if not str(path) or not path.exists()]
    if missing:
        pytest.skip("Missing external repro inputs: " + ", ".join(missing))

    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "external"

    common = [
        "uv", "run", "matcha",
        "-r", str(receptor),
        "-l", str(ligand),
        "--overwrite",
        "-o", str(output_root),
        "--scorer-path", str(scorer_path),
        "--keep-workdir",
    ]

    env = os.environ.copy()
    if "MATCHA_REPRO_CUDA_VISIBLE_DEVICES" in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = os.environ["MATCHA_REPRO_CUDA_VISIBLE_DEVICES"]

    for run_name in ("repro_ext_a", "repro_ext_b"):
        cmd = common + ["--run-name", run_name]
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "External command failed.\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

    snap_a = _collect_repro_snapshot(output_root / "repro_ext_a", "repro_ext_a")
    snap_b = _collect_repro_snapshot(output_root / "repro_ext_b", "repro_ext_b")

    assert snap_a == snap_b
