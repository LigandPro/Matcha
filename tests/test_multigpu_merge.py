import json
from pathlib import Path

import numpy as np
import pytest

from matcha.utils.multigpu import merge_shard_outputs


def _prepare_shard_run(root: Path, run_name: str, uid: str) -> dict:
    run_dir = root / run_name
    (run_dir / "best_poses").mkdir(parents=True, exist_ok=True)
    (run_dir / "all_poses").mkdir(parents=True, exist_ok=True)
    (run_dir / "best_poses" / f"{uid}.sdf").write_text("best", encoding="utf-8")
    (run_dir / "all_poses" / f"{uid}_poses.sdf").write_text("all", encoding="utf-8")

    shard_inner = run_dir / "work" / "runs" / run_name / "any_conf"
    (shard_inner / "best_scored_predictions").mkdir(parents=True, exist_ok=True)
    (shard_inner / "scored_sdf_predictions").mkdir(parents=True, exist_ok=True)
    (shard_inner / "best_scored_predictions" / f"{uid}.sdf").write_text("best_scored", encoding="utf-8")
    (shard_inner / "scored_sdf_predictions" / f"{uid}.sdf").write_text("scored", encoding="utf-8")

    with open(shard_inner / "filters_results.json", "w", encoding="utf-8") as f:
        json.dump({uid: {"posebusters_filters_passed_count_fast": [4]}}, f)
    np.save(run_dir / "work" / "runs" / run_name / "any_conf_final_preds.npy", [{uid: {"sample_metrics": []}}])

    return {"run_dir": str(run_dir), "run_name": run_name, "gpu_id": 0, "ligand_count": 1}


def test_merge_shard_outputs_success(tmp_path: Path):
    shard_a = _prepare_shard_run(tmp_path, "run_a", "lig_a")
    shard_b = _prepare_shard_run(tmp_path, "run_b", "lig_b")

    merged = merge_shard_outputs(
        shard_records=[shard_a, shard_b],
        merged_root=tmp_path / "merged",
        expected_ligands=2,
    )

    assert merged["merged_prediction_uids"] == 2
    assert (tmp_path / "merged" / "best_poses" / "lig_a.sdf").exists()
    assert (tmp_path / "merged" / "best_poses" / "lig_b.sdf").exists()
    assert (tmp_path / "merged" / "any_conf_final_preds.npy").exists()
    assert (tmp_path / "merged" / "filters_results.json").exists()


def test_merge_shard_outputs_duplicate_uid_raises(tmp_path: Path):
    shard_a = _prepare_shard_run(tmp_path, "run_a", "dup")
    shard_b = _prepare_shard_run(tmp_path, "run_b", "dup")

    with pytest.raises(RuntimeError, match="Duplicate uid"):
        merge_shard_outputs(
            shard_records=[shard_a, shard_b],
            merged_root=tmp_path / "merged",
            expected_ligands=2,
        )
