from pathlib import Path

from matcha.utils.multigpu import build_shard_command


def test_build_shard_command_contains_required_gpu_flags():
    cmd = build_shard_command(
        receptor=Path("/tmp/receptor.pdb"),
        shard_ligand_dir=Path("/tmp/shard_00"),
        out_dir=Path("/tmp/out"),
        run_name="run_gpu2",
        checkpoints=Path("/tmp/checkpoints"),
        config=Path("/tmp/config.yaml"),
        center_x=None,
        center_y=None,
        center_z=None,
        autobox_ligand=None,
        box_json=Path("/tmp/box.json"),
        n_samples=10,
        n_confs=None,
        docking_batch_limit=1234,
        recursive=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        scorer_type="gnina",
        scorer_path=None,
        scorer_minimize=True,
        gnina_batch_mode="combined",
        physical_only=False,
    )

    assert cmd[:3] == ["uv", "run", "matcha"]
    assert "--ligand-dir" in cmd
    assert "--device" in cmd
    assert cmd[cmd.index("--device") + 1] == "cuda:0"
    assert "--keep-workdir" in cmd
    assert "--overwrite" in cmd
    assert "--box-json" in cmd
    assert "--num-workers" in cmd
    assert "--pin-memory" in cmd
    assert "--persistent-workers" in cmd
    assert "--gnina-batch-mode" in cmd
