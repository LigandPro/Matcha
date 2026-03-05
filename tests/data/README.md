# Test Fixtures

This folder contains local test fixtures for Matcha tests.

## Local baseline test

The local CLI reproducibility test compares the current run output against the
committed baseline in `tests/data/repro_baselines/local_cli_scores_filters_baseline.json`.

```bash
uv run pytest -n0 tests/test_cli_reproducibility.py::test_cli_matches_committed_baseline_for_scores_and_filters
```

## External baseline test

The external CLI reproducibility test is opt-in and compares the current run
against a baseline JSON generated from an accepted branch or commit.

```bash
MATCHA_RUN_EXTERNAL_REPRO_TEST=1 \
MATCHA_REPRO_CUDA_VISIBLE_DEVICES=7 \
MATCHA_REPRO_RECEPTOR=/mnt/ligandpro/db/datasets/posebusters/astex_diverse_set/1N1M_A3M/1N1M_A3M_protein.pdb \
MATCHA_REPRO_LIGAND=/mnt/ligandpro/db/datasets/posebusters/astex_diverse_set/1N1M_A3M/1N1M_A3M_ligand_start_conf.sdf \
MATCHA_REPRO_SCORER_PATH=/mnt/ligandpro/soft/docking/gnina/run_gnina.sh \
MATCHA_REPRO_BASELINE_JSON=/path/to/baseline.json \
uv run pytest -n0 tests/test_cli_reproducibility.py::test_cli_external_matches_baseline_with_real_assets
```

Baseline JSON format:

```json
{
  "scores": [-7.1, -6.9, -6.5],
  "best_score": -7.1,
  "filters": {
    "not_too_far_away": [true, true, true],
    "no_internal_clash": [true, true, true],
    "no_clashes": [true, false, true],
    "no_volume_clash": [true, false, true],
    "is_buried_fraction": [0.8, 0.5, 0.7],
    "posebusters_filters_passed_count_fast": [4, 2, 4]
  }
}
```
