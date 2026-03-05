# Test Fixtures

This folder contains local test fixtures for Matcha tests.

## Reproducibility E2E (external assets)

The external CLI reproducibility test is opt-in and compares `scores + filters`
between two identical runs.

```bash
MATCHA_RUN_EXTERNAL_REPRO_TEST=1 \
MATCHA_REPRO_CUDA_VISIBLE_DEVICES=7 \
MATCHA_REPRO_RECEPTOR=/mnt/ligandpro/db/datasets/posebusters/astex_diverse_set/1N1M_A3M/1N1M_A3M_protein.pdb \
MATCHA_REPRO_LIGAND=/mnt/ligandpro/db/datasets/posebusters/astex_diverse_set/1N1M_A3M/1N1M_A3M_ligand_start_conf.sdf \
MATCHA_REPRO_SCORER_PATH=/mnt/ligandpro/soft/docking/gnina/run_gnina.sh \
uv run pytest -n0 tests/test_cli_reproducibility.py::test_cli_external_reproducibility_with_real_assets
```

The local deterministic test (`test_cli_reproducible_scores_and_filters_with_fixed_seed`) does not require `/mnt` assets.
