# Analogue/FEP implementation report

Date: 2026-04-27

## Implemented scope

This patch adds a production-facing analogue/FEP seed mode to Matcha.  It is intentionally split into a lightweight RDKit front-end and an optional Matcha neural refinement path.

### New package

```text
matcha/analogue/
  __init__.py
  standardize.py
  mcs.py
  constrained_embed.py
  torsion_mc.py
  ranking.py
  fep_export.py
  workflow.py
```

### Main features

- robust MCS fallback ladder:
  - strict;
  - stereo-relaxed;
  - aromatic/bond-order relaxed;
  - scaffold fallback;
  - last-resort heavy-atom mapping;
- template-constrained conformer generation with MCS coordinate anchors;
- optional torsional Monte Carlo outside the MCS core;
- FEP-aware pose ranking by core RMSD, MCS coverage, internal clashes, and strain estimate;
- `analogue_seed_transforms.npy` for feeding full initial poses into Matcha stage 3;
- seed FEP bundle export:
  - `aligned_series.sdf`;
  - `quality_report.csv`;
  - `mcs_mappings.json`;
  - `fep_manifest.json`;
  - `rbfe_graph.json`;
  - per-ligand `complexes/<ligand_id>/best_pose.sdf`, `poses.sdf`, `quality.json`, `atom_mapping.json`;
- refined FEP bundle export from final Matcha best poses when neural refinement is enabled;
- deterministic duplicate ligand UID handling: `name`, `name__2`, `name__3`, ...;
- fixed CLI `--n-confs` propagation into `conf.n_confs_override`;
- extended inference pipeline to support:
  - `initial_pose_transforms_path`;
  - `start_stage=1/2/3`;
  - merging partial stage outputs, including stage-3-only analogue refinement.

## CLI examples

Seed-only mode:

```bash
uv run matcha \
  -r protein.pdb \
  --ligand-dir analogs.sdf \
  --analogue-template reference_bound.sdf \
  --analogue-only \
  -o out \
  --run-name analogue_seed
```

Seed plus Matcha stage-3 refinement:

```bash
uv run matcha \
  -r protein.pdb \
  --ligand-dir analogs.sdf \
  --analogue-template reference_bound.sdf \
  -o out \
  --run-name analogue_refined \
  --n-samples 64 \
  --analogue-start-stage 3
```

## Verification performed in this environment

The container does not provide the full heavy Matcha runtime in a normal Python invocation, so verification was split into syntax checks and targeted RDKit analogue checks.

Commands that passed:

```bash
/usr/bin/python3 -m compileall -q matcha scripts tests
```

Direct analogue workflow tests passed under the available RDKit environment:

```bash
PYTHONPATH=.:/opt/pyvenv/lib/python3.13/site-packages \
/opt/pyvenv/bin/python -S - <<'PY'
from pathlib import Path
import tempfile
from tests.test_analogue_workflow import (
    test_robust_mcs_maps_congeneric_core,
    test_analogue_workflow_writes_fep_bundle,
)
test_robust_mcs_maps_congeneric_core()
test_analogue_workflow_writes_fep_bundle(Path(tempfile.mkdtemp()))
PY
```

CLI analogue-only smoke also completed with status 0 on a tiny generated receptor/template/analogue set and wrote:

```text
analogue/analogue_seed_transforms.npy
analogue/fep_bundle_seed/aligned_series.sdf
analogue/fep_bundle_seed/quality_report.csv
analogue/fep_bundle_seed/fep_manifest.json
analogue/fep_bundle_seed/rbfe_graph.json
```

## Honest limitations

- Full neural Matcha stage-3 refinement was not run here because model checkpoints and the full scientific stack are not available in this container.
- g-xTB is not implemented in this patch; the hook remains conceptual and should be added later as optional top-N strain/refinement QC.
- The FEP manifest is generic/OpenFE-friendly metadata, not a complete OpenFE execution script.
- The RBFE graph is a recommended starting graph; force-field parameterization, solvent/complex setup, minimization, and RBFE execution must still be validated downstream.
