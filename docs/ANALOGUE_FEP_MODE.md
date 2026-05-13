# Matcha Analogue/FEP mode

Matcha now includes a template-aware analogue docking front-end for congeneric ligand series.  The mode is designed to produce FEP/RBFE-ready starting structures before, or together with, the normal neural Matcha refinement path.

## CLI

Seed-only FEP bundle:

```bash
matcha \
  -r protein.pdb \
  --ligand-dir analogs.sdf \
  --analogue-template reference_bound.sdf \
  --analogue-only \
  -o out \
  --run-name analogue_seed
```

Seed plus Matcha local refinement:

```bash
matcha \
  -r protein.pdb \
  --ligand-dir analogs.sdf \
  --analogue-template reference_bound.sdf \
  -o out \
  --run-name analogue_refined \
  --n-samples 64 \
  --analogue-start-stage 3 \
  --analogue-core-rmsd-cutoff 1.0
```

## Workflow

1. Standardize template and analogues conservatively without tautomer/protomer edits.
2. Find robust MCS using a fallback ladder:
   - strict atom/bond/stereo/ring matching;
   - stereo-relaxed;
   - aromatic/bond-order relaxed;
   - scaffold fallback;
   - last-resort heavy-atom mapping.
3. Generate template-constrained conformers with RDKit ETKDG and MCS coordinate anchors.
4. Optionally diversify poses using torsional Monte Carlo outside the MCS core.
5. Rank poses with FEP-aware gates: MCS coverage, core RMSD, internal clashes, strain estimate, and receptor clash/contact terms when a receptor is supplied.
6. Export a generic FEP bundle and an RBFE graph manifest.
7. Optionally feed full seed poses into Matcha stage 3 via `analogue_seed_transforms.npy`.

## Output

Seed-only output is written under:

```text
<out>/<run-name>/analogue/
  analogue_seed_transforms.npy
  analogue_mappings.json
  analogue_failures.json
  analogue_summary.json
  fep_bundle_seed/
    aligned_series.sdf
    quality_report.csv
    fep_manifest.json
    rbfe_graph.json
    mcs_mappings.json
    failures.json
    complexes/<ligand_id>/best_pose.sdf
    complexes/<ligand_id>/poses.sdf
    complexes/<ligand_id>/quality.json
    complexes/<ligand_id>/atom_mapping.json
```

When neural refinement is enabled, the final refined bundle is also written to:

```text
<out>/<run-name>/analogue_fep_refined/
```

## FEP-ready criteria

A pose is marked `FEP_READY` when it has:

- acceptable MCS mapping;
- MCS ligand coverage above `--analogue-min-mcs-fraction`;
- MCS core RMSD below `--analogue-core-rmsd-cutoff`;
- no detected internal ligand clashes.
- no detected receptor heavy-atom clashes when receptor-aware ranking is enabled.

`NEEDS_REVIEW` poses are exported but clearly marked in `quality_report.csv` and `quality.json`.

Use `--no-analogue-receptor-aware` to disable receptor clash/contact scoring for isolated ligand-only experiments.
