from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS


CORE_SMARTS = "CSc1ncnc(N)c1CC(=O)N"


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _first_sdf(path: Path):
    return next((m for m in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False) if m), None)


def _prep(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(mol)
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol, catchErrors=True)
    mol = Chem.RemoveHs(mol, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        pass
    return mol


def _coords(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(mol.GetConformer().GetPositions(), dtype=float)


def _mcs_rmsd(pred: Chem.Mol, ref: Chem.Mol) -> tuple[float, int]:
    pred = _prep(pred)
    ref = _prep(ref)
    res = rdFMCS.FindMCS([pred, ref], timeout=10, ringMatchesRingOnly=True, completeRingsOnly=True)
    patt = Chem.MolFromSmarts(res.smartsString) if res.smartsString else None
    if patt is None:
        return float("nan"), 0
    patt.UpdatePropertyCache(strict=False)
    try:
        Chem.GetSymmSSSR(patt)
    except Exception:
        pass
    pmatch = pred.GetSubstructMatch(patt)
    rmatch = ref.GetSubstructMatch(patt)
    if not pmatch or not rmatch or len(pmatch) != len(rmatch):
        return float("nan"), 0
    delta = _coords(pred)[list(pmatch)] - _coords(ref)[list(rmatch)]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))), len(pmatch)


def _rowan_subset(benchmark_dir: Path) -> list[tuple[str, str]]:
    ann = pd.read_csv(benchmark_dir / "structure/processed_outputs/annotated_complexes.csv")
    aff = pd.read_csv(benchmark_dir / "affinity/reference/fragalysis_compound_reference.csv")
    affinity_names = set(aff.fragalysis_code.astype(str))
    core = Chem.MolFromSmarts(CORE_SMARTS)
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in ann.itertuples(index=False):
        smiles = str(row.smiles).split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not mol.HasSubstructMatch(core):
            continue
        if row.smiles in seen:
            continue
        seen.add(row.smiles)
        if row.complex_name == "A71EV2A-x7259a":
            continue
        if row.complex_name not in affinity_names:
            continue
        selected.append((row.complex_name, smiles))
    if len(selected) != 76:
        raise RuntimeError(f"Expected 76 Rowan-like ligands, got {len(selected)}")
    return selected


def _write_subset_sdf(selected: list[tuple[str, str]], path: Path) -> None:
    writer = Chem.SDWriter(str(path))
    for name, smiles in selected:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        status = AllChem.EmbedMolecule(mol, randomSeed=20260513)
        if status != 0:
            AllChem.EmbedMolecule(mol, randomSeed=20260513, useRandomCoords=True)
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=100)
        except Exception:
            pass
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Matcha analogue mode on Rowan/OpenBind EV-A71.")
    parser.add_argument("--openbind-data", required=True, type=Path)
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--torsion-mc-steps", type=int, default=0)
    parser.add_argument("--gnina-path", type=Path, default=None)
    parser.add_argument("--gnina-minimize", action="store_true")
    parser.add_argument("--gnina-score-type", default="Affinity")
    parser.add_argument("--gnina-cnn-scoring", default="none")
    parser.add_argument("--gnina-timeout-seconds", type=int, default=300)
    parser.add_argument("--analogue-embed-timeout-seconds", type=int, default=30)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    selected = _rowan_subset(args.benchmark_dir)

    receptor = args.openbind_data / "structures/x7259a/A71EV2A-x7259a/A71EV2A-x7259a_prepared.pdb"
    template = args.openbind_data / "structures/x7259a/A71EV2A-x7259a/A71EV2A-x7259a_ligand_ref.sdf"
    suffix = ""
    if args.gnina_path is not None:
        suffix = f"_gnina_{_slug(args.gnina_score_type)}"
        if args.gnina_cnn_scoring != "none":
            suffix += f"_{_slug(args.gnina_cnn_scoring)}"
        if args.gnina_minimize:
            suffix += "_min"
    run_name = f"openbind_x7259a_76_n{args.n_samples}_tmc{args.torsion_mc_steps}{suffix}"
    subset_sdf = args.out / f"{run_name}_input.sdf"
    _write_subset_sdf(selected, subset_sdf)
    cmd = [
        sys.executable,
        "-m",
        "matcha.cli",
        "-r",
        str(receptor),
        "--ligand-dir",
        str(subset_sdf),
        "--analogue-template",
        str(template),
        "--analogue-only",
        "--n-samples",
        str(args.n_samples),
        "--analogue-final-poses",
        str(min(8, args.n_samples)),
        "--analogue-torsion-mc-steps",
        str(args.torsion_mc_steps),
        "--analogue-embed-timeout-seconds",
        str(args.analogue_embed_timeout_seconds),
        "--no-analogue-rbfe-pairwise-edges",
        "-o",
        str(args.out),
        "--run-name",
        run_name,
        "--overwrite",
    ]
    if args.gnina_path is not None:
        cmd.extend([
            "--scorer",
            "gnina",
            "--scorer-path",
            str(args.gnina_path),
            "--gnina-score-type",
            args.gnina_score_type,
            "--gnina-cnn-scoring",
            args.gnina_cnn_scoring,
            "--gnina-timeout-seconds",
            str(args.gnina_timeout_seconds),
        ])
        if not args.gnina_minimize:
            cmd.append("--no-scorer-minimize")
    log = args.out / f"{run_name}.log"
    with log.open("w") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)

    bundle = args.out / run_name / "analogue/fep_bundle_seed"
    ref_by_name = {p.parent.name: p for p in (args.openbind_data / "structures").rglob("*_ligand_ref.sdf")}
    rows: list[dict] = []
    for name, _ in selected:
        ref = _first_sdf(ref_by_name[name])
        cdir = bundle / "complexes" / name
        best = _first_sdf(cdir / "best_pose.sdf")
        poses = [m for m in Chem.SDMolSupplier(str(cdir / "poses.sdf"), removeHs=False, sanitize=False) if m]
        best_rmsd, atoms = _mcs_rmsd(best, ref)
        ensemble = [_mcs_rmsd(m, ref)[0] for m in poses]
        finite = [x for x in ensemble if np.isfinite(x)]
        rows.append({
            "complex_name": name,
            "mcs_atoms": atoms,
            "single_best_rmsd": best_rmsd,
            "ensemble_min_rmsd": min(finite) if finite else float("nan"),
            "n_poses": len(poses),
        })

    report = args.out / f"{run_name}_rmsd.csv"
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n": len(rows),
        "bundle": str(bundle),
        "report": str(report),
        "log": str(log),
        "gnina_path": str(args.gnina_path) if args.gnina_path is not None else None,
        "gnina_minimize": bool(args.gnina_minimize),
        "gnina_score_type": args.gnina_score_type,
        "gnina_cnn_scoring": args.gnina_cnn_scoring,
        "gnina_timeout_seconds": args.gnina_timeout_seconds,
        "analogue_embed_timeout_seconds": args.analogue_embed_timeout_seconds,
    }
    for threshold in [0.5, 1.0, 2.0, 3.0]:
        summary[f"single_le_{threshold}A"] = sum(row["single_best_rmsd"] <= threshold for row in rows)
        summary[f"ensemble_le_{threshold}A"] = sum(row["ensemble_min_rmsd"] <= threshold for row in rows)
    summary_path = args.out / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
