import argparse
import ctypes
import importlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETKDGv3
from rdkit.Geometry import Point3D


def _prepend_ld_library_path(path: Path):
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    path_str = str(path)
    if path_str not in parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join([path_str, *parts]) if parts else path_str


def _prepare_runtime():
    nvidia_modules = [
        "nvidia.cuda_runtime",
        "nvidia.cublas",
        "nvidia.cudnn",
        "nvidia.cufft",
        "nvidia.curand",
        "nvidia.cusolver",
        "nvidia.cusparse",
        "nvidia.nvjitlink",
        "nvidia.nvtx",
    ]
    discovered_lib_dirs = []
    for module_name in nvidia_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        module_dir = Path(module_file).resolve().parent
        lib_dir = module_dir / "lib"
        if lib_dir.exists():
            _prepend_ld_library_path(lib_dir)
            discovered_lib_dirs.append(lib_dir)

    try:
        venv_dir = Path(sys.executable).resolve().parents[1]
        local_root = venv_dir / ".local"
        for cuda_home in sorted(local_root.glob("cuda-*/usr/local/cuda-*")):
            for lib_dir in (
                cuda_home / "targets" / "x86_64-linux" / "lib",
                cuda_home / "lib64",
            ):
                if lib_dir.exists():
                    _prepend_ld_library_path(lib_dir)
                    discovered_lib_dirs.append(lib_dir)
    except Exception:
        pass

    try:
        import rdkit

        rdkit_pkg_dir = Path(rdkit.__file__).resolve().parent
        rdkit_libs_dir = rdkit_pkg_dir.parent / "rdkit.libs"
        if rdkit_libs_dir.exists():
            _prepend_ld_library_path(rdkit_libs_dir)
            for candidate in sorted(rdkit_libs_dir.glob("libRDKit*.so*")):
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    continue
            for candidate in sorted(rdkit_libs_dir.glob("libboost_python*.so*")):
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    break
                except Exception:
                    continue
        for candidate in sorted(rdkit_pkg_dir.rglob("*.so")):
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            except Exception:
                continue
    except Exception:
        pass

    for lib_dir in discovered_lib_dirs:
        for lib_name in ("libcudart.so.12", "libnvrtc.so.12"):
            candidate = lib_dir / lib_name
            if candidate.exists():
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Generate conformers with nvMolKit in isolated process")
    parser.add_argument("--input-sdf", required=True)
    parser.add_argument("--params-json", required=True)
    parser.add_argument("--output-sdf", required=True)
    parser.add_argument("--meta-json", required=True)
    return parser.parse_args()


def _write_meta(meta_path: Path, payload: dict):
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_input_molecules(input_sdf: Path):
    supplier = Chem.SDMolSupplier(str(input_sdf), sanitize=False, removeHs=False)
    molecules = []
    names = []
    valid_mask = []
    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        uid = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{idx}"
        if not uid.strip():
            uid = f"mol_{idx}"
        current = Chem.Mol(mol)
        current.SetProp("_Name", uid)
        try:
            Chem.SanitizeMol(current)
            current.RemoveAllConformers()
            valid_mask.append(True)
        except Exception:
            valid_mask.append(False)
        molecules.append(current)
        names.append(uid)
    invalid_count = valid_mask.count(False)
    return molecules, names, valid_mask, invalid_count


def _ensure_minimum_conformers(mol, confs_per_mol: int, seed: int | None):
    if mol.GetNumConformers() >= max(1, int(confs_per_mol)):
        return False
    if mol.GetNumConformers() > 0:
        return False

    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom_idx in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(atom_idx, Point3D(0.0, 0.0, 0.0))
    mol.AddConformer(conf, assignId=True)
    return True


def _conf_has_finite_coords(conf) -> bool:
    for atom_idx in range(conf.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        if not (math.isfinite(pos.x) and math.isfinite(pos.y) and math.isfinite(pos.z)):
            return False
    return True


def _drop_non_finite_conformers(mol):
    bad_ids = [
        cid for cid in range(mol.GetNumConformers())
        if not _conf_has_finite_coords(mol.GetConformer(cid))
    ]
    for cid in reversed(bad_ids):
        mol.RemoveConformer(cid)
    return len(bad_ids)


def _write_output_molecules(output_sdf: Path, molecules, names, confs_per_mol: int):
    writer = Chem.SDWriter(str(output_sdf))
    writer.SetKekulize(False)
    written = 0
    for uid, mol in zip(names, molecules):
        count = min(int(confs_per_mol), mol.GetNumConformers())
        for cid in range(count):
            out = Chem.Mol(mol)
            conf = mol.GetConformer(cid)
            out.RemoveAllConformers()
            out.AddConformer(conf, assignId=True)
            out.SetProp("_Name", uid)
            out.SetProp("conf_id", str(cid))
            out.SetProp("ID", f"conformer_{cid}")
            writer.write(out)
            written += 1
    writer.close()
    return written


def main():
    args = parse_args()
    input_sdf = Path(args.input_sdf)
    params_json = Path(args.params_json)
    output_sdf = Path(args.output_sdf)
    meta_json = Path(args.meta_json)

    started_at = time.time()
    try:
        _prepare_runtime()

        with open(params_json, "r", encoding="utf-8") as f:
            params = json.load(f)

        confs_per_mol = int(params.get("confs_per_mol", 1))
        seed = params.get("seed")
        optimize = bool(params.get("optimize", True))
        chunk_size = max(1, int(params.get("chunk_size", 128)))

        molecules, names, valid_mask, invalid_count = _read_input_molecules(input_sdf)
        if not molecules:
            _write_meta(
                meta_json,
                {
                    "ok": True,
                    "elapsed_sec": time.time() - started_at,
                    "input_molecules": 0,
                    "written_conformers": 0,
                    "errors": [],
                },
            )
            output_sdf.parent.mkdir(parents=True, exist_ok=True)
            output_sdf.write_text("", encoding="utf-8")
            return 0

        valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
        if valid_indices:
            valid_molecules = [molecules[i] for i in valid_indices]

            from nvmolkit.embedMolecules import EmbedMolecules as nvMolKitEmbed  # type: ignore
            from nvmolkit.mmffOptimization import (  # type: ignore
                MMFFOptimizeMoleculesConfs as nvMolKitMMFFOptimize,
            )
            from nvmolkit.types import HardwareOptions  # type: ignore

            params_embed = ETKDGv3()
            params_embed.useRandomCoords = True
            if seed is not None:
                params_embed.randomSeed = int(seed)

            embed_hw = HardwareOptions(
                preprocessingThreads=2,
                batchSize=min(chunk_size, len(valid_molecules)),
                batchesPerGpu=2,
            )
            nvMolKitEmbed(
                molecules=valid_molecules,
                params=params_embed,
                confsPerMolecule=confs_per_mol,
                maxIterations=-1,
                hardwareOptions=embed_hw,
            )

            if optimize:
                mmff_hw = HardwareOptions(preprocessingThreads=4, batchSize=0)
                nvMolKitMMFFOptimize(
                    molecules=valid_molecules,
                    maxIters=200,
                    nonBondedThreshold=100.0,
                    hardwareOptions=mmff_hw,
                )

        fallback_generated = 0
        dropped_bad_confs = 0
        for mol in molecules:
            dropped_bad_confs += _drop_non_finite_conformers(mol)
            if _ensure_minimum_conformers(mol, confs_per_mol=confs_per_mol, seed=seed):
                fallback_generated += 1

        output_sdf.parent.mkdir(parents=True, exist_ok=True)
        written_conformers = _write_output_molecules(output_sdf, molecules, names, confs_per_mol)

        _write_meta(
            meta_json,
            {
                "ok": True,
                "elapsed_sec": time.time() - started_at,
                "input_molecules": len(molecules),
                "written_conformers": written_conformers,
                "invalid_input_molecules": invalid_count,
                "fallback_generated_molecules": fallback_generated,
                "dropped_non_finite_conformers": dropped_bad_confs,
                "errors": [],
            },
        )
        return 0
    except Exception as exc:
        _write_meta(
            meta_json,
            {
                "ok": False,
                "elapsed_sec": time.time() - started_at,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(f"nvmolkit worker failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
