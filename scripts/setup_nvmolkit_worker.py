#!/usr/bin/env python3
"""Set up an isolated nvMolKit-based conformer worker environment.

This script creates `.venv-nvmolkit-worker/` in the repository root and installs
nvMolKit + build prerequisites into that virtual environment.
It also installs the local `matcha-nvmolkit-worker` package into that env.

Notes:
  - This is intended to run on Linux with NVIDIA drivers (e.g. Kolmogorov).
  - The main Matcha environment stays unchanged; Matcha will auto-discover the
    worker at `.venv-nvmolkit-worker/bin/python scripts/nvmolkit_worker.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import gzip
import urllib.request
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"+ {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _uv_pip_install(
    python: Path,
    packages: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    upgrade: bool = False,
) -> None:
    cmd = ["uv", "pip", "install", "--python", str(python)]
    if upgrade:
        cmd.append("-U")
    cmd.extend(packages)
    _run(cmd, cwd=cwd, env=env)


def _site_packages(python: Path, repo_root: Path) -> Path:
    code = "import site; print(site.getsitepackages()[0])"
    out = subprocess.check_output([str(python), "-c", code], cwd=str(repo_root), text=True).strip()
    return Path(out)


def _cmake_bin_dir(python: Path, repo_root: Path) -> Path | None:
    """Return directory that contains the real `cmake` binary from the pip package."""
    code = (
        "from pathlib import Path; import cmake; "
        "print(Path(cmake.__file__).resolve().parent / 'data' / 'bin')"
    )
    try:
        out = subprocess.check_output([str(python), "-c", code], cwd=str(repo_root), text=True).strip()
    except Exception:
        return None
    p = Path(out)
    return p if (p / "cmake").exists() else None


def _make_rdkit_lib_symlink_farm(python: Path, repo_root: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        if p.is_symlink() or p.is_file():
            p.unlink()

    code = r"""
import pathlib
import rdkit

pkg_dir = pathlib.Path(rdkit.__file__).resolve().parent
so_files = sorted(pkg_dir.rglob("*.so"))
print("\n".join(str(p) for p in so_files))
"""
    so_list = subprocess.check_output([str(python), "-c", code], cwd=str(repo_root), text=True)
    so_paths = [Path(line.strip()) for line in so_list.splitlines() if line.strip()]
    for src in so_paths:
        dst = out_dir / src.name
        if dst.exists():
            continue
        dst.symlink_to(src)

    non_files = [p for p in out_dir.iterdir() if not p.is_file()]
    if non_files:
        raise RuntimeError(f"Unexpected non-files in RDKit lib dir: {non_files[:3]}")
    return out_dir


def _find_include_dir(python: Path, repo_root: Path, module_name: str) -> Path:
    code = (
        "import importlib, pathlib; "
        f"m=importlib.import_module({module_name!r}); "
        "print(pathlib.Path(m.__file__).resolve().parent / 'include')"
    )
    out = subprocess.check_output([str(python), "-c", code], cwd=str(repo_root), text=True).strip()
    inc = Path(out)
    if not inc.exists():
        raise RuntimeError(f"Include dir not found for {module_name}: {inc}")
    # rdkit-headers wheels vendor headers under include/rdkit/; CMake expects GraphMol/ directly.
    if (inc / "rdkit" / "GraphMol").exists() and not (inc / "GraphMol").exists():
        return inc / "rdkit"
    return inc


def _find_cuda_toolkit_root(python: Path, repo_root: Path) -> Path:
    """Best-effort CUDA toolkit root discovery.

    Tries:
      1) `nvcc` in PATH (system/toolkit install)
      2) `nvidia.cuda_nvcc` package directory (pip-based CUDA components)
    """
    code = r"""
import shutil
from pathlib import Path

nvcc = shutil.which("nvcc")
if nvcc:
    p = Path(nvcc).resolve()
    print(str(p.parent.parent))
    raise SystemExit(0)

try:
    import nvidia.cuda_nvcc as m  # type: ignore
except Exception as e:
    raise SystemExit("CUDA toolkit not found (no nvcc in PATH, no nvidia.cuda_nvcc module)") from e

module_dir = Path(m.__file__).resolve().parent
print(str(module_dir))
"""
    out = subprocess.check_output([str(python), "-c", code], cwd=str(repo_root), text=True).strip()
    root = Path(out)
    if not root.exists():
        raise RuntimeError(f"CUDA toolkit root does not exist: {root}")
    return root


def _os_release_id() -> tuple[str, str]:
    path = Path("/etc/os-release")
    if not path.exists():
        return ("", "")
    kv = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip().strip('"')
    return (kv.get("ID", ""), kv.get("VERSION_ID", ""))


def _parse_cuda_repo_packages(repo_url: str) -> dict[str, str]:
    """Return mapping package_name -> deb relative filename from NVIDIA CUDA repo."""
    pkgs_gz = f"{repo_url.rstrip('/')}/Packages.gz"
    with urllib.request.urlopen(pkgs_gz) as r:
        raw = r.read()
    text = gzip.decompress(raw).decode("utf-8", errors="replace")
    package = None
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("Package: "):
            package = line.split(":", 1)[1].strip()
        elif line.startswith("Filename: ") and package is not None:
            filename = line.split(":", 1)[1].strip().removeprefix("./")
            mapping[package] = filename
        elif not line.strip():
            package = None
    return mapping


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"+ download {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url) as r:
        data = r.read()
    dest.write_bytes(data)


def _extract_deb(deb_path: Path, extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    if shutil.which("dpkg-deb") is None:
        raise RuntimeError(
            "dpkg-deb is required to extract CUDA .deb packages but was not found in PATH. "
            "Install it via your system package manager (e.g. apt install dpkg) or ensure "
            "a full CUDA toolkit is available so the .deb fallback is not needed."
        )
    _run(["dpkg-deb", "-x", str(deb_path), str(extract_root)], cwd=extract_root)


def _cuda_has_required_libs(cuda_root: Path) -> bool:
    """Return True if the toolkit root looks complete enough for nvMolKit builds.

    We primarily need cuSolver (CMake target CUDA::cusolver) and its common deps.
    Pip-provided nvcc packages may not include these shared libraries.
    """
    required_globs = [
        "**/libcusolver.so*",
        "**/libcublas.so*",
        "**/libcusparse.so*",
        "**/libcudart.so*",
    ]
    for pat in required_globs:
        if not any(cuda_root.glob(pat)):
            return False
    return True


def _symlink_cuda_libs_into_toolkit(cuda_root: Path, extract_root: Path, *, cuda_ver: str) -> None:
    """Make a user-space extracted toolkit discoverable by FindCUDAToolkit.

    Some CUDA Debian packages place shared libraries under `usr/lib/x86_64-linux-gnu`
    rather than directly under the toolkit root. CMake's FindCUDAToolkit expects
    libraries under the toolkit root, so we symlink the common CUDA libs into
    `targets/x86_64-linux/lib` when needed.
    """
    src_dirs = [
        extract_root / "usr" / "lib" / "x86_64-linux-gnu",
        extract_root / "usr" / "local" / f"cuda-{cuda_ver}" / "lib64",
    ]
    dst_dir = cuda_root / "targets" / "x86_64-linux" / "lib"
    dst_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        "libcublas.so*",
        "libcusolver.so*",
        "libcusparse.so*",
        "libcurand.so*",
        "libcufft.so*",
        "libnvrtc.so*",
        "libcudart.so*",
    ]
    for src_dir in src_dirs:
        if not src_dir.exists():
            continue
        for pat in patterns:
            for src in src_dir.glob(pat):
                if not src.is_file():
                    continue
                dst = dst_dir / src.name
                if dst.exists():
                    continue
                try:
                    dst.symlink_to(src)
                except FileExistsError:
                    continue


def _rdkit_version_tuple(python: Path, repo_root: Path) -> tuple[int, int, int]:
    code = "import rdkit; print(rdkit.__version__)"
    out = subprocess.check_output([str(python), "-c", code], cwd=str(repo_root), text=True).strip()
    parts = out.split(".")
    if len(parts) < 3:
        raise RuntimeError(f"Unexpected RDKit version string: {out!r}")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as e:
        raise RuntimeError(f"Unexpected RDKit version string: {out!r}") from e


def _patch_rdkit_versions_header(*, rdkit_incdir: Path, python: Path, repo_root: Path) -> None:
    """Patch rdkit-headers placeholder versions.h into a configured header.

    The `rdkit-headers` wheel ships `RDGeneral/versions.h` with CMake placeholders
    like `@RDKit_Year @`, which breaks C++ compilation. We rewrite the file with
    concrete values based on the installed Python RDKit version.
    """
    versions_h = rdkit_incdir / "RDGeneral" / "versions.h"
    if not versions_h.exists():
        raise RuntimeError(f"RDKit versions header not found: {versions_h}")
    text = versions_h.read_text(encoding="utf-8", errors="ignore")
    if "@RDKit_Year" not in text:
        return

    year, month, rev = _rdkit_version_tuple(python, repo_root)
    patched = f"""// Autogenerated by Matcha: scripts/setup_nvmolkit_worker.py
// This file replaces unconfigured CMake placeholders shipped in rdkit-headers wheels.

#pragma once

#include <RDGeneral/export.h>

// Version check macro
// Can be used like: #if (RDKIT_VERSION >= RDKIT_VERSION_CHECK(2018, 3, 1))
#define RDKIT_VERSION_CHECK(year, month, rev) ((year * 1000) + (month * 10) + (rev))

#define RDKIT_VERSION_MAJOR {year}
#define RDKIT_VERSION_MINOR {month}
#define RDKIT_VERSION_PATCH {rev}

// RDKIT_VERSION is (year*1000) + (month*10) + (rev)
#define RDKIT_VERSION RDKIT_VERSION_CHECK(RDKIT_VERSION_MAJOR, RDKIT_VERSION_MINOR, RDKIT_VERSION_PATCH)

namespace RDKit {{
RDKIT_RDGENERAL_EXPORT extern const char* rdkitVersion;
RDKIT_RDGENERAL_EXPORT extern const char* boostVersion;
RDKIT_RDGENERAL_EXPORT extern const char* rdkitBuild;
}}  // namespace RDKit
"""
    versions_h.write_text(patched, encoding="utf-8")
    print(f"[matcha] patched RDKit versions header: {versions_h}", flush=True)


def _prepare_nvmolkit_source(*, venv_dir: Path, repo_root: Path, ref: str) -> Path:
    """Clone nvMolKit source and patch the build to use the C++11 ABI."""
    src_root = venv_dir / ".cache" / "matcha" / "nvmolkit_src"
    if src_root.exists():
        shutil.rmtree(src_root)
    src_root.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--depth",
            "1",
            "--branch",
            ref,
            "https://github.com/NVIDIA-Digital-Bio/nvMolKit",
            str(src_root),
        ],
        cwd=repo_root,
    )

    host_flags = src_root / "cmake" / "host_compiler_flags.cmake"
    if not host_flags.exists():
        raise RuntimeError(f"nvMolKit file not found: {host_flags}")
    text = host_flags.read_text(encoding="utf-8", errors="ignore")
    text = text.replace('message(STATUS "Using pre-cxx11 ABI")', 'message(STATUS "Using C++11 ABI")')
    text = text.replace(
        "add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)",
        "add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)",
    )
    host_flags.write_text(text, encoding="utf-8")
    return src_root


def _ensure_nvcc_from_nvidia_debs(*, venv_dir: Path, repo_root: Path, cuda_ver: str = "12.8") -> Path:
    """Install a user-space CUDA toolkit by downloading NVIDIA .deb packages."""
    os_id, os_ver = _os_release_id()
    if os_id != "ubuntu":
        raise RuntimeError(f"Unsupported OS for auto nvcc download: {os_id} {os_ver}")

    if not os_ver.startswith("24.04"):
        raise RuntimeError(f"Unsupported Ubuntu version for auto nvcc download: {os_ver}")

    repo_url = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64"
    pkg_to_file = _parse_cuda_repo_packages(repo_url)

    suffix = "12-8"
    required = [
        f"cuda-nvcc-{suffix}",
        f"cuda-nvvm-{suffix}",
        f"cuda-crt-{suffix}",
        f"cuda-cccl-{suffix}",
        f"cuda-cudart-{suffix}",
        f"cuda-cudart-dev-{suffix}",
        f"cuda-nvrtc-{suffix}",
        f"cuda-nvrtc-dev-{suffix}",
        f"cuda-nvtx-{suffix}",
        f"cuda-driver-dev-{suffix}",
        f"libcublas-{suffix}",
        f"libcublas-dev-{suffix}",
        f"libcusolver-{suffix}",
        f"libcusolver-dev-{suffix}",
        f"libcusparse-{suffix}",
        f"libcusparse-dev-{suffix}",
        f"libcurand-{suffix}",
        f"libcurand-dev-{suffix}",
        f"libcufft-{suffix}",
        f"libcufft-dev-{suffix}",
        f"cuda-toolkit-{suffix}-config-common",
        "cuda-toolkit-12-config-common",
        "cuda-toolkit-config-common",
    ]

    cache_dir = venv_dir / ".cache" / "matcha" / "cuda_debs"
    extract_root = venv_dir / ".local" / f"cuda-{cuda_ver}"
    for pkg in required:
        rel = pkg_to_file.get(pkg)
        if rel is None:
            raise RuntimeError(f"CUDA repo package not found: {pkg}")
        url = f"{repo_url}/{rel}"
        deb_path = cache_dir / Path(rel).name
        _download(url, deb_path)
        _extract_deb(deb_path, extract_root)

    cuda_home = extract_root / "usr" / "local" / f"cuda-{cuda_ver}"
    _symlink_cuda_libs_into_toolkit(cuda_home, extract_root, cuda_ver=cuda_ver)
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.exists():
        raise RuntimeError(f"nvcc not found after extraction: {nvcc}")
    if not _cuda_has_required_libs(cuda_home):
        raise RuntimeError(f"Extracted CUDA toolkit is missing required shared libraries: {cuda_home}")
    return cuda_home


def _smoke_worker(python: Path, repo_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="matcha_nvmolkit_worker_smoke_") as tmp:
        tmp_dir = Path(tmp)
        input_sdf = tmp_dir / "input.sdf"
        params_json = tmp_dir / "params.json"
        output_sdf = tmp_dir / "output.sdf"
        meta_json = tmp_dir / "meta.json"

        gen_script = f"""
from rdkit import Chem
from rdkit.Chem import AllChem

writer = Chem.SDWriter({str(input_sdf)!r})
writer.SetKekulize(False)
for i, smi in enumerate(["CCO", "c1ccccc1"]):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    mol.SetProp("_Name", f"mol_{{i}}")
    writer.write(mol)
writer.close()
"""
        _run([str(python), "-c", gen_script], cwd=repo_root)
        params_json.write_text(
            json.dumps({"confs_per_mol": 2, "seed": 1, "optimize": True, "chunk_size": 32}, indent=2),
            encoding="utf-8",
        )

        _run(
            [
                str(python), "-m", "matcha_nvmolkit_worker.cli",
                "--input-sdf", str(input_sdf),
                "--params-json", str(params_json),
                "--output-sdf", str(output_sdf),
                "--meta-json", str(meta_json),
            ],
            cwd=repo_root,
        )
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        if not meta.get("ok"):
            raise RuntimeError(f"Worker smoke failed: {meta}")
        written = int(meta.get("written_conformers", 0))
        if written < 2:
            raise RuntimeError(f"Worker smoke produced too few conformers: {meta}")
        print(f"[ok] worker smoke: written_conformers={written}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up isolated nvMolKit conformer worker env")
    parser.add_argument(
        "--venv-dir",
        default=".venv-nvmolkit-worker",
        help="Worker venv directory (relative to repo root). Default: .venv-nvmolkit-worker",
    )
    parser.add_argument(
        "--nvMolKit-ref",
        default="v0.4.0",
        help="Git ref to install nvMolKit from. Default: v0.4.0",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip running the post-install worker smoke test.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    venv_dir = (repo_root / args.venv_dir).resolve()

    if not sys.platform.startswith("linux"):
        print(
            "ERROR: scripts/setup_nvmolkit_worker.py is intended to run on Linux with an NVIDIA GPU.",
            file=sys.stderr,
        )
        return 2

    if shutil.which("uv") is None:
        print("ERROR: uv is required but not found in PATH.", file=sys.stderr)
        return 2

    print(f"[matcha] repo_root: {repo_root}", flush=True)
    print(f"[matcha] worker_venv: {venv_dir}", flush=True)

    _run(["uv", "venv", "--seed", "--clear", str(venv_dir)], cwd=repo_root)
    python = _venv_python(venv_dir)
    if not python.exists():
        print(f"ERROR: venv python not found: {python}", file=sys.stderr)
        return 2

    _uv_pip_install(
        python,
        ["pip", "setuptools", "wheel"],
        cwd=repo_root,
        upgrade=True,
    )
    _uv_pip_install(
        python,
        ["cmake", "ninja", "scikit-build", "pybind11"],
        cwd=repo_root,
        upgrade=True,
    )

    _uv_pip_install(
        python,
        [
            "rdkit==2025.09.3",
            "rdkit-headers==2025.9.3.post1",
            "boost-headers==1.81.0",
            "nvidia-cuda-nvcc-cu12==12.8.*",
        ],
        cwd=repo_root,
    )

    rdkit_libdir = venv_dir / ".cache" / "matcha" / "rdkit_pip_libs"
    _make_rdkit_lib_symlink_farm(python, repo_root, rdkit_libdir)
    rdkit_incdir = _find_include_dir(python, repo_root, "rdkit_headers")
    _patch_rdkit_versions_header(rdkit_incdir=rdkit_incdir, python=python, repo_root=repo_root)
    boost_incdir = _find_include_dir(python, repo_root, "boost_headers")
    cuda_root = _find_cuda_toolkit_root(python, repo_root)
    if not (cuda_root / "bin" / "nvcc").exists() or not _cuda_has_required_libs(cuda_root):
        print(
            "[matcha] CUDA toolkit is incomplete (missing nvcc and/or required CUDA libs); "
            "downloading a user-space toolkit from NVIDIA .debs.",
            flush=True,
        )
        cuda_root = _ensure_nvcc_from_nvidia_debs(venv_dir=venv_dir, repo_root=repo_root, cuda_ver="12.8")

    env = os.environ.copy()
    env["NVMOLKIT_BUILD_AGAINST_PIP_RDKIT"] = "1"
    env["NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR"] = str(rdkit_libdir)
    env["NVMOLKIT_BUILD_AGAINST_PIP_INCDIR"] = str(rdkit_incdir)
    env["NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR"] = str(boost_incdir)
    env["CUDA_HOME"] = str(cuda_root)
    env["CUDAToolkit_ROOT"] = str(cuda_root)
    cmake_bin = _cmake_bin_dir(python, repo_root)
    path_parts = [p for p in [cmake_bin, venv_dir / "bin", cuda_root / "bin"] if p is not None]
    env["PATH"] = ":".join([str(p) for p in path_parts] + [env.get("PATH", "")])
    year, month, rev = _rdkit_version_tuple(python, repo_root)
    cmake_args = env.get("CMAKE_ARGS", "")
    extra = (
        f"-DRDKit_VERSION_MAJOR={year} -DRDKit_VERSION_MINOR={month} -DRDKit_VERSION_PATCH={rev} "
    )
    env["CMAKE_ARGS"] = f"{cmake_args} {extra}".strip()

    worker_pkg = repo_root / "packages" / "matcha_nvmolkit_worker"
    if not worker_pkg.exists():
        raise RuntimeError(f"Worker package not found: {worker_pkg}")
    _uv_pip_install(python, [str(worker_pkg)], cwd=repo_root)

    nvmolkit_src = _prepare_nvmolkit_source(venv_dir=venv_dir, repo_root=repo_root, ref=args.nvMolKit_ref)
    _uv_pip_install(python, [str(nvmolkit_src)], cwd=repo_root, env=env)

    if not args.skip_smoke:
        _smoke_worker(python, repo_root)

    print("[ok] nvMolKit worker env is ready.", flush=True)
    print(
        "[matcha] Matcha will auto-discover the worker at "
        f"{venv_dir}/bin/matcha-nvmolkit-worker when MATCHA_CONFORMER_BACKEND=auto.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
