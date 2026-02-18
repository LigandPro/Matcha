import copy
import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
from queue import Queue
from pathlib import Path

import numpy as np
from rdkit.Chem.rdchem import BondType as BT
from rdkit import Chem
from rdkit.Chem import AllChem, GetPeriodicTable
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

import prody
from prody import confProDy
from matcha.utils.log import get_logger

confProDy(verbosity='none')
logger = get_logger(__name__)

_CONFORMER_BACKEND_LOGGED: set[str] = set()


_PDB_PARSE_CACHE_LOCK = threading.Lock()
_PDB_PARSE_CACHE: dict[tuple, object] = {}


biopython_parser = PDBParser()
periodic_table = GetPeriodicTable()
allowable_features = {
    'possible_atomic_num_list': [1, 5, 6, 7, 8, 9, 12, 14, 15, 16, 17, 26, 33, 34, 35, 44, 45, 51, 53, 75, 77, 78, 'misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 0)  # number of scalar features

atom_order = {'G': ['N', 'CA', 'C', 'O'],
              'A': ['N', 'CA', 'C', 'O', 'CB'],
              'S': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
              'C': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
              'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
              'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
              'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
              'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
              'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
              'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
              'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
              'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
              'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
              'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
              'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
              'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
              'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
              'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
              'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
              'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2'],
              'X': ['N', 'CA', 'C', 'O']}     # unknown amino acid

aa_short2long = {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS', 'I': 'ILE',
                 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN', 'G': 'GLY', 'H': 'HIS',
                 'L': 'LEU', 'R': 'ARG', 'W': 'TRP', 'A': 'ALA', 'V': 'VAL', 'E': 'GLU',
                 'Y': 'TYR', 'M': 'MET'}

aa_long2short = {aa_long: aa_short for aa_short,
                 aa_long in aa_short2long.items()}
aa_long2short['MSE'] = 'M'


def lig_atom_featurizer(mol):
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append([
            safe_index(
                allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(
                str(atom.GetChiralTag())),
            safe_index(
                allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(
                allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetValence(
                Chem.ValenceType.IMPLICIT)),
            safe_index(
                allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(
                allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(
                atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(
                atom.GetIsAromatic()),
            safe_index(
                allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(
                ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(
                ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(
                ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(
                ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(
                ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(
                ringinfo.IsAtomInRingOfSize(idx, 8)),
        ])
    # +1 because 0 is the padding index, needed for nn.Embedding
    return np.array(atom_features_list) + 1


def _embed_confs_with_timeout(mol, num_conformers, ps, timeout_seconds):
    """Helper function to run EmbedMultipleConfs in a separate thread with timeout."""
    result_queue = Queue()
    exception_queue = Queue()
    
    def _run_embed():
        try:
            ids = AllChem.EmbedMultipleConfs(mol, num_conformers, ps)
            result_queue.put(ids)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=_run_embed, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running, operation timed out
        return None, True  # None result, timed_out=True
    
    if not exception_queue.empty():
        raise exception_queue.get()
    
    if not result_queue.empty():
        return result_queue.get(), False  # Result, timed_out=False
    
    return None, True  # No result, assume timeout


def _optimize_confs_with_timeout(mol, start_idx, end_idx, timeout_seconds):
    """Helper function to optimize conformers with timeout."""
    result_queue = Queue()
    exception_queue = Queue()
    
    def _run_optimize():
        try:
            for i in range(start_idx, end_idx):
                AllChem.MMFFOptimizeMolecule(mol, confId=i)
            result_queue.put(True)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=_run_optimize, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        return True  # Timed out
    if not exception_queue.empty():
        raise exception_queue.get()
    return False  # Success


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning(f"Invalid integer for {name}: {value}, using default {default}")
        return default
    if parsed <= 0:
        logger.warning(f"Expected positive integer for {name}: {value}, using default {default}")
        return default
    return parsed


def _resolve_worker_command() -> list[str]:
    raw_cmd = os.environ.get("MATCHA_CONFORMER_WORKER_CMD", "").strip()
    if raw_cmd:
        cmd_parts = shlex.split(raw_cmd)
        if not cmd_parts:
            raise ValueError("MATCHA_CONFORMER_WORKER_CMD has no executable")
        return cmd_parts

    # Auto-discover a local worker venv created via scripts/setup_nvmolkit_worker.py.
    repo_root = Path(__file__).resolve().parents[2]
    worker_entry = repo_root / ".venv-nvmolkit-worker" / "bin" / "matcha-nvmolkit-worker"
    if worker_entry.exists():
        return [str(worker_entry)]

    worker_script = repo_root / "scripts" / "nvmolkit_worker.py"
    venv_python = repo_root / ".venv-nvmolkit-worker" / "bin" / "python"
    if worker_script.exists() and venv_python.exists():
        return [str(venv_python), str(worker_script)]

    raise ValueError(
        "MATCHA_CONFORMER_WORKER_CMD is not set and no local worker command was found at "
        f"{worker_entry}. Run: uv run python scripts/setup_nvmolkit_worker.py"
    )


def _split_single_conformer_mols(mol, num_conformers: int):
    if mol.GetNumConformers() == 0:
        fallback = copy.deepcopy(mol)
        if fallback.GetNumConformers() == 0:
            return [fallback]
        first = Chem.Mol(fallback)
        conf0 = fallback.GetConformer(0)
        first.RemoveAllConformers()
        first.AddConformer(conf0, assignId=True)
        first.SetProp("ID", "conformer_0")
        return [first]

    conformer_mols = []
    for cid in range(mol.GetNumConformers()):
        m = Chem.Mol(mol)
        conf = mol.GetConformer(cid)
        m.RemoveAllConformers()
        m.AddConformer(conf, assignId=True)
        m.SetProp("ID", f"conformer_{cid}")
        conformer_mols.append(m)
        if len(conformer_mols) >= num_conformers:
            break
    return conformer_mols


def _remove_hs_safe(mol):
    try:
        return Chem.RemoveAllHs(mol)
    except Exception:
        return Chem.RemoveAllHs(mol, sanitize=False)


def _conformer_has_finite_coords(mol) -> bool:
    if mol is None or mol.GetNumConformers() == 0:
        return False
    try:
        pos = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
    except Exception:
        return False
    return pos.ndim == 2 and pos.shape[1] == 3 and np.isfinite(pos).all()


def _generate_worker_batch(
    molecules,
    confs_per_mol: int,
    seed: int | None,
    optimize: bool,
    chunk_size: int,
):
    worker_timeout_sec = _get_env_int("MATCHA_CONFORMER_WORKER_TIMEOUT_SEC", 300)
    worker_cmd = _resolve_worker_command()

    with tempfile.TemporaryDirectory(prefix="matcha_conformer_worker_") as tmp_dir:
        work_dir = os.path.abspath(tmp_dir)
        input_sdf = os.path.join(work_dir, "input.sdf")
        params_json = os.path.join(work_dir, "params.json")
        output_sdf = os.path.join(work_dir, "output.sdf")
        meta_json = os.path.join(work_dir, "meta.json")

        writer = Chem.SDWriter(input_sdf)
        writer.SetKekulize(False)
        uids = []
        for idx, mol in enumerate(molecules):
            uid = f"mol_{idx}"
            current = Chem.Mol(mol)
            current.SetProp("_Name", uid)
            writer.write(current)
            uids.append(uid)
        writer.close()

        params = {
            "confs_per_mol": int(confs_per_mol),
            "seed": int(seed) if seed is not None else None,
            "optimize": bool(optimize),
            "chunk_size": int(chunk_size),
        }
        with open(params_json, "w", encoding="utf-8") as f:
            json.dump(params, f)

        cmd = [
            *worker_cmd,
            "--input-sdf",
            input_sdf,
            "--params-json",
            params_json,
            "--output-sdf",
            output_sdf,
            "--meta-json",
            meta_json,
        ]

        started_at = time.time()
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=worker_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Worker timeout after {worker_timeout_sec}s for command: {' '.join(cmd)}"
            ) from exc
        elapsed = time.time() - started_at

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            stdout = proc.stdout.strip()
            worker_message = stderr or stdout or "worker exited with non-zero status"
            raise RuntimeError(
                f"Worker process failed (code={proc.returncode}, elapsed={elapsed:.2f}s): "
                f"{worker_message}"
            )

        if not os.path.exists(output_sdf):
            raise RuntimeError("Worker finished without output.sdf")

        grouped = {uid: [] for uid in uids}
        supplier = Chem.SDMolSupplier(output_sdf, sanitize=False, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue
            uid = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            if uid in grouped:
                grouped[uid].append(mol)

        missing = [uid for uid, confs in grouped.items() if not confs]
        if missing:
            raise RuntimeError(f"Worker did not return conformers for {len(missing)} molecules")

        return [grouped[uid][: int(confs_per_mol)] for uid in uids]


def generate_multiple_conformers(orig_mol, num_conformers):
    mol = copy.deepcopy(orig_mol)
    ps = AllChem.ETKDGv3()
    failures, ids = 0, []
    max_failures = 3
    max_iterations = max_failures  # Prevent infinite loops

    iteration = 0
    while mol.GetNumConformers() < num_conformers and iteration < max_iterations:
        current_count = mol.GetNumConformers()
        needed = num_conformers - current_count

        # Generate conformers on a temporary molecule to avoid replacing existing ones
        temp_mol = copy.deepcopy(orig_mol)
        temp_mol.RemoveAllConformers()
        try:
            ids, timed_out = _embed_confs_with_timeout(temp_mol, needed, ps, 60)
            if timed_out or ids is None:
                logger.warning("Conformer embedding timed out, using initial molecule")
                return orig_mol
        except Exception:
            logger.warning("Unable to generate conformers, using initial molecule")
            return orig_mol

        ids = [id for id in ids if id != -1]

        # Manually add each new conformer to the main molecule
        added_count = 0
        for conf_id in ids:
                conf = temp_mol.GetConformer(conf_id)
                mol.AddConformer(conf, assignId=True)
                added_count += 1
        
        new_count = mol.GetNumConformers()

        if added_count == 0:
            # No new conformers were added
            logger.debug(f"No new conformers added. Retrying {iteration + 1}/{max_iterations}")
            failures += 1
            if failures >= max_failures:
                break
        else:
            # Successfully added some conformers, reset failure counter
            failures = 0
            logger.debug(f"Added {new_count - current_count} conformers (total: {new_count}/{num_conformers})")
        
        iteration += 1

    if mol.GetNumConformers() == 0:
        logger.debug("RDKit coords generation failed without random coords, using random coords")
        ps.useRandomCoords = True
        ids, timed_out = _embed_confs_with_timeout(mol, min(num_conformers, 10), ps, 60)
            
        if not timed_out and ids is not None:
            ids = [id for id in ids if id != -1]
            for conf_id in ids:
                conf = mol.GetConformer(conf_id)
                # Optimize with timeout
                timed_out = _optimize_confs_with_timeout(mol, conf_id, conf_id + 1, 60)
                if timed_out:
                    logger.warning(f"Optimization timed out for conformer {conf_id}")
        else:
            logger.debug("using random coords now with 1 conformer")
            ids, timed_out = _embed_confs_with_timeout(mol, 1, ps, 60)
                
            if not timed_out and ids is not None:
                ids = [id for id in ids if id != -1]
                for conf_id in ids:
                    conf = mol.GetConformer(conf_id)
                    # Optimize with timeout
                    timed_out = _optimize_confs_with_timeout(mol, conf_id, conf_id + 1, 60)
                    if timed_out:
                        logger.error(f'Optimization timed out for conformer {conf_id}')
    
    if mol.GetNumConformers() == 0:
        logger.warning("No conformers generated, using original molecule")
        return orig_mol
    else:
        logger.debug(f"Generated {mol.GetNumConformers()} conformers")
        return mol


def generate_conformer_mols(orig_mol, num_conformers, backend: str | None = None):
    """Generate a list of single-conformer RDKit molecules."""
    return generate_conformer_mols_batch(
        [orig_mol],
        confs_per_mol=int(num_conformers),
        backend=backend,
    )[0]


def generate_conformer_mols_batch(
    mols,
    confs_per_mol: int,
    backend: str | None = None,
    seed: int | None = None,
    optimize: bool = True,
    chunk_size: int = 128,
):
    """Generate conformers for a batch of molecules.

    Returns:
        A list of lists, aligned with the input `mols`. Each inner list contains
        single-conformer RDKit molecules.
    """
    backend = (backend or os.environ.get("MATCHA_CONFORMER_BACKEND", "auto")).lower()
    if backend not in {"auto", "rdkit", "worker"}:
        raise ValueError(f"Unknown conformer backend: {backend}")

    if seed is None:
        seed_env = os.environ.get("MATCHA_CONFORMER_SEED")
        if seed_env is not None and seed_env.strip():
            try:
                seed = int(seed_env)
            except ValueError:
                seed = None

    if confs_per_mol <= 0:
        return [[copy.deepcopy(m)] for m in mols]

    if chunk_size < 1:
        chunk_size = len(mols) or 1
    worker_chunk_size = _get_env_int("MATCHA_CONFORMER_WORKER_CHUNK_SIZE", int(chunk_size))

    results = [None] * len(mols)

    # Prepare molecules: keep originals intact for safety.
    prepared = []
    for mol in mols:
        init = copy.deepcopy(mol)
        init.RemoveAllConformers()
        init = Chem.AddHs(init)
        prepared.append(init)

    worker_cmd_configured = bool(os.environ.get("MATCHA_CONFORMER_WORKER_CMD", "").strip())
    if not worker_cmd_configured:
        repo_root = Path(__file__).resolve().parents[2]
        worker_entry = repo_root / ".venv-nvmolkit-worker" / "bin" / "matcha-nvmolkit-worker"
        worker_script = repo_root / "scripts" / "nvmolkit_worker.py"
        venv_python = repo_root / ".venv-nvmolkit-worker" / "bin" / "python"
        worker_cmd_configured = worker_entry.exists() or (worker_script.exists() and venv_python.exists())

    use_worker = backend == "worker" or (backend == "auto" and worker_cmd_configured)
    worker_disabled = False

    for start in range(0, len(prepared), chunk_size):
        end = min(len(prepared), start + chunk_size)
        chunk = prepared[start:end]
        chunk_results = None

        try:
            if use_worker and not worker_disabled:
                chunk_results = _generate_worker_batch(
                    chunk,
                    confs_per_mol=int(confs_per_mol),
                    seed=seed,
                    optimize=optimize,
                    chunk_size=worker_chunk_size,
                )
                if "worker" not in _CONFORMER_BACKEND_LOGGED:
                    logger.info("Conformer backend: worker")
                    _CONFORMER_BACKEND_LOGGED.add("worker")
            else:
                chunk_results = []
                for mol in chunk:
                    mol_with_confs = generate_multiple_conformers(mol, int(confs_per_mol))
                    mol_with_confs = _remove_hs_safe(mol_with_confs)
                    chunk_results.append(
                        _split_single_conformer_mols(mol_with_confs, int(confs_per_mol))
                    )
                if "rdkit" not in _CONFORMER_BACKEND_LOGGED:
                    logger.info("Conformer backend: RDKit")
                    _CONFORMER_BACKEND_LOGGED.add("rdkit")
        except Exception as exc:
            if backend == "worker":
                raise RuntimeError(f"Worker conformer generation failed: {exc}") from exc
            worker_disabled = True
            logger.warning(f"Worker conformer backend failed, falling back to RDKit: {exc}")
            chunk_results = []
            for mol in chunk:
                mol_with_confs = generate_multiple_conformers(mol, int(confs_per_mol))
                mol_with_confs = _remove_hs_safe(mol_with_confs)
                chunk_results.append(_split_single_conformer_mols(mol_with_confs, int(confs_per_mol)))
            if "rdkit" not in _CONFORMER_BACKEND_LOGGED:
                logger.info("Conformer backend: RDKit")
                _CONFORMER_BACKEND_LOGGED.add("rdkit")

        for local_idx, conformer_mols in enumerate(chunk_results):
            processed = []
            for conf_idx, conf_mol in enumerate(conformer_mols):
                conf_mol = _remove_hs_safe(conf_mol)
                if conf_mol.GetNumConformers() == 0:
                    continue
                if not _conformer_has_finite_coords(conf_mol):
                    continue
                if not conf_mol.HasProp("ID"):
                    conf_mol.SetProp("ID", f"conformer_{conf_idx}")
                processed.append(conf_mol)
                if len(processed) >= int(confs_per_mol):
                    break
            if not processed:
                fallback = copy.deepcopy(mols[start + local_idx])
                if fallback.GetNumConformers() > 0 and not _conformer_has_finite_coords(fallback):
                    fallback.RemoveAllConformers()
                processed = _split_single_conformer_mols(fallback, 1)
            results[start + local_idx] = processed

    # Satisfy type checker: results is fully populated.
    return [r if r is not None else [copy.deepcopy(mols[i])] for i, r in enumerate(results)]


def save_multiple_confs(mol, output_conf_path, num_conformers):
    init_mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    mol = Chem.AddHs(mol)

    mol = generate_multiple_conformers(mol, num_conformers)
    mol = Chem.RemoveAllHs(mol)

    if mol.GetNumConformers() == 0:
        mol = Chem.RemoveAllHs(init_mol)

    writer = Chem.SDWriter(output_conf_path)
    writer.SetKekulize(False)
    for cid in range(mol.GetNumConformers()):
        mol.SetProp('ID', f'conformer_{cid}')
        writer.write(mol, confId=cid)
    writer.close()
    return mol


def safe_index(items, element):
    """Return index of element in items or the last index if absent."""
    try:
        return items.index(element)
    except ValueError:
        return len(items) - 1


def parse_receptor(pdbid, pdbbind_dir, dataset_type):
    rec = parsePDB(pdbid, pdbbind_dir, dataset_type)
    return rec


def parsePDB(pdbid, pdbbind_dir, dataset_type):
    if dataset_type in {'pdbbind', 'pdbbind_conf', 'dockgen', 'dockgen_full', 'dockgen_full_conf'}:
        rec_path = os.path.join(
            pdbbind_dir, pdbid, f'{pdbid}_protein_processed.pdb')
    elif dataset_type.startswith(('posebusters', 'astex', 'any')):
        rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein.pdb')
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')
    protein = parse_pdb_from_path(rec_path)
    return protein


def parse_pdb_from_path(path):
    """Parse a PDB file via ProDy with a small in-process cache.

    Batch docking often repeats the same receptor for hundreds of ligands. In CLI
    batch mode we symlink each per-ligand receptor path to a single underlying
    file, so caching by realpath + stat dramatically reduces repeated parsing.
    """
    real_path = os.path.realpath(path)
    st = os.stat(real_path)
    key = (
        real_path,
        getattr(st, "st_dev", None),
        getattr(st, "st_ino", None),
        st.st_size,
        getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)),
    )

    with _PDB_PARSE_CACHE_LOCK:
        cached = _PDB_PARSE_CACHE.get(key)
    if cached is not None:
        try:
            return cached.copy()
        except Exception:
            return cached

    pdb = prody.parsePDB(real_path)
    with _PDB_PARSE_CACHE_LOCK:
        _PDB_PARSE_CACHE[key] = pdb
    return pdb


def get_coords(prody_pdb):
    resindices = sorted(set(prody_pdb.ca.getResindices()))
    coords = np.full((len(resindices), 14, 3), np.nan)
    atom_names = np.full((len(resindices), 14), np.nan).astype(object)
    seq = []
    for i, resind in enumerate(resindices):
        sel = prody_pdb.select(f'resindex {resind}')
        resname = sel.getResnames()[0]
        seq.append(sel.ca.getSequence()[0])
        for j, name in enumerate(atom_order[aa_long2short[resname] if resname in aa_long2short else 'X']):
            sel_resnum_name = sel.select(f'name {name}')
            if sel_resnum_name is not None:
                coords[i, j, :] = sel_resnum_name.getCoords()[0]
                atom_names[i, j] = sel_resnum_name.getElements()[0]
            else:
                coords[i, j, :] = [np.nan, np.nan, np.nan]
                atom_names[i, j] = 'X'
    seq = np.array([s for s in seq])
    return coords, atom_names, seq, np.array(resindices)


def read_pdbbind_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".mol2") and 'rdkit' not in file:
            lig = read_molecule(os.path.join(
                pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True, strict=True)
            # read sdf file if mol2 file cannot be sanitized
            if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-5] + ".sdf")):
                logger.debug("Using the .mol2 file failed. We found a .sdf file instead and are trying to use that")
                lig = read_molecule(os.path.join(
                    pdbbind_dir, name, file[:-5] + ".sdf"), remove_hs=remove_hs, sanitize=True)
                if lig is None:
                    lig = read_molecule(os.path.join(pdbbind_dir, name, 
                                        file), remove_hs=remove_hs, sanitize=True)
            if lig is not None:
                ligs.append(lig)
    return ligs


def read_molecule(molecule_file, sanitize=False, remove_hs=False, strict=False):
    """
    Read a molecular structure from a file and optionally process it.

    This function reads a molecular structure from various file formats and provides options to sanitize the molecule,
    calculate Gasteiger charges, and remove hydrogen atoms.

    Parameters:
    molecule_file (str): Path to the molecular structure file. Supported formats are .mol2, .sdf, .pdbqt, and .pdb.
    sanitize (bool): If True, sanitize the molecule (default: False).
    remove_hs (bool): If True, remove hydrogen atoms from the molecule (default: False).

    Returns:
    RDKit.Chem.Mol or None: The RDKit molecule object if the molecule is successfully read and processed, None otherwise.

    Raises:
    ValueError: If the file format is not supported.

    Notes:
    - Sanitization ensures the molecule's valence states are correct and that the structure is reasonable.
    - Gasteiger charges are partial charges used for computational chemistry methods.
    - Removing hydrogen atoms can be useful for simplifying the molecule, though it may lose information.

    Example:
    >>> from rdkit import Chem
    >>> mol = read_molecule('molecule.mol2', sanitize=True, remove_hs=True)
    >>> if mol:
    >>>     print(Chem.MolToSmiles(mol))
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(
            molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(
            molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(
            molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                logger.warning("RDKit was unable to sanitize the molecule")
                if strict:
                    return None

        if remove_hs:
            try:
                mol = Chem.RemoveAllHs(mol, sanitize=sanitize)
            except Exception:
                logger.warning("RDKit was unable to remove hydrogen atoms from the molecule")
                if strict:
                    return None
                mol = Chem.RemoveAllHs(mol, sanitize=False)

    except Exception as e:
        logger.error(f"RDKit was unable to read the molecule: {e}")
        return None

    return mol


def read_sdf_with_multiple_confs(molecule_file, sanitize=False, remove_hs=False):
    supplier = Chem.SDMolSupplier(
        molecule_file, sanitize=False, removeHs=False)
    mols = []
    for mol in supplier:
        try:
            if sanitize:
                Chem.SanitizeMol(mol)

            if remove_hs:
                mol = Chem.RemoveAllHs(mol, sanitize=sanitize)
        except Exception as e:
            logger.error(f"RDKit was unable to read the molecule: {e}")
            mol = None

        if mol is not None:
            mols.append(mol)
    return mols


def extract_receptor_structure_prody(rec, lig, sequences_to_embeddings):
    """
    Extract and process the structure of a receptor in the context of its interaction with a ligand.

    This function extracts the atomic coordinates of amino acids in the receptor, particularly focusing on
    backbone atoms (C-alpha, N, and C). It filters out non-amino acid residues and identifies the chains
    that are valid (contain amino acids) and those that are in close proximity to the ligand.

    Parameters:
    rec (Bio.PDB.Structure.Structure): The receptor structure, typically a Bio.PDB structure object.
    lig (rdkit.Chem.Mol): The ligand molecule, typically an RDKit molecule object.
    lm_embedding_chains (list of np.ndarray, optional): Optional embeddings for each chain from a language model.
        If provided, it should have the same number of chains as the receptor structure.

    Returns:
    tuple:
        - rec (Bio.PDB.Structure.Structure): The modified receptor structure with invalid chains removed.
        - c_alpha_coords (np.ndarray): A numpy array of shape (n_residues, 3) containing the C-alpha atom coordinates of
          valid residues.
        - lm_embeddings (np.ndarray or None): A concatenated numpy array of the valid language model embeddings for the chains,
          if lm_embedding_chains is provided. Otherwise, None.
    """
    if lig is not None:
        conf = lig.GetConformer()
        lig_coords = conf.GetPositions()
    seq = rec.ca.getSequence()
    coords, atom_names, seq_new, resindices = get_coords(rec)

    res_chain_ids = rec.ca.getChids()
    res_seg_ids = rec.ca.getSegnames()
    res_chain_ids = np.asarray(
        [s + c for s, c in zip(res_seg_ids, res_chain_ids)])
    chain_ids = np.unique(res_chain_ids)
    seq = np.array([s for s in seq])

    sequences = []
    lm_embeddings = []
    c_alpha_coords = []
    full_coords = []
    full_atom_names = []
    full_atom_residue_ids = []
    min_distances_to_lig = []
    chain_distances_list = []
    chain_distances = {}
    start_res_index = 0
    for i, chain_id in enumerate(chain_ids):
        chain_mask = res_chain_ids == chain_id
        chain_seq = ''.join(seq[chain_mask])
        chain_coords = coords[chain_mask]

        chain_atom_residue_ids = np.arange(start_res_index, start_res_index + chain_coords.shape[0]).repeat(14)
        start_res_index += chain_coords.shape[0]
        chain_atom_names = atom_names[chain_mask]

        nonempty_coords = chain_coords.reshape(-1, 3)
        notnan_mask = np.isnan(nonempty_coords).sum(axis=1) == 0
        nonempty_coords = nonempty_coords[notnan_mask]

        chain_atom_residue_ids = chain_atom_residue_ids[notnan_mask]

        chain_atom_names = chain_atom_names.reshape(-1)
        chain_atom_names = chain_atom_names[notnan_mask]

        min_dist_to_lig = 0
        if lig is not None:
            # Fast nearest-neighbour distances: for each ligand atom, compute distance to
            # the closest protein atom in this chain (equivalent to min over rows).
            tree = cKDTree(nonempty_coords)
            try:
                min_dist_arr, _ = tree.query(lig_coords, k=1, workers=-1)
            except TypeError:
                # Older SciPy without `workers` argument.
                min_dist_arr, _ = tree.query(lig_coords, k=1)
            min_dist_arr = np.asarray(min_dist_arr)
            min_dist_to_lig = float(min_dist_arr.min())
            chain_distances[chain_id] = min_dist_to_lig

        if min_dist_to_lig < 4.5:
            logger.debug(f'keep chain {chain_id} with distance {min_dist_to_lig}')
            # if min_dist_to_lig < 10:
            embeddings, tokenized_seq = sequences_to_embeddings[chain_seq]
            chain_distances_list.append(min_dist_to_lig)
            sequences.append(tokenized_seq)
            lm_embeddings.append(embeddings)
            c_alpha_coords.append(chain_coords[:, 1].astype(np.float32))
            full_coords.append(nonempty_coords)
            full_atom_names.append(chain_atom_names)
            full_atom_residue_ids.append(chain_atom_residue_ids)
            if lig is not None:
                min_distances_to_lig.append(min_dist_arr)
        else:
            logger.debug(f'drop irrelevant chain {chain_id} with distance {min_dist_to_lig}')

    if len(c_alpha_coords) == 0:
        logger.error(f"NO VALID CHAIN found, chain_distances: {chain_distances}")
        return None, None, None, None, None, None, None

    chain_lengths = [(len(seq), dist) for seq, dist in zip(sequences, chain_distances_list)]
    c_alpha_coords = np.concatenate(c_alpha_coords, axis=0)  # [n_residues, 3]
    full_coords = np.concatenate(full_coords, axis=0)  # [n_protein_atoms, 3]
    full_atom_names = np.concatenate(full_atom_names, axis=0)
    full_atom_residue_ids = np.concatenate(full_atom_residue_ids, axis=0)
    lm_embeddings = np.concatenate(lm_embeddings, axis=0)
    sequences = np.concatenate(sequences, axis=0)

    if lig is not None:
        min_distances_to_lig = np.stack(min_distances_to_lig)
        min_distances_to_lig = min_distances_to_lig.min(axis=0)

        distance_cutoff = 5.
        is_buried_threshold = 0.3
        buried_atoms_mask = min_distances_to_lig <= distance_cutoff
        fraction_buried = buried_atoms_mask.mean()

        if fraction_buried < is_buried_threshold:
            logger.warning(
                f"Ligand is not buried (fraction_buried = {fraction_buried})")
            return None, None, None, None, None, None, None

    return c_alpha_coords, lm_embeddings, sequences, chain_lengths, full_coords, full_atom_names, full_atom_residue_ids
