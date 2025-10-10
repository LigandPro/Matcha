# Taken from https://github.com/RMeli/spyrmsd and https://github.com/gcorso/DiffDock/


from typing import Any, List, Optional, Tuple, Union
import numpy as np

from spyrmsd import graph, molecule, qcp, utils, molecule

import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def compute_all_isomorphisms(rdkit_mol):
    try:
        with time_limit(2):
            mol = molecule.Molecule.from_rdkit(rdkit_mol)
            # Convert molecules to graphs
            G1 = graph.graph_from_adjacency_matrix(
                mol.adjacency_matrix, mol.atomicnums)

            # Get all the possible graph isomorphisms
            isomorphisms = graph.match_graphs(G1, G1)
    except TimeoutException:
        isomorphisms = [(list(range(rdkit_mol.GetNumAtoms())),
                         list(range(rdkit_mol.GetNumAtoms())))]
    return isomorphisms


def get_symmetry_rmsd_with_isomorphisms(coords1, coords2, isomorphisms):
    with time_limit(1):
        assert coords1.shape == coords2.shape

        n = coords1.shape[0]

        # Minimum result
        # Squared displacement (not minimize) or RMSD (minimize)
        min_result = np.inf

        # Loop over all graph isomorphisms to find the lowest RMSD
        for idx1, idx2 in isomorphisms:
            # Use the isomorphism to shuffle coordinates around (from original order)
            c1i = coords1[idx1, :]
            c2i = coords2[idx2, :]

            # Compute square displacement
            # Avoid dividing by n and an expensive sqrt() operation
            result = np.sum((c1i - c2i) ** 2)

            if result < min_result:
                min_result = result

        # Compute actual RMSD from square displacement
        min_result = np.sqrt(min_result / n)

        # Return the actual RMSD
        return min_result


def get_symmetry_rmsd(mol, coords1, coords2, mol2=None, return_permutation=False):
    with time_limit(10):
        mol = molecule.Molecule.from_rdkit(mol)
        mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
        mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
        mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
        RMSD = symmrmsd(
            coords1,
            coords2,
            mol.atomicnums,
            mol2_atomicnums,
            mol.adjacency_matrix,
            mol2_adjacency_matrix,
            return_permutation=return_permutation
        )
        return RMSD


def _rmsd_isomorphic_core(
    coords1: np.ndarray,
    coords2: np.ndarray,
    aprops1: np.ndarray,
    aprops2: np.ndarray,
    am1: np.ndarray,
    am2: np.ndarray,
    center: bool = False,
    minimize: bool = False,
    isomorphisms: Optional[List[Tuple[List[int], List[int]]]] = None,
    atol: float = 1e-9,
) -> Tuple[float, List[Tuple[List[int], List[int]]], Tuple[List[int], List[int]]]:
    """
    Compute RMSD using graph isomorphism.

    Parameters
    ----------
    coords1: np.ndarray
        Coordinate of molecule 1
    coords2: np.ndarray
        Coordinates of molecule 2
    aprops1: np.ndarray
        Atomic properties for molecule 1
    aprops2: np.ndarray
        Atomic properties for molecule 2
    am1: np.ndarray
        Adjacency matrix for molecule 1
    am2: np.ndarray
        Adjacency matrix for molecule 2
    center: bool
        Centering flag
    minimize: bool
        Compute minized RMSD
    isomorphisms: Optional[List[Dict[int,int]]]
        Previously computed graph isomorphism
    atol: float
        Absolute tolerance parameter for QCP (see :func:`qcp_rmsd`)

    Returns
    -------
    Tuple[float, List[Dict[int, int]]]
        RMSD (after graph matching) and graph isomorphisms
    """

    assert coords1.shape == coords2.shape

    n = coords1.shape[0]

    # Center coordinates if required
    c1 = utils.center(coords1) if center or minimize else coords1
    c2 = utils.center(coords2) if center or minimize else coords2

    # No cached isomorphisms
    if isomorphisms is None:
        # Convert molecules to graphs
        G1 = graph.graph_from_adjacency_matrix(am1, aprops1)
        G2 = graph.graph_from_adjacency_matrix(am2, aprops2)

        # Get all the possible graph isomorphisms
        isomorphisms = graph.match_graphs(G1, G2)

    # Minimum result
    # Squared displacement (not minimize) or RMSD (minimize)
    min_result = np.inf
    min_isomorphisms = None

    # Loop over all graph isomorphisms to find the lowest RMSD
    for idx1, idx2 in isomorphisms:
        # Use the isomorphism to shuffle coordinates around (from original order)
        c1i = c1[idx1, :]
        c2i = c2[idx2, :]

        if not minimize:
            # Compute square displacement
            # Avoid dividing by n and an expensive sqrt() operation
            result = np.sum((c1i - c2i) ** 2)
        else:
            # Compute minimized RMSD using QCP
            result = qcp.qcp_rmsd(c1i, c2i, atol)

        if result < min_result:
            min_result = result
            min_isomorphisms = (idx1, idx2)

    if not minimize:
        # Compute actual RMSD from square displacement
        min_result = np.sqrt(min_result / n)

    # Return the actual RMSD
    return min_result, isomorphisms, min_isomorphisms


def symmrmsd(
    coordsref: np.ndarray,
    coords: Union[np.ndarray, List[np.ndarray]],
    apropsref: np.ndarray,
    aprops: np.ndarray,
    amref: np.ndarray,
    am: np.ndarray,
    center: bool = False,
    minimize: bool = False,
    cache: bool = True,
    atol: float = 1e-9,
    return_permutation: bool = False,
) -> Any:
    """
    Compute RMSD using graph isomorphism for multiple coordinates.

    Parameters
    ----------
    coordsref: np.ndarray
        Coordinate of reference molecule
    coords: List[np.ndarray]
        Coordinates of other molecule
    apropsref: np.ndarray
        Atomic properties for reference
    aprops: np.ndarray
        Atomic properties for other molecule
    amref: np.ndarray
        Adjacency matrix for reference molecule
    am: np.ndarray
        Adjacency matrix for other molecule
    center: bool
        Centering flag
    minimize: bool
        Minimum RMSD
    cache: bool
        Cache graph isomorphisms
    atol: float
        Absolute tolerance parameter for QCP (see :func:`qcp_rmsd`)

    Returns
    -------
    float: Union[float, List[float]]
        Symmetry-corrected RMSD(s) and graph isomorphisms

    Notes
    -----

    Graph isomorphism is introduced for symmetry corrections. However, it is also
    useful when two molecules do not have the atoms in the same order since atom
    matching according to atomic numbers and the molecular connectivity is
    performed. If atoms are in the same order and there is no symmetry, use the
    `rmsd` function.
    """

    if isinstance(coords, list):  # Multiple RMSD calculations
        RMSD: Any = []
        isomorphism = None
        min_iso = []

        for c in coords:
            if not cache:
                # Reset isomorphism
                isomorphism = None

            srmsd, isomorphism, min_i = _rmsd_isomorphic_core(
                coordsref,
                c,
                apropsref,
                aprops,
                amref,
                am,
                center=center,
                minimize=minimize,
                isomorphisms=isomorphism,
                atol=atol,
            )
            min_iso.append(min_i)
            RMSD.append(srmsd)

    else:  # Single RMSD calculation
        RMSD, isomorphism, min_iso = _rmsd_isomorphic_core(
            coordsref,
            coords,
            apropsref,
            aprops,
            amref,
            am,
            center=center,
            minimize=minimize,
            isomorphisms=None,
            atol=atol,
        )

    if return_permutation:
        return RMSD, min_iso
    return RMSD
