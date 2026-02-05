import copy

import numpy as np
from rdkit import Chem

import networkx as nx

from matcha.dataset.complex_dataclasses import Bonds, Ligand
from matcha.utils.transforms import apply_tor_changes_to_pos
from matcha.utils.spyrmsd import get_symmetry_rmsd
from matcha.utils.log import get_logger
logger = get_logger(__name__)


BOND_TYPE_MAPPING = {
    'UNSPECIFIED': 0,
    'SINGLE': 1,
    'DOUBLE': 2,
    'TRIPLE': 3,
    'QUADRUPLE': 4,
    'QUINTUPLE': 5,
    'HEXTUPLE': 6,
    'ONEANDAHALF': 7,
    'TWOANDAHALF': 8,
    'THREEANDAHALF': 9,
    'FOURANDAHALF': 10,
    'FIVEANDAHALF': 11,
    'AROMATIC': 12,
    'IONIC': 13,
    'HYDROGEN': 14,
    'THREECENTER': 15,
    'DATIVEONE': 16,
    'DATIVE': 17,
    'DATIVEL': 18,
    'DATIVER': 19,
    'OTHER': 20,
    'ZERO': 21
}



def mol_to_graph(mol):
    # Initialize graph
    G = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i)
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    return G


def split_molecule(mol, min_lig_size=7):
    G = mol_to_graph(mol)

    molecule_parts = []
    for atom_indices in nx.connected_components(G):

        # take the connected component
        atoms_to_remove = list(set(G.nodes) - set(atom_indices))
        atoms_to_remove.sort(reverse=True)

        em1 = Chem.EditableMol(copy.deepcopy(mol))
        for atom in atoms_to_remove:
            em1.RemoveAtom(atom)
            
        mol_part = em1.GetMol()
        try:
            Chem.SanitizeMol(mol_part)
        except:
            logger.warning('mol_part sanitization failed')
        if mol_part.GetNumAtoms() >= min_lig_size:
            molecule_parts.append(mol_part)
    return molecule_parts


def get_similarly_oriented_bonds(rotatable_bonds, pos):
    bond_rotvecs = (pos[rotatable_bonds[:, 0]] - pos[rotatable_bonds[:, 1]])
    bond_rotvecs = bond_rotvecs / np.linalg.norm(bond_rotvecs, axis=-1)[:, None]
    rotvec_sims = np.abs(bond_rotvecs @ bond_rotvecs.T)
    similar_bonds = []
    for i in range(len(rotvec_sims)):
        similar_bonds.append(tuple(sorted(np.arange(len(rotvec_sims))[rotvec_sims[i] > 0.995])))
    similar_bonds = sorted(set(similar_bonds))

    similar_bond_pairs = []
    for bond_tuple in similar_bonds:
        for i in range(len(bond_tuple) - 1):
            for j in range(i + 1, len(bond_tuple)):
                similar_bond_pairs.append((bond_tuple[i], bond_tuple[j]))
    return similar_bond_pairs


def swap_atoms_in_similar_bonds(rotatable_bonds, similar_bonds):
    bond_pair = rotatable_bonds[list(similar_bonds)]
    element1 = sorted(set(bond_pair[0]) - set(bond_pair[1]))[0]
    element2 = sorted(set(bond_pair[1]) - set(bond_pair[0]))[0]
    
    bond_pair[1, 0] = element1
    bond_pair[1, 1] = element2
    return bond_pair


def get_bond_pairs_on_one_line(rotatable_bonds, pos):
    similar_bond_pairs = get_similarly_oriented_bonds(rotatable_bonds, pos)

    bonds_on_one_line = []
    for similar_bonds in similar_bond_pairs:
        swapped_bond_pair = swap_atoms_in_similar_bonds(rotatable_bonds, similar_bonds)
        are_on_one_line = get_similarly_oriented_bonds(swapped_bond_pair, pos)
        if len(are_on_one_line) > 0:
            bonds_on_one_line.append(similar_bonds)
    return bonds_on_one_line


def get_bonds_on_one_line(rotatable_bonds, pos):
    bonds_pairs_on_one_line = get_bond_pairs_on_one_line(rotatable_bonds, pos)
    bond_sets_on_one_line = [set(bond_pair) for bond_pair in bonds_pairs_on_one_line]
    
    i = 0
    while i < len(bond_sets_on_one_line):
        new_bond_set = []
        bond_set1 = bond_sets_on_one_line[i]
        for j, bond_set2 in enumerate(bond_sets_on_one_line[i + 1:]):
            if len(bond_set1.intersection(bond_set2)) > 0:
                bond_set1 = bond_set1.union(bond_set2)
            else:
                new_bond_set.append(bond_set2)
        bond_sets_on_one_line = bond_sets_on_one_line[:i] + [bond_set1] + new_bond_set
        i += 1
    
    return [sorted(bond_set) for bond_set in bond_sets_on_one_line]


def fix_mask_rotate_using_bonds_on_one_line(rotatable_bonds, mask_rotate, pos):
    if len(rotatable_bonds) == 0:
        return mask_rotate
    bonds_sets_on_one_line = get_bonds_on_one_line(rotatable_bonds, pos)
    for i in range(len(bonds_sets_on_one_line)):
        bonds_on_one_line = bonds_sets_on_one_line[i]
        logger.debug(f'Fix mask_rotate for bonds_on_one_line: {bonds_on_one_line}') 
        bonds_on_one_line = np.array(bonds_on_one_line)[np.argsort(mask_rotate[list(bonds_on_one_line)].sum(axis=1))]
        for i, bond_i in enumerate(bonds_on_one_line):
            for bond_j in bonds_on_one_line[i + 1:]:
                # compute mask of the common region
                to_zero_mask = (mask_rotate[bond_i] == mask_rotate[bond_j]) & mask_rotate[bond_i]
                mask_rotate[bond_j][to_zero_mask] = False

    if not (mask_rotate.sum(1) != 0).all():
        raise ValueError('Error in fix_mask_rotate')
    return mask_rotate


def get_rotatable_bond_periods(rotatable_bonds, mask_rotate, init_pos, mol):
    num_rotatable_bonds = len(rotatable_bonds)
    ligand = Ligand()
    if len(rotatable_bonds) > 0:
        ligand.rotatable_bonds = rotatable_bonds
        ligand.mask_rotate = mask_rotate
        ligand.init_tor = np.zeros(ligand.rotatable_bonds.shape[0], dtype=np.float32)
    
    bond_periods = []
    for i, (bond, cur_mask) in enumerate(zip(rotatable_bonds, mask_rotate)):
    
        bond_period = 2 * np.pi
    
        atoms_to_remove = np.where(~cur_mask)[0]
        em1 = Chem.EditableMol(copy.deepcopy(mol))
        em1.RemoveBond(int(bond[0]), int(bond[1]))
        atoms_to_remove = sorted(atoms_to_remove, reverse=True)
        for atom in atoms_to_remove:
            em1.RemoveAtom(int(atom))
            
        mol_part = em1.GetMol()
        try:
            Chem.SanitizeMol(mol_part)
        except:
            logger.warning('mol_part sanitization failed')

        for angle_set in [[np.pi], [2 * np.pi / 3, -2 * np.pi / 3], [np.pi / 2, np.pi, -np.pi / 2], 
                          [0, np.pi / 5.6, np.pi / 6.7, np.pi / 4.3] # random angles to detect nonrotatable bond
                          ]:
            similar_rmsds = []
            for angle in angle_set:
                torsion_updates = np.zeros(num_rotatable_bonds).astype(np.float32)
                torsion_updates[i] = angle
            
                new_pos = apply_tor_changes_to_pos(np.copy(init_pos), ligand.rotatable_bonds, ligand.mask_rotate, 
                                                   torsion_updates, is_reverse_order=True, shift_center_back=False)
                try:
                    rmsd = get_symmetry_rmsd(copy.deepcopy(mol_part), np.copy(init_pos)[cur_mask], new_pos[cur_mask], 
                                            mol2=None, return_permutation=False)
                except:
                    rmsd = 1000
    
                if rmsd < 0.2:
                    similar_rmsds.append(rmsd)
            if len(similar_rmsds) == len(angle_set):
                bond_period = angle_set[0]
    
        bond_periods.append(bond_period)
    return np.array(bond_periods)


def get_rotatable_bonds_and_mask_rotate(mol):
    """
    Identify rotatable bonds in a molecule and create a rotation mask.

    This function takes a molecule as input, identifies rotatable single bonds,
    and creates a mask indicating which parts of the molecule can rotate around
    these bonds.

    Parameters:
    mol (rdkit.Chem.Mol): An RDKit molecule object.

    Returns:
    tuple:
        - rotatable_bonds (list of tuples): A list of tuples, each representing
          a rotatable bond and its neighboring atoms. Each tuple is of the form
          (neighbor_atom_1, atom_1, atom_2, neighbor_atom_2).
        - mask_rotate (numpy.ndarray): A boolean array where each row corresponds
          to a rotatable bond and each column corresponds to an atom in the molecule.
          An entry is True if the corresponding atom is part of the rotation group
          for that bond.

    Notes:
    - A rotatable bond is defined as a single bond that, when removed, splits the
      molecule into two separate connected components.
    - The mask indicates which atoms are part of the substructure that can rotate
      around each rotatable bond.

    Example:
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> AllChem.EmbedMolecule(mol)
    >>> rotatable_bonds, mask_rotate = get_rotatable_bonds_and_mask_rotate(mol)
    >>> print(rotatable_bonds)
    >>> print(mask_rotate)
    """
    rotatable_bonds = []
    to_rotate = {}

    G = mol_to_graph(mol)

    if len(list(nx.connected_components(G))) > 1:
        raise ValueError('molecule is not connected')
    
    # Find rotatable bonds and prepare transformation masks
    for e in G.edges():
        bond = mol.GetBondBetweenAtoms(e[0], e[1])
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            G2 = copy.deepcopy(G)
            G2.remove_edge(*e)
            if not nx.is_connected(G2):
                l = list(sorted(nx.connected_components(G2), key=len)[0])
                if len(l) > 1:
                    neighbors0 = list(G2.neighbors(e[0]))
                    neighbors1 = list(G2.neighbors(e[1]))
                    if e[0] in l:
                        rotatable_bonds.append((neighbors1, e[1], e[0], neighbors0))
                    else:
                        rotatable_bonds.append((neighbors0, e[0], e[1], neighbors1))
                    to_rotate[len(rotatable_bonds) - 1] = l
                    continue

    mask_rotate = np.zeros((len(rotatable_bonds), len(G.nodes())), dtype=bool)
    for idx, rotated_nodes_indices in to_rotate.items():
        mask_rotate[idx][rotated_nodes_indices] = True

    bonds_to_neighbors = {(bond[1], bond[2]): (bond[0], bond[3]) for bond in rotatable_bonds}
    if len(rotatable_bonds) > 0:
        rotatable_bonds = np.array([[bond[1], bond[2]] for bond in rotatable_bonds]) # total_number_of_rotatable_bonds × 2

    pos = mol.GetConformer(0).GetPositions().astype(np.float32)
    mask_rotate_after_fixing = fix_mask_rotate_using_bonds_on_one_line(rotatable_bonds, np.copy(mask_rotate), pos)
    bond_periods = get_rotatable_bond_periods(rotatable_bonds, mask_rotate_after_fixing, pos, mol)

    good_torsion_period_mask = bond_periods > np.pi / 12
    if good_torsion_period_mask.sum() < len(bond_periods):
        logger.debug('Filtering out rotatable bonds with no torsion period (truly nonrot bonds)')

        rotatable_bonds = rotatable_bonds[good_torsion_period_mask]
        mask_rotate = mask_rotate[good_torsion_period_mask]
        mask_rotate_after_fixing = mask_rotate_after_fixing[good_torsion_period_mask]
        bond_periods = bond_periods[good_torsion_period_mask]
        logger.debug(f'bond_periods: {np.round(bond_periods, 2)}')

    # Add neighbors to rotatable bonds
    neighbors_of_start = []
    neighbors_of_end = []
    for bond in rotatable_bonds:
        neighbors = bonds_to_neighbors[(bond[0], bond[1])]
        neighbors_of_start.append(neighbors[0])
        neighbors_of_end.append(neighbors[1])

    rotatable_bonds = np.array(rotatable_bonds)
    return rotatable_bonds, mask_rotate, mask_rotate_after_fixing, bond_periods, neighbors_of_start, neighbors_of_end


def get_rotatable_and_nonrotatable_bonds(mol):
    """
    Identify all bonds in a molecule and separate them into rotatable and non-rotatable bonds.

    This function takes a molecule as input, identifies all bonds, and categorizes them
    based on their ability to rotate.

    Parameters:
    mol (rdkit.Chem.Mol): An RDKit molecule object.

    Returns:
    tuple:
        - rotatable_bonds (Bonds object): A Bonds object that contains rotatable bonds.
    """

    def add_bond(bond_dict, *, start, end, bond_type, length,
                 is_aromatic, is_conjugated, is_in_ring, is_rotatable,
                 bond_period=None,
                 angle=None,
                 mask_rotate=None,
                 neighbor_of_start=None,
                 neighbor_of_end=None,
                 all_neighbors_of_start=None,
                 all_neighbors_of_end=None):
        bond_dict["start"].append(start)
        bond_dict["end"].append(end)
        bond_dict["bond_type"].append(bond_type)
        bond_dict["bond_periods"].append(bond_period)
        bond_dict["length"].append(length)
        bond_dict["mask_rotate"].append(mask_rotate)
        bond_dict["is_aromatic"].append(is_aromatic)
        bond_dict["is_conjugated"].append(is_conjugated)
        bond_dict["is_in_ring"].append(is_in_ring)
        bond_dict["is_rotatable"].append(is_rotatable)
        bond_dict["neighbor_of_start"].append(neighbor_of_start)
        bond_dict["neighbor_of_end"].append(neighbor_of_end)
        bond_dict["all_neighbors_of_start"].append(all_neighbors_of_start)
        bond_dict["all_neighbors_of_end"].append(all_neighbors_of_end)
        bond_dict["angles"].append(angle)
    empty_bond = {
        "start": [],
        "end": [],
        "bond_type": [],
        "bond_periods": [],
        "length": [],
        "mask_rotate": [],
        "is_aromatic": [],
        "is_conjugated": [],
        "is_in_ring": [],
        "is_rotatable": [],
        "neighbor_of_start": [],
        "neighbor_of_end": [],
        "all_neighbors_of_start": [],
        "all_neighbors_of_end": [],
        "angles": [],
    }
    rotatable_bonds = copy.deepcopy(empty_bond)

    rot_bonds, mask_rotate_before_fixing, mask_rotate_after_fixing, bond_periods, neighbors_of_start, neighbors_of_end = get_rotatable_bonds_and_mask_rotate(mol)
                               
    for (start, end), cur_mask_after, bond_period, all_n0, all_n1 in zip(rot_bonds, mask_rotate_after_fixing, bond_periods, neighbors_of_start, neighbors_of_end):
        start = int(start)
        end = int(end)
        all_n0 = np.array(all_n0)
        all_n1 = np.array(all_n1)
        # We add +1 for all categorical features because we want to use 0 as padding index in nn.Embedding layers.
        bond = mol.GetBondBetweenAtoms(start, end)  # Get the bond object directly
        # bond_type = bond.GetBondTypeAsDouble()  # Get bond type as a numerical value (Real)
        bond_type_int = BOND_TYPE_MAPPING[str(bond.GetBondType())] + 1  # Map bond type to int
        length = float(Chem.rdMolTransforms.GetBondLength(mol.GetConformer(), start, end))  # Pass the Conformer object
        is_aromatic = int(bond.GetIsAromatic()) + 1  # Check if the bond is aromatic (Categorical: True/False)
        is_conjugated = int(bond.GetIsConjugated()) + 1  # Check if the bond is conjugated (Categorical: True/False)
        is_in_ring = int(bond.IsInRing()) + 1  # Check if the bond is part of a ring (Categorical)

        add_bond(rotatable_bonds,
                start=start,
                end=end,
                all_neighbors_of_start=all_n0,
                all_neighbors_of_end=all_n1,
                bond_type=bond_type_int,  # Use the integer bond type
                length=length,
                is_aromatic=is_aromatic,
                is_conjugated=is_conjugated,
                is_in_ring=is_in_ring,
                is_rotatable=True,
                mask_rotate=cur_mask_after,
                bond_period=bond_period,
                )
                        
    rotatable_bonds = {key: np.array(value) if not key.startswith('all_neighbors_of_') else value for key, value in rotatable_bonds.items()}
    return Bonds(**rotatable_bonds), rot_bonds, mask_rotate_before_fixing, mask_rotate_after_fixing, bond_periods
