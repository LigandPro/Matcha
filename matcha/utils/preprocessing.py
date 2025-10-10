import os
import warnings

import numpy as np
import struct
import torch
from Bio.PDB import PDBParser
from rdkit.Chem.rdchem import BondType as BT
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select

import prody
from prody import confProDy
confProDy(verbosity='none')


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
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}

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


def generate_multiple_conformers(mol, num_conformers):
    ps = AllChem.ETKDGv3()
    failures, ids = 0, []
    while failures < 3 and len(ids) == 0:
        if failures > 0:
            print(
                f'rdkit coords could not be generated. trying again {failures}.')
        ids = AllChem.EmbedMultipleConfs(mol, num_conformers, ps)
        ids = [id for id in ids]
        ids = [id for id in ids if id != -1]
        failures += 1
    if len(ids) == 0:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMultipleConfs(mol, min(num_conformers, 10), ps)
        for i in range(mol.GetNumConformers()):
            AllChem.MMFFOptimizeMolecule(mol, confId=i)
        return True
    return False


def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def parse_receptor(pdbid, pdbbind_dir, dataset_type):
    rec = parsePDB(pdbid, pdbbind_dir, dataset_type)
    return rec


def parsePDB(pdbid, pdbbind_dir, dataset_type):
    if dataset_type == 'pdbbind_conf' or dataset_type == 'dockgen_full_conf':
        rec_path = os.path.join(
            pdbbind_dir, pdbid, f'{pdbid}_protein_processed.pdb')
    elif dataset_type == 'posebusters_conf' or dataset_type == 'astex_conf' or dataset_type == 'any_conf':
        rec_path = os.path.join(pdbbind_dir, pdbid, f'{pdbid}_protein.pdb')
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')
    protein = parse_pdb_from_path(rec_path)
    return protein


def parse_pdb_from_path(path):
    pdb = prody.parsePDB(path)
    return pdb


def get_coords(prody_pdb):
    resindices = sorted(set(prody_pdb.ca.getResindices()))
    coords = np.full((len(resindices), 14, 3), np.nan)
    atom_names = np.full((len(resindices), 14), np.nan).astype(object)
    for i, resind in enumerate(resindices):
        sel = prody_pdb.select(f'resindex {resind}')
        resname = sel.getResnames()[0]
        for j, name in enumerate(atom_order[aa_long2short[resname] if resname in aa_long2short else 'X']):
            sel_resnum_name = sel.select(f'name {name}')
            if sel_resnum_name is not None:
                coords[i, j, :] = sel_resnum_name.getCoords()[0]
                atom_names[i, j] = sel_resnum_name.getElements()[0]
            else:
                coords[i, j, :] = [np.nan, np.nan, np.nan]
                atom_names[i, j] = 'X'
    return coords, atom_names


def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".sdf") and 'rdkit' not in file:
            lig = read_molecule(os.path.join(
                pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
            # read mol2 file if sdf file cannot be sanitized
            if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):
                print(
                    'Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
                lig = read_molecule(os.path.join(
                    pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True)
            if lig is not None:
                ligs.append(lig)
    return ligs


def read_molecule(molecule_file, sanitize=False, remove_hs=False):
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
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
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
            except Exception as e:
                print("RDKit was unable to sanitize the molecule.")

        if remove_hs:
            try:
                mol = Chem.RemoveAllHs(mol, sanitize=sanitize)
            except Exception as e:
                print("RDKit was unable to remove hydrogen atoms from the molecule.")
                mol = Chem.RemoveAllHs(mol, sanitize=False)

    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None

    return mol


def read_sdf_with_multiple_confs(molecule_file, sanitize=False, remove_hs=False):
    supplier = Chem.SDMolSupplier(
        molecule_file, sanitize=False, removeHs=False)
    mols = []
    print(f'Reading {len(supplier)} conformations from', molecule_file)
    for mol in supplier:
        try:
            if sanitize:
                Chem.SanitizeMol(mol)
            if remove_hs:
                mol = Chem.RemoveAllHs(mol, sanitize=sanitize)
        except Exception as e:
            print(e)
            print("RDKit was unable to read the molecule.")
            mol = None

        if mol is not None:
            mols.append(mol)
    return mols


def save_multiple_confs(mol, output_conf_path, num_conformers):
    mol.RemoveAllConformers()
    mol = Chem.AddHs(mol)

    generate_multiple_conformers(mol, num_conformers)
    mol = Chem.RemoveAllHs(mol)

    writer = Chem.SDWriter(output_conf_path)
    for cid in range(mol.GetNumConformers()):
        mol.SetProp('ID', f'conformer_{cid}')
        writer.write(mol, confId=cid)


def extract_receptor_structure_prody(rec, sequences_to_embeddings):
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
    seq = rec.ca.getSequence()
    coords, atom_names = get_coords(rec)

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
    chain_distances = {}
    for i, chain_id in enumerate(chain_ids):
        chain_mask = res_chain_ids == chain_id
        chain_seq = ''.join(seq[chain_mask])
        chain_coords = coords[chain_mask]

        chain_atom_names = atom_names[chain_mask]

        nonempty_coords = chain_coords.reshape(-1, 3)
        notnan_mask = np.isnan(nonempty_coords).sum(axis=1) == 0
        nonempty_coords = nonempty_coords[notnan_mask]

        chain_atom_names = chain_atom_names.reshape(-1)
        chain_atom_names = chain_atom_names[notnan_mask]

        embeddings, tokenized_seq = sequences_to_embeddings[chain_seq]
        sequences.append(tokenized_seq)
        lm_embeddings.append(embeddings)
        c_alpha_coords.append(chain_coords[:, 1].astype(np.float32))
        full_coords.append(nonempty_coords)
        full_atom_names.append(chain_atom_names)

    if len(c_alpha_coords) == 0:
        print('NO VALID CHAIN!!!')
        print(chain_distances)
        return None, None, None, None, None, None

    chain_lengths = [len(seq) for seq in sequences]
    c_alpha_coords = np.concatenate(c_alpha_coords, axis=0)  # [n_residues, 3]
    full_coords = np.concatenate(full_coords, axis=0)  # [n_protein_atoms, 3]
    full_atom_names = np.concatenate(full_atom_names, axis=0)
    lm_embeddings = np.concatenate(lm_embeddings, axis=0)
    sequences = np.concatenate(sequences, axis=0)

    return c_alpha_coords, lm_embeddings, sequences, chain_lengths, full_coords, full_atom_names
