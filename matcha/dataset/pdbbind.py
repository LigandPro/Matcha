import torch
import os
import copy
import re
import pickle
import numpy as np
from deli import load
from collections import defaultdict
from typing import List, Optional
from torch.utils.data import Dataset
from rdkit.Chem import RemoveAllHs
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from torch.nn.utils.rnn import pad_sequence
from matcha.dataset.complex_dataclasses import Ligand, Protein, Complex, LigandBatch, ProteinBatch, ComplexBatch, BondsBatch
from matcha.utils.preprocessing import (parse_receptor, read_pdbbind_mols,
                                          extract_receptor_structure_prody, lig_atom_featurizer,
                                          read_molecule, save_multiple_confs, read_sdf_with_multiple_confs)
from matcha.utils.bond_processing import get_rotatable_and_nonrotatable_bonds, split_molecule
from matcha.utils.transforms import (
    apply_tor_changes_to_pos, get_torsion_angles, find_rigid_alignment, get_bond_properties_for_angles)
from matcha.utils.log import get_logger

logger = get_logger(__name__)

protein_atom_names = {"C": 0, "N": 1, "O": 2,
                      "S": 3, "unknown": 4, "padding": 5}


def get_ligand_without_randomization(mol_, protein_center=None, parse_rotbonds=True):
    """
    Fill the fields of a Ligand object that are not randomized.

    Parameters:
    ----------
    mol_ : rdkit.Chem.Mol
        The input molecule.
    protein_center : numpy.ndarray, optional
        The center of the protein (default is None).

    Returns:
    -------
    Ligand
        The Ligand object with the filled fields.
    """
    mol_maybe_noh = copy.deepcopy(mol_)

    try:
        mol_maybe_noh = RemoveAllHs(mol_maybe_noh, sanitize=True)
    except Exception as e:
        mol_maybe_noh = RemoveAllHs(mol_maybe_noh, sanitize=False)

    if parse_rotbonds:
        rotatable_bonds_ext, rotatable_bonds, mask_rotate_before_fixing, mask_rotate_after_fixing, bond_periods = get_rotatable_and_nonrotatable_bonds(
            mol_maybe_noh)
        if len(rotatable_bonds) == 0:
            logger.warning(
                f"No rotatable bonds found for ligand, but still using the molecule.")

    ligand = Ligand()
    ligand.pos = mol_maybe_noh.GetConformer(0).GetPositions().astype(np.float32) - protein_center

    ligand.orig_mol = mol_maybe_noh  # original mol
    # features are conformer-invariant
    try:
        ligand.x = lig_atom_featurizer(mol_maybe_noh)
    except Exception as e:
        logger.warning(f"Failed to featurize ligand: {e}")
        ligand.x = None
    
    ligand.final_tr = ligand.pos.mean(0).astype(np.float32).reshape(1, 3)

    if parse_rotbonds:
        # Fill ligand properties
        ligand.rotatable_bonds_ext = copy.deepcopy(rotatable_bonds_ext)
        if len(rotatable_bonds) > 0:
            ligand.rotatable_bonds = rotatable_bonds
            ligand.mask_rotate = mask_rotate_after_fixing
            ligand.mask_rotate_before_fixing = mask_rotate_before_fixing
            ligand.bond_periods = bond_periods
            ligand.init_tor = np.zeros(
                ligand.rotatable_bonds.shape[0], dtype=np.float32)
            assert ligand.rotatable_bonds.shape[0] == ligand.rotatable_bonds_ext.start.shape[0]
        else:
            ligand.rotatable_bonds = np.array([], dtype=np.int32)
            ligand.mask_rotate = np.array([], dtype=np.int32)
            ligand.mask_rotate_before_fixing = np.array([], dtype=np.int32)
            ligand.init_tor = np.array([], dtype=np.float32)
            ligand.bond_periods = np.array([], dtype=np.float32)  # Initialize as empty array instead of None

        ligand.t = None
        ligand.init_rot = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
    else:
        # If parse_rotbonds is False, ensure bond_periods is initialized
        if not hasattr(ligand, 'bond_periods') or ligand.bond_periods is None:
            ligand.bond_periods = np.array([], dtype=np.float32)
    return ligand


def randomize_ligand_with_preds(ligand: Ligand, with_preds: bool,  tr_mean: float = 0., tr_std: float = 5.):
    """
    Randomize the position, rotation, and torsion of a ligand.

    Parameters:
    ----------
    ligand : Ligand
        The input ligand to be randomized.

    Returns:
    -------
    None
    """
    pos = np.copy(ligand.orig_pos)

    # Tr:
    if with_preds:
        tr = ligand.pred_tr.reshape(1, 3)
    else:
        tr = np.random.normal(0, 5, 3).astype(np.float32).reshape(1, 3)

    # Rot:
    rot = R.random().as_matrix().astype(np.float32)

    # apply predicted rotation and translation
    pos = (pos - pos.mean(axis=0).reshape(1, 3)) @ rot.T + tr.reshape(1, 3)

    # Tor:
    num_rotatable_bonds = ligand.rotatable_bonds.shape[0]
    if num_rotatable_bonds > 0:
        torsion_updates = np.random.uniform(
            -ligand.bond_periods / 2, ligand.bond_periods / 2)
    else:
        torsion_updates = np.empty(0).astype(np.float32)

    pos = apply_tor_changes_to_pos(pos, ligand.rotatable_bonds, ligand.mask_rotate,
                                   torsion_updates, is_reverse_order=True)

    ligand.init_tr = tr.reshape(1, 3)
    ligand.final_rot = rot.T[None, :, :]
    ligand.final_tor = -torsion_updates
    ligand.pos = np.copy(pos)

    # Time is randomized from Uniform[0, 1]:
    ligand.t = torch.rand(1)
    if ligand.rmsd is None:
        ligand.rmsd = torch.zeros(1)
    ligand.stage_num = torch.tensor([2])


def set_ligand_data_from_preds(ligand: Ligand, name: str):
    """
    Randomize the position, rotation, and torsion of a ligand.

    Parameters:
    ----------
    ligand : Ligand
        The input ligand to be randomized.

    Returns:
    -------
    None
    """
    true_pos = np.copy(ligand.orig_pos)
    pred_pos = np.copy(ligand.predicted_pos)

    # Tor:
    num_rotatable_bonds = ligand.rotatable_bonds.shape[0]
    if num_rotatable_bonds > 0:
        bond_properties_for_angles = get_bond_properties_for_angles(
            ligand.rotatable_bonds_ext)
        true_bond_periods = bond_properties_for_angles['bond_periods']
        bond_properties_for_angles['bond_periods'] = np.ones_like(
            true_bond_periods) * 2 * np.pi
        angles_true = get_torsion_angles(
            np.copy(true_pos), bond_atoms_for_angles=bond_properties_for_angles)
        angles_pred = get_torsion_angles(
            np.copy(pred_pos), bond_atoms_for_angles=bond_properties_for_angles)
        torsion_updates = angles_pred - angles_true

        pos_new = apply_tor_changes_to_pos(np.copy(pred_pos), ligand.rotatable_bonds, ligand.mask_rotate_before_fixing,
                                           torsion_updates, is_reverse_order=True)
    else:
        torsion_updates = np.empty(0).astype(np.float32)
        pos_new = np.copy(pred_pos)

    # compute tr and rot alignment
    rot_align, _ = find_rigid_alignment(pos_new, true_pos)

    ligand.init_tr = ligand.pred_tr.reshape(1, 3)
    ligand.final_rot = rot_align[None, :, :]
    ligand.final_tor = torsion_updates
    ligand.pos = pred_pos

    # Time is randomized from Uniform[0, 1]:
    ligand.t = torch.rand(1)
    if ligand.rmsd is None:
        ligand.rmsd = torch.zeros(1)
    ligand.stage_num = torch.tensor([3])


def randomize_complex(complex: Complex, std_protein_pos: float,
                      esm_emb_noise_std: float,
                      std_lig_pos: float, ligand_mask_ratio: float, protein_mask_ratio: float,
                      tr_mean: float, tr_std: float, use_pred_ligand_transforms: bool = False,
                      use_predicted_tr_only: bool = True, randomize_bond_neighbors: bool = True,
                      stage_num: int = None):

    # 0. Rotate complex
    apply_random_rotation_inplace(complex)

    # Add noise to ESM embeddings
    complex.protein.x += np.random.normal(0,
                                          esm_emb_noise_std, complex.protein.x.shape)

    # Add noise to protein and ligand atom positions
    complex.protein.pos += np.random.normal(0,
                                            std_protein_pos, complex.protein.pos.shape)
    complex.ligand.pos += np.random.normal(0,
                                           std_lig_pos, complex.ligand.pos.shape)

    # 5. Compute ligand gt values
    complex.set_ground_truth_values()

    # Randomly mask protein and ligand atoms
    complex.randomly_mask_complex(
        ligand_mask_ratio=ligand_mask_ratio, protein_mask_ratio=protein_mask_ratio)

    # 6. Randomly sample neighbors of rotatable bonds
    if randomize_bond_neighbors:
        complex.ligand.randomly_sample_neighbors_of_rotatable_bonds()
    else:
        complex.ligand.sample_first_neighbor_of_rotatable_bonds()

    # 7. Randomize ligand for NN input
    if use_pred_ligand_transforms:
        if use_predicted_tr_only:
            randomize_ligand_with_preds(complex.ligand, with_preds=True) # stage 2
        else:
            set_ligand_data_from_preds(complex.ligand, complex.name) # stage 3
    else:
        randomize_ligand_with_preds(
            complex.ligand, tr_mean=tr_mean, tr_std=tr_std, with_preds=False)
    return complex


class PDBBind(Dataset):
    def __init__(self, data_dir, split_path, esm_embeddings_path, sequences_path,
                 tr_std=1., tr_mean=None, cache_path='data/cache', num_dataset_workers=1,
                 std_protein_pos=0.1, std_lig_pos=0.1,
                 ligand_mask_ratio=0., protein_mask_ratio=0.,
                 esm_emb_noise_std=0.0,
                 predicted_ligand_transforms_path=None, dataset_type='pdbbind',
                 add_all_atom_pos=False,
                 use_predicted_tr_only=True, randomize_bond_neighbors=True,
                 data_dir_conf=None, is_train_dataset=True,
                 n_preds_to_use=1, use_all_chains=False,
                 min_lig_size=7, stage_num=None,
                 n_confs_override=None):
        self.data_dir = data_dir
        self.data_dir_conf = data_dir_conf
        if dataset_type.endswith('_conf'):
            self.n_confs_to_use = min(
                10, n_preds_to_use) if n_confs_override is None else n_confs_override
        else:
            self.n_confs_to_use = 0
        self.esm_embeddings_path = esm_embeddings_path
        self.sequences_path = sequences_path
        self.cache_path = cache_path
        self.num_dataset_workers = num_dataset_workers
        self.dataset_type = dataset_type
        self.tr_std = tr_std
        self.tr_mean = tr_mean
        self.std_protein_pos = std_protein_pos
        self.std_lig_pos = std_lig_pos
        self.ligand_mask_ratio = ligand_mask_ratio
        self.protein_mask_ratio = protein_mask_ratio
        self.esm_emb_noise_std = esm_emb_noise_std
        self.use_pred_ligand_transforms = predicted_ligand_transforms_path is not None
        self.add_all_atom_pos = add_all_atom_pos
        self.use_predicted_tr_only = use_predicted_tr_only
        self.randomize_bond_neighbors = randomize_bond_neighbors
        self.is_train_dataset = is_train_dataset
        self.use_all_chains = use_all_chains
        self.min_lig_size = min_lig_size
        self.stage_num = stage_num

        self.split_path = split_path

        self.full_cache_path = self._get_cache_folder_path()

        # TODO keep 0 for padding
        aa_list = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
                   'R', 'S', 'T', 'V', 'W', 'Y']
        self.aa_mapping = {aa: i for i, aa in enumerate(aa_list)}

        # loads data to self.complexes list:
        logger.debug(f"Cache path: {self.full_cache_path}")
        if os.path.exists(os.path.join(self.full_cache_path, 'complexes.pkl')):
            self._load_from_cache()
        else:
            os.makedirs(self.full_cache_path, exist_ok=True)
            self._preprocess_and_save_to_cache()

        if self.dataset_type.endswith('_conf'):
            self._set_all_conformer_proteins()

        # save orig_pos_before_augm for each ligand
        complexes = []
        for idx in range(len(self.complexes)):
            complex = self.complexes[idx]
            complex.ligand.true_pos = np.copy(complex.ligand.pos)

            complex.ligand.orig_pos_before_augm = np.copy(
                complex.ligand.pos)
            complexes.append(complex)
        self.complexes = complexes

        if self.is_train_dataset:
            logger.debug(f"Complexes count before filtering: {len(self.complexes)}")
            self.complexes = [complex for complex in self.complexes if complex.ligand.pos.shape[0] <
                              150 and complex.ligand.pos.shape[0] > 6 and complex.protein.pos.shape[0] + complex.ligand.pos.shape[0] < 2000]
            logger.debug(f"Complexes count after filtering: {len(self.complexes)}")

        # self.complexes = [compl for compl in self.complexes
        #                   if compl.protein.pos.shape[0] + compl.ligand.pos.shape[0] > 1000 or compl.ligand.pos.shape[0] > 70]


        if self.dataset_type.endswith('_conf'):
            self._explode_ligand_conformers(n_preds_to_use)
        else:
            self.complexes = self.complexes * n_preds_to_use
        if self.use_pred_ligand_transforms:
            self._set_predicted_ligand_transforms(
                predicted_ligand_transforms_path, n_preds_to_use)

    def _explode_ligand_conformers(self, n_preds_to_use):
        name2complexes = defaultdict(list)
        for complex in self.complexes:
            name2complexes[complex.name.split('_conf')[0]].append(complex)
        new_complexes = []
        for name, conformers in name2complexes.items():
            while len(conformers) < n_preds_to_use:
                new_conformers = copy.deepcopy(conformers)
                for i, conformer in enumerate(new_conformers):
                    conformer.name = conformer.name.split(
                        '_conf')[0] + f'_conf{len(conformers)+i}'
                conformers = conformers + new_conformers
            if len(conformers) > n_preds_to_use:
                conformers = conformers[:n_preds_to_use]
            new_complexes.extend(conformers)

        self.complexes = new_complexes

    def _set_all_conformer_proteins(self):
        name2protein = {}
        for complex in self.complexes:
            if complex.name.endswith('_conf0'):
                # copy.deepcopy(complex.protein)
                name2protein[complex.name.split('_conf')[0]] = complex.protein

        logger.debug(f"Name2protein length: {len(name2protein)}; complexes length: {len(self.complexes)}")

        new_complexes = []
        for complex in self.complexes:
            if not complex.name.endswith('_conf0'):
                # copy.deepcopy(name2protein[complex.name.split('_conf')[0]])
                complex.protein = name2protein[complex.name.split('_conf')[0]]
            new_complexes.append(complex)
        self.complexes = new_complexes

    def reset_predicted_ligand_transforms(self, predicted_ligand_transforms_path, n_preds_to_use):
        self.use_pred_ligand_transforms = True
        self._set_predicted_ligand_transforms(
            predicted_ligand_transforms_path, n_preds_to_use)

    def _set_predicted_ligand_transforms(self, predicted_ligand_transforms_path, n_preds_to_use):

        self.predicted_ligand_transforms = np.load(
            predicted_ligand_transforms_path, allow_pickle=True)[0]
        self.n_repeats = 1
        n_preds_to_use_real = min(n_preds_to_use, len(
            self.predicted_ligand_transforms[self.complexes[0].name]))
        self.complexes = [
            complex for complex in self.complexes if complex.name in self.predicted_ligand_transforms]

        # initialize extended complexes
        extended_complexes = []
        processed_names = set()
        for complex in tqdm(self.complexes, desc='Setting predicted ligand transforms...'):
            if complex.name in processed_names:
                continue
            processed_names.add(complex.name)
            for i in range(n_preds_to_use_real):
                extended_complex = copy.deepcopy(complex)
                pred_data = self.predicted_ligand_transforms[complex.name][i]
                extended_complex.ligand.pred_tr = pred_data['tr_pred_init'] + \
                    pred_data['full_protein_center'] - \
                    extended_complex.protein.full_protein_center
                extended_complex.ligand.pred_tr = extended_complex.ligand.pred_tr.astype(
                    np.float32)

                if not self.use_predicted_tr_only:
                    pred_pos = pred_data['transformed_orig'] + pred_data['full_protein_center'] - \
                        extended_complex.protein.full_protein_center
                    extended_complex.ligand.predicted_pos = pred_pos

                extended_complexes.append(extended_complex)
        self.complexes = extended_complexes

    def __len__(self):
        return len(self.complexes)

    def __get_nonrand_item__(self, idx):
        complex_idx = idx
        complex = copy.deepcopy(self.complexes[complex_idx])
        return complex

    def __getitem__(self, idx):
        complex = self.__get_nonrand_item__(idx)
        complex = randomize_complex(complex=complex,
                                    std_protein_pos=self.std_protein_pos, std_lig_pos=self.std_lig_pos,
                                    ligand_mask_ratio=self.ligand_mask_ratio,
                                    protein_mask_ratio=self.protein_mask_ratio,
                                    tr_mean=self.tr_mean, tr_std=self.tr_std,
                                    esm_emb_noise_std=self.esm_emb_noise_std,
                                    use_pred_ligand_transforms=self.use_pred_ligand_transforms,
                                    use_predicted_tr_only=self.use_predicted_tr_only,
                                    randomize_bond_neighbors=self.randomize_bond_neighbors,
                                    stage_num=self.stage_num)
        return complex

    def _get_cache_folder_path(self):
        split_name = os.path.basename(
            self.split_path) if self.split_path is not None else 'full'
        values_for_cache_path = [self.dataset_type, f'{self.n_confs_to_use}conformations',
                                 os.path.basename(self.esm_embeddings_path),
                                 split_name]
        str_for_cache_path = map(str, values_for_cache_path)
        args_str = '_'.join(str_for_cache_path)
        # replace any unsafe characters:
        pattern = r'[^A-Za-z0-9\-_]'
        safe_args_str = re.sub(pattern, '_', args_str)
        if self.use_all_chains:
            safe_args_str = f'allchains_' + safe_args_str
        cache_folder_path = os.path.join(self.cache_path, safe_args_str)
        return cache_folder_path

    def _load_embeddings(self, embeddings_path, sequences_path, complex_names):
        try:
            id_to_embeddings = torch.load(embeddings_path, weights_only=False)
            id_to_sequence = load(sequences_path)
        except FileNotFoundError:
            raise ValueError(
                f"Embeddings file not found at {embeddings_path} or sequences file not found at {sequences_path}")
        except Exception as e:
            raise ValueError(
                f"An error occurred while loading embeddings: {e}")

        chain_embeddings_dictlist = defaultdict(list)
        chain_sequences_dictlist = defaultdict(list)
        tokenized_chain_sequences_dictlist = defaultdict(list)

        complex_names_set = set(complex_names)
        for key_base, embedding in id_to_embeddings.items():
            keys_all = [key_base]
            for key in keys_all:
                try:
                    key_name = '_'.join(key.split('_')[:-2])  # cut _chain_i
                except IndexError:
                    raise ValueError(
                        f"Invalid key format in embeddings: {key}")

                if key_name in complex_names_set:

                    tokenized_aa_sequence = np.array(
                        [self.aa_mapping.get(aa, 0) for aa in id_to_sequence[key]])[:, None]
                    aa_sequence = np.array([aa for aa in id_to_sequence[key]])

                    if '_superlig' in key_name:
                        key_name = key_name.split('_superlig')[0]
                    chain_embeddings_dictlist[key_name].append(embedding)
                    chain_sequences_dictlist[key_name].append(aa_sequence)
                    tokenized_chain_sequences_dictlist[key_name].append(
                        tokenized_aa_sequence)

        lm_embeddings_chains_all = [chain_embeddings_dictlist.get(
            name, []) for name in complex_names]
        sequence_chains_all = [chain_sequences_dictlist.get(
            name, []) for name in complex_names]
        tokenized_sequence_chains_all = [
            tokenized_chain_sequences_dictlist.get(name, []) for name in complex_names]
        logger.info("LLM embeddings are loaded")
        return lm_embeddings_chains_all, sequence_chains_all, tokenized_sequence_chains_all

    def _process_complex(self, complex_names, sequences_to_embeddings):
        try:
            return self._get_complex(complex_names, sequences_to_embeddings)
        except Exception as e:
            logger.error(f"Error processing complex: {e}")
            return None

    def _preprocess_and_save_to_cache(self):
        if self.split_path is not None and os.path.exists(self.split_path):
            logger.info(
                f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')
            # Get names of complexes:
            with open(self.split_path, 'r') as file:
                lines = file.readlines()
                complex_names_all = [line.rstrip() for line in lines]
            logger.info(f"Complexes loaded from {self.split_path}")
        else:
            complex_names_all = [name for name in os.listdir(
                self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        logger.info(f"Loading {len(complex_names_all)} complexes")

        # Load embeddings:
        lm_embeddings_chains_all, sequence_chains_all, tokenized_sequence_chains_all = self._load_embeddings(self.esm_embeddings_path,
                                                                                                             self.sequences_path,
                                                                                                             complex_names_all)
        self.complexes = []
        with tqdm(total=len(complex_names_all), desc='Loading complexes') as pbar:
            for complex_name, lm_embeddings, sequence_chains, tokenized_sequence_chains in zip(complex_names_all,
                                                                                                lm_embeddings_chains_all,
                                                                                                sequence_chains_all,
                                                                                                tokenized_sequence_chains_all):
                sequences_to_embeddings = {''.join(seq): (emb, tokenized_seq) for seq, emb, tokenized_seq in zip(sequence_chains, lm_embeddings,
                                                                                                                    tokenized_sequence_chains)}
                processed_complexes = self._process_complex(
                    [complex_name], sequences_to_embeddings)
                if processed_complexes is not None and len(processed_complexes) > 0:
                    if type(processed_complexes[0]) == list:
                        processed_complexes = [
                            complex for complex_list in processed_complexes for complex in complex_list]
                    self.complexes += processed_complexes
                    
                pbar.update()
        # Filter out empty complexes:
        self.complexes = [complex for complex in self.complexes if (
            complex.ligand is not None) and (complex.protein is not None)]

        # Save:
        filepath = os.path.join(self.full_cache_path, 'complexes.pkl')
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.complexes, f)
            logger.info(f"Data successfully saved to {filepath}")
        except IOError as e:
            logger.error(f"Error saving data to {filepath}: {e}")

    def _load_from_cache(self):
        filepath = os.path.join(self.full_cache_path, 'complexes.pkl')
        try:
            with open(filepath, 'rb') as f:
                self.complexes = pickle.load(f)
            logger.info(f"Data successfully loaded from {filepath}")
        except IOError as e:
            logger.error(f"Error loading data from {filepath}: {e}")

    def _get_complex(self, complex_names, sequences_to_embeddings):
        logger.debug(f"Complex names: {complex_names}")
        try:
            rec_model = parse_receptor(
                complex_names[0], self.data_dir, self.dataset_type)
        except Exception as e:
            logger.warning(f"Skipping {complex_names[0]} because of error: {e}")
            return [], []

        complexes = []
        for name in complex_names:
            logger.debug(f"Processing complex: {name}")

            orig_ligs = None
            if self.dataset_type == 'pdbbind':
                ligs = read_pdbbind_mols(self.data_dir, name, remove_hs=False)
            elif self.dataset_type.endswith('_conf'):
                if self.dataset_type == 'pdbbind_conf':
                    orig_ligs = read_pdbbind_mols(self.data_dir, name, remove_hs=False)
                elif self.dataset_type == 'posebusters_conf' or self.dataset_type == 'astex_conf' or \
                        self.dataset_type == 'any_conf':
                    orig_ligs = [read_molecule(os.path.join(
                        self.data_dir, name, f'{name}_ligand.sdf'), remove_hs=False, sanitize=True)]
                else:
                    orig_ligs = [read_molecule(os.path.join(
                        self.data_dir, name, f'{name}_ligand.pdb'), remove_hs=False, sanitize=True)]

                orig_ligs = [split_molecule(
                    lig_mol, min_lig_size=self.min_lig_size) for lig_mol in orig_ligs]
                orig_ligs = [
                    lig_mol for lig_mol_list in orig_ligs for lig_mol in lig_mol_list if lig_mol is not None]

                fname_with_confs = os.path.join(
                    self.data_dir_conf, f'{name}_conf.sdf')
                save_multiple_confs(
                    copy.deepcopy(orig_ligs[0]), fname_with_confs, num_conformers=self.n_confs_to_use)
                ligs = [read_sdf_with_multiple_confs(
                    fname_with_confs, remove_hs=False, sanitize=True)]

            elif self.dataset_type == 'dockgen' or self.dataset_type == 'dockgen_full':
                ligs = [read_molecule(os.path.join(
                    self.data_dir, name, f'{name}_ligand.pdb'), remove_hs=False, sanitize=True)]
            elif self.dataset_type == 'astex' or self.dataset_type == 'posebusters' or self.dataset_type == 'any':
                ligs = [read_molecule(os.path.join(
                    self.data_dir, name, f'{name}_ligand.sdf'), remove_hs=False, sanitize=True)]
            else:
                raise ValueError(f'Unknown dataset type: {self.dataset_type}')

            if len(ligs) > 0 and type(ligs[0]) == list:
                ligs = [split_molecule(
                    lig_mol, min_lig_size=self.min_lig_size) for lig_mol in ligs[0]]
                ligs = [
                    lig_mol for cur_lig_mol_list in ligs for lig_mol in cur_lig_mol_list if lig_mol is not None]
                ligs = [ligs]
            else:
                ligs = [split_molecule(
                    lig_mol, min_lig_size=self.min_lig_size) for lig_mol in ligs]
                ligs = [
                    lig_mol for lig_mol_list in ligs for lig_mol in lig_mol_list if lig_mol is not None]

            for lig_idx, lig_mol in enumerate(ligs):
                if type(lig_mol) == list:  # multiple conformations
                    lig_mol_list = lig_mol
                    lig_mol = lig_mol[0]
                else:
                    lig_mol_list = [lig_mol]

                try:
                    # Process protein:
                    if self.dataset_type.endswith('_conf'):
                        c_alpha_coords_list, lm_embeddings_list, sequences_list, chain_lengths, full_coords, full_atom_names, full_atom_residue_ids = extract_receptor_structure_prody(
                            rec_model, orig_ligs[lig_idx] if not self.use_all_chains else None, sequences_to_embeddings)
                    else:
                        c_alpha_coords_list, lm_embeddings_list, sequences_list, chain_lengths, full_coords, full_atom_names, full_atom_residue_ids = extract_receptor_structure_prody(
                            rec_model, lig_mol, sequences_to_embeddings)

                    # positions are positions of C-alpha, other positions are not used
                    if not self.add_all_atom_pos:
                        full_coords = None
                        full_atom_names = None
                        full_atom_residue_ids = None
                    protein = Protein(x=lm_embeddings_list, pos=c_alpha_coords_list, seq=sequences_list,
                                      all_atom_pos=full_coords, all_atom_names=full_atom_names, all_atom_residue_ids=full_atom_residue_ids)
                    protein_center = protein.pos.mean(axis=0).reshape(1, 3)
                    protein.pos -= protein_center
                    if protein.all_atom_pos is not None:
                        protein.all_atom_pos -= protein_center
                    protein.full_protein_center = protein_center
                    protein.chain_lengths = chain_lengths

                    # Process ligand:
                    parse_rotbonds = True
                    current_complexes = []
                    for conf_id, lig_mol in enumerate(lig_mol_list):
                        ligand = get_ligand_without_randomization(lig_mol, protein_center, parse_rotbonds=parse_rotbonds)
                        if parse_rotbonds:
                            ligand_with_bonds = copy.deepcopy(ligand)
                            cur_ligand = ligand
                        else:
                            cur_ligand = copy.deepcopy(ligand_with_bonds)
                            cur_ligand.pos = copy.deepcopy(ligand.pos)
                            cur_ligand.x = copy.deepcopy(ligand.x)
                            cur_ligand.final_tr = copy.deepcopy(ligand.final_tr)
                            cur_ligand.orig_mol = copy.deepcopy(ligand.orig_mol)

                        parse_rotbonds = False
                        complex = Complex()
                        complex.ligand = cur_ligand
                        if conf_id == 0:
                            complex.protein = copy.deepcopy(protein)
                        else:
                            complex.protein = []  # avoid copying protein in cache

                        if self.dataset_type.endswith('_conf'):
                            complex.name = f'{name}_mol{lig_idx}_conf{conf_id}'
                        else:
                            complex.name = f'{name}_mol{lig_idx}'
                        if complex.ligand.bond_periods is not None and len(complex.ligand.bond_periods) > 0:
                            logger.debug(f"Successfully parsed complex {complex.name} with bond periods: {np.round(complex.ligand.bond_periods, 2)}")
                        else:
                            logger.debug(f"Successfully parsed complex {complex.name} (no rotatable bonds)")
                        current_complexes.append(complex)

                    # Filter out complexes with incorrect ligand.x.
                    ok_complexes = [complex for complex in current_complexes if complex.ligand.x is not None]
                    logger.debug(f"Found {len(ok_complexes)} out of {len(current_complexes)} complexes with correct ligand.x for {name}")
                    if len(ok_complexes) > 0:
                        lig_features = ok_complexes[0].ligand.x
                        for complex in current_complexes:
                            if complex.ligand.x is None:
                                complex.ligand.x = copy.deepcopy(lig_features)
                    else:
                        logger.error(f"No complexes with correct ligand.x found for {name}")
                        current_complexes = []
                    
                    complexes += current_complexes

                except Exception as e:
                    logger.error(f"Skipping {name} because of error: {e}")
                    continue

        return complexes


class PDBBindWithSortedBatching(Dataset):
    def __init__(self, dataset, batch_limit, data_collator):
        self.dataset = dataset
        self.batch_limit = batch_limit
        self.data_collator = data_collator

        self._form_batches(batch_limit)

    def _init_sorted_indices(self):
        protein_lengths = np.array(
            [complex.protein.pos.shape[0] for complex in self.dataset.complexes])
        ligand_lengths = np.array([complex.ligand.pos.shape[0]
                                  for complex in self.dataset.complexes])
        sorted_indices = np.lexsort((ligand_lengths, protein_lengths))
        return protein_lengths + ligand_lengths, sorted_indices

    def reset_predicted_ligand_transforms(self, predicted_ligand_transforms_path, n_preds_to_use):
        self.dataset.reset_predicted_ligand_transforms(
            predicted_ligand_transforms_path, n_preds_to_use)
        self._form_batches(self.batch_limit)

    def _get_sorted_batches(self, lengths, sorted_indices, batch_limit):
        batch_indices = []
        cur_batch = []
        for real_ind, cur_len in zip(sorted_indices, lengths[sorted_indices]):
            if (len(cur_batch) + 1) * cur_len <= batch_limit:
                cur_batch.append(real_ind)
            else:
                batch_indices.append(cur_batch)
                cur_batch = [real_ind]

        batch_indices.append(cur_batch)
        return batch_indices

    def _form_batches(self, batch_limit):
        lengths, sorted_indices = self._init_sorted_indices()
        self.batch_indices = self._get_sorted_batches(
            lengths, sorted_indices, batch_limit)

    def __len__(self):
        return len(self.batch_indices)

    def __getitem__(self, idx):
        batch_complexes = []
        for i in self.batch_indices[idx]:
            complex = self.dataset.__get_nonrand_item__(i)
            complex = randomize_complex(complex=complex,
                                        std_protein_pos=self.dataset.std_protein_pos, std_lig_pos=self.dataset.std_lig_pos,
                                        ligand_mask_ratio=self.dataset.ligand_mask_ratio,
                                        protein_mask_ratio=self.dataset.protein_mask_ratio,
                                        esm_emb_noise_std=self.dataset.esm_emb_noise_std,
                                        tr_mean=self.dataset.tr_mean, tr_std=self.dataset.tr_std,
                                        use_pred_ligand_transforms=self.dataset.use_pred_ligand_transforms,
                                        use_predicted_tr_only=self.dataset.use_predicted_tr_only if hasattr(
                                            self.dataset, 'use_predicted_tr_only') else True,
                                        randomize_bond_neighbors=self.dataset.randomize_bond_neighbors,
                                        stage_num=self.dataset.stage_num)
            batch_complexes.append(complex)
        return self.data_collator(batch_complexes)


def apply_random_rotation_inplace(complex):
    aug_rot = R.random().as_matrix().astype(np.float32)

    complex.ligand.pos = complex.ligand.pos @ aug_rot.T
    if complex.ligand.pred_tr is not None:
        complex.ligand.pred_tr = complex.ligand.pred_tr @ aug_rot.T
    if complex.ligand.predicted_pos is not None:
        complex.ligand.predicted_pos = complex.ligand.predicted_pos @ aug_rot.T
    if complex.ligand.final_tr is not None:
        complex.ligand.final_tr = complex.ligand.final_tr @ aug_rot.T
    if complex.ligand.init_tr is not None:
        complex.ligand.init_tr = complex.ligand.init_tr @ aug_rot.T

    complex.protein.pos = complex.protein.pos @ aug_rot.T
    if complex.protein.all_atom_pos is not None:
        complex.protein.all_atom_pos = complex.protein.all_atom_pos @ aug_rot.T
    complex.original_augm_rot = aug_rot


def complex_collate_fn(batch: List[Complex]) -> ComplexBatch:
    """
    Collate function to pad sequences and output a ComplexBatch.

    Parameters:
    batch (List[Complex]): A list of Complex objects, where each Complex contains:
        - ligand (Ligand): The ligand object with attributes x and pos.
        - protein (Protein): The protein object with attributes x and pos.

    Returns:
    ComplexBatch: A batch object containing padded sequences for ligands and proteins.
    """

    # Extract components from the batch
    lig_xs = [torch.from_numpy(complex.ligand.x) for complex in batch]
    lig_positions = [torch.from_numpy(complex.ligand.pos) for complex in batch]
    lig_orig_positions = [torch.from_numpy(
        complex.ligand.orig_pos) for complex in batch]
    lig_orig_positions_before_augm = [torch.from_numpy(
        complex.ligand.orig_pos_before_augm) for complex in batch]
    try:
        lig_true_positions = [torch.from_numpy(
            complex.ligand.true_pos) for complex in batch]
    except Exception as e:
        lig_true_positions = None
    orig_mols = [complex.ligand.orig_mol for complex in batch]
    mask_rotate = [torch.from_numpy(complex.ligand.mask_rotate)
                   for complex in batch]
    protein_xs = [torch.from_numpy(complex.protein.x) if isinstance(
        complex.protein.x, np.ndarray) else complex.protein.x for complex in batch]
    protein_positions = [torch.from_numpy(
        complex.protein.pos) for complex in batch]
    protein_sequences = [torch.from_numpy(
        complex.protein.seq) for complex in batch]

    init_tr = torch.cat([torch.from_numpy(complex.ligand.init_tr)
                        for complex in batch])
    init_rot = torch.cat(
        [torch.from_numpy(complex.ligand.init_rot) for complex in batch])
    init_tor = torch.cat(
        [torch.from_numpy(complex.ligand.init_tor) for complex in batch])
    final_tr = torch.cat(
        [torch.from_numpy(complex.ligand.final_tr) for complex in batch])
    final_rot = torch.cat(
        [torch.from_numpy(complex.ligand.final_rot) for complex in batch])
    final_tor = torch.cat(
        [torch.from_numpy(complex.ligand.final_tor) for complex in batch])

    try:
        num_resamples = torch.cat(
            [torch.tensor([complex.ligand.num_resamples]) for complex in batch])
    except Exception as e:
        num_resamples = None

    all_atom_pos = [torch.from_numpy(
        complex.protein.all_atom_pos).float() for complex in batch]
    # convert numpy array with names to index array using protein_atom_names without loop
    unknown_atom_idx = protein_atom_names["unknown"]

    def get_atom_name(name): return protein_atom_names.get(
        name, unknown_atom_idx)
    vectorized_lookup = np.vectorize(get_atom_name, otypes=[np.int64])
    all_atom_names = [torch.from_numpy(vectorized_lookup(
        complex.protein.all_atom_names)) for complex in batch]
    all_atom_residue_ids = [
        torch.from_numpy(complex.protein.all_atom_residue_ids).long() for complex in batch]

    # max_n_residues = max(complex.protein.x.shape[0] for complex in batch)
    # max_n_atoms_in_residue = max(max(sum(residue_ids == idx) for idx in torch.unique(residue_ids, sorted=True)) for residue_ids in all_atom_residue_ids)
    # residue_all_atom_names = torch.zeros((len(batch), max_n_residues, max_n_atoms_in_residue), dtype=torch.long)
    # residue_all_atom_names.fill_(protein_atom_names["padding"])
    # residue_all_atom_pos = torch.zeros((len(batch), max_n_residues, max_n_atoms_in_residue, 3), dtype=torch.float32)
    # # Group atoms by residue for each complex
    # for i, (atom_names, atom_pos, residue_ids) in enumerate(zip(all_atom_names, all_atom_pos, all_atom_residue_ids)):
    #     unique_residue_ids = torch.unique(residue_ids, sorted=True)
    #     for j, residue_id in enumerate(unique_residue_ids):
    #         mask = residue_ids == residue_id
    #         residue_all_atom_names[i, j, :len(atom_names[mask])] = atom_names[mask]
    #         residue_all_atom_pos[i, j, :len(atom_pos[mask]), :] = atom_pos[mask]

    # except Exception as e:
    #     all_atom_pos = None
    #     all_atom_names = None
    #     all_atom_residue_ids = None

    num_rotatable_bonds = torch.tensor(
        [len(complex.ligand.final_tor) for complex in batch], dtype=torch.long)
    t = torch.cat([complex.ligand.t for complex in batch])
    rmsd = torch.cat([complex.ligand.rmsd for complex in batch])

    try:
        fast_filters = torch.cat(
            [complex.ligand.fast_filters for complex in batch])
    except Exception as e:
        fast_filters = None
    stage_num = torch.cat(
        [complex.ligand.stage_num if complex.ligand.stage_num is not None else torch.tensor([0]) for complex in batch])
    names = [complex.name for complex in batch]
    orig_augm_rot = torch.cat(
        [torch.from_numpy(complex.original_augm_rot[None, :]) for complex in batch])
    full_protein_center = torch.cat(
        [torch.from_numpy(complex.protein.full_protein_center) for complex in batch])

    try:
        pred_tr = torch.cat(
            [torch.from_numpy(complex.ligand.pred_tr) for complex in batch])
    except Exception as e:
        pred_tr = None

    # Pad ligand sequences
    lig_x_padded = pad_sequence(lig_xs, batch_first=True, padding_value=0.0)
    lig_pos_padded = pad_sequence(
        lig_positions, batch_first=True, padding_value=0.0)
    lig_orig_pos_padded = pad_sequence(
        lig_orig_positions, batch_first=True, padding_value=0.0)
    lig_orig_pos_before_augm_padded = pad_sequence(lig_orig_positions_before_augm,
                                                    batch_first=True, padding_value=0.0)

    try:
        lig_true_pos_padded = pad_sequence(
            lig_true_positions, batch_first=True, padding_value=0.0)
    except Exception as e:
        lig_true_pos_padded = None

    # Pad protein sequences
    protein_x_padded = pad_sequence(
        protein_xs, batch_first=True, padding_value=0.0)
    protein_pos_padded = pad_sequence(
        protein_positions, batch_first=True, padding_value=0.0)
    protein_seq_padded = pad_sequence(
        protein_sequences, batch_first=True, padding_value=0.0)
    protein_all_atom_pos_padded = pad_sequence(
        all_atom_pos, batch_first=True, padding_value=0.0)
    num_max_res = protein_x_padded.shape[1]
    protein_all_atom_names_padded = pad_sequence(
        all_atom_names, batch_first=True, padding_value=protein_atom_names["padding"])
    protein_all_atom_residue_ids_padded = pad_sequence(
        all_atom_residue_ids, batch_first=True, padding_value=-1)

    mask_all_atom_residue = torch.arange(
        num_max_res)[None, :, None] == protein_all_atom_residue_ids_padded[:, None, :]

    rotatable_bonds_list = []
    for complex in batch:
        if len(complex.ligand.rotatable_bonds) > 0:
            rotatable_bonds_list.append(
                torch.from_numpy(complex.ligand.rotatable_bonds))
    if len(rotatable_bonds_list) > 0:
        rotatable_bonds = torch.concat(rotatable_bonds_list)
    else:
        rotatable_bonds = torch.empty((0, 2))

    bond_periods_list = [torch.from_numpy(complex.ligand.bond_periods) for complex in batch
                         if complex.ligand.bond_periods is not None]
    if len(bond_periods_list) > 0:
        bond_periods = torch.cat(bond_periods_list)
    else:
        bond_periods = torch.empty((0,))

    # Extract and pad rotatable and non-rotatable bonds
    rotatable_bonds_ext = [
        complex.ligand.rotatable_bonds_ext for complex in batch]

    # Pad bonds
    def pad_bonds(bonds_list, max_num_bonds, max_num_atoms, is_rotatable_bonds=False):
        bond_keys = ['bond_type', 'is_conjugated', 'is_in_ring', 'is_aromatic', 'is_rotatable',
                     'start', 'end', 'neighbor_of_start', 'neighbor_of_end', 'length',
                     'bond_periods', 'angles', 'angle_histograms']
        padded_bonds = {key: [] for key in bond_keys}
        for bonds in bonds_list:
            for key in padded_bonds.keys():
                value = getattr(bonds, key)
                if value is not None:
                    # This patch looks ugly, but it is to fix the case where the number of bonds is 0, and start/end are empty
                    # Without this patch, np.pad returns float instead of int
                    if len(value) == 0 and (key == 'start' or key == 'end' or key == 'neighbor_of_start' or
                                            key == 'neighbor_of_end'):
                        padded_value = np.zeros(max_num_bonds, dtype=np.int32)
                    else:
                        if key == 'length' or key == 'bond_periods' or key == 'angles':
                            value = value.astype(np.float32)
                        constant_value = 2 * np.pi if key == 'bond_periods' else 0
                        if key == 'angle_histograms':
                            padded_value = np.pad(value, ((0, max_num_bonds - len(value)), (0, 0)), 'constant',
                                                  constant_values=constant_value)
                        else:
                            padded_value = np.pad(value, (0, max_num_bonds - len(value)), 'constant',
                                                  constant_values=constant_value)
                    padded_bonds[key].append(torch.from_numpy(
                        padded_value))  # Convert to tensor
                else:
                    padded_bonds[key] = None

        res = {}
        for key, value in padded_bonds.items():
            if value is not None:
                res[key] = torch.stack(value)
            else:
                res[key] = None

        res['num_rotatable_bonds'] = torch.tensor(
            [len(bonds.start) for bonds in bonds_list])

        # mask_rotate is a special case because it is a 2D tensor
        if is_rotatable_bonds:
            padded_bonds['mask_rotate'] = []

            for bonds in bonds_list:
                if len(bonds.start):
                    value = bonds.mask_rotate
                    padded_value = np.pad(value, ((
                        0, max_num_bonds - value.shape[0]), (0, max_num_atoms - value.shape[1])), 'constant', constant_values=0)
                    padded_bonds['mask_rotate'].append(
                        torch.from_numpy(padded_value))
                else:
                    padded_bonds['mask_rotate'].append(torch.zeros(
                        max_num_bonds, max_num_atoms, dtype=torch.bool))

            res['mask_rotate'] = torch.stack(padded_bonds['mask_rotate'])

        max_num_bonds = max([bonds.start.shape[0] for bonds in bonds_list])

        res['is_padded_mask'] = torch.ones(
            len(bonds_list), max_num_bonds, dtype=torch.bool)
        for idx, bonds in enumerate(bonds_list):
            res['is_padded_mask'][idx, :bonds.start.shape[0]] = False

        return res

    max_rotatable_bonds_ext = max(len(bond.start)
                                  for bond in rotatable_bonds_ext)
    max_num_atoms = lig_pos_padded.shape[1]

    padded_rotatable_bonds_ext = pad_bonds(
        rotatable_bonds_ext, max_rotatable_bonds_ext, max_num_atoms, is_rotatable_bonds=True)

    # Create BondsBatch objects
    rotatable_bonds_batch_ext = BondsBatch(**padded_rotatable_bonds_ext)

    # Create tensors indicating the number of rotatable and non-rotatable bonds in each ligand
    num_rotatable_bonds_ext = torch.tensor(
        [len(bond.start) for bond in rotatable_bonds_ext], dtype=torch.long)

    # Fill in is_padded_mask_...
    # We first create a batch_size × max_seq_len matrices, then flatten them
    batch_size, max_lig_seq_len = lig_pos_padded.shape[0], lig_pos_padded.shape[1]
    max_protein_seq_len = protein_pos_padded.shape[1]
    is_padded_mask_ligand = torch.ones(
        batch_size, max_lig_seq_len, dtype=torch.bool)
    is_padded_mask_protein = torch.ones(
        batch_size, max_protein_seq_len, dtype=torch.bool)

    for idx, complex in enumerate(batch):
        is_padded_mask_ligand[idx, :complex.ligand.pos.shape[0]] = False
        is_padded_mask_protein[idx, :complex.protein.pos.shape[0]] = False

    # Compute num_atoms and tor_ptr using numpy
    num_atoms = torch.tensor([x.shape[0]
                             for x in lig_positions], dtype=torch.long)
    tor_ptr = [
        0] + list(np.cumsum([complex.ligand.rotatable_bonds.shape[0] for complex in batch]))

    # Create a padded tor tensor for init_tor and final_tor
    tor_padded_init_ext = pad_sequence(
        [torch.from_numpy(complex.ligand.init_tor) for complex in batch],
        batch_first=True,
        padding_value=0.0
    )
    tor_padded_final_ext = pad_sequence(
        [torch.from_numpy(complex.ligand.final_tor) for complex in batch],
        batch_first=True,
        padding_value=0.0
    )

    # Create ComplexBatch
    batch = ComplexBatch(
        ligand=LigandBatch(
            x=lig_x_padded,
            pos=lig_pos_padded,
            orig_pos=lig_orig_pos_padded,
            orig_pos_before_augm=lig_orig_pos_before_augm_padded,
            true_pos=lig_true_pos_padded,
            random_pos=lig_pos_padded.clone(),
            mask_rotate=mask_rotate,
            init_tr=init_tr,
            init_rot=init_rot,
            init_tor=init_tor,
            final_tr=final_tr,
            final_rot=final_rot,
            final_tor=final_tor,
            pred_tr=pred_tr,
            num_atoms=num_atoms,
            bond_periods=bond_periods,
            tor_ptr=tor_ptr,
            rotatable_bonds=rotatable_bonds,
            num_rotatable_bonds=num_rotatable_bonds,
            t=t,
            rmsd=rmsd,
            stage_num=stage_num,
            is_padded_mask=is_padded_mask_ligand,
            orig_mols=orig_mols,

            rotatable_bonds_ext=rotatable_bonds_batch_ext,
            num_rotatable_bonds_ext=num_rotatable_bonds_ext,
            init_tor_ext=tor_padded_init_ext,
            final_tor_ext=tor_padded_final_ext,
        ),
        protein=ProteinBatch(x=protein_x_padded, pos=protein_pos_padded,
                             seq=protein_seq_padded,
                             is_padded_mask=is_padded_mask_protein,
                             full_protein_center=full_protein_center,
                             all_atom_pos=protein_all_atom_pos_padded,
                             all_atom_names=protein_all_atom_names_padded,
                             all_atom_residue_ids=protein_all_atom_residue_ids_padded,
                             mask_all_atom_residue=mask_all_atom_residue),
        names=names,
        original_augm_rot=orig_augm_rot,
    )
    batch.ligand.fast_filters = fast_filters
    batch.ligand.num_resamples = num_resamples

    return {"batch": batch, "labels": batch.ligand.rmsd}

def dummy_ranking_collate_fn(batch: List[ComplexBatch]) -> ComplexBatch:
    return batch[0]
