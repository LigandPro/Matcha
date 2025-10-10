import torch
import os
import copy
import re
import pickle
import numpy as np
from deli import load
from collections import defaultdict
from typing import List
from torch.utils.data import Dataset
from rdkit.Chem import AllChem, RemoveAllHs
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from torch.nn.utils.rnn import pad_sequence


from matcha.dataset.complex_dataclasses import Ligand, Protein, Complex, LigandBatch, ProteinBatch, ComplexBatch
from matcha.utils.preprocessing import (parse_receptor, read_mols,
                                        extract_receptor_structure_prody, lig_atom_featurizer,
                                        read_molecule, read_sdf_with_multiple_confs, save_multiple_confs)
from matcha.utils.bond_processing import get_rotatable_bonds_and_mask_rotate, split_molecule
from matcha.utils.transforms import apply_tor_changes_to_pos


def get_ligand_without_randomization(mol_, protein_center=None):
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

    # Ensure the molecule has 3D coordinates
    if not mol_maybe_noh.GetNumConformers():
        AllChem.EmbedMolecule(mol_maybe_noh, randomSeed=13)
        if not mol_maybe_noh.GetNumConformers():
            raise ValueError(
                "Embedding of the molecule failed, unable to generate conformer.")

    rotatable_bonds, mask_rotate_before_fixing, mask_rotate_after_fixing, bond_periods = get_rotatable_bonds_and_mask_rotate(
        mol_maybe_noh)

    ligands = {}
    conf_id = 0
    ligand = Ligand()
    ligand.pos = mol_maybe_noh.GetConformer(
        conf_id).GetPositions().astype(np.float32) - protein_center

    ligand.orig_mol = mol_maybe_noh  # original mol
    # features are conformer-invariant
    ligand.x = lig_atom_featurizer(mol_maybe_noh)
    ligand.final_tr = ligand.pos.mean(0).astype(np.float32).reshape(1, 3)

    # Fill ligand properties
    if len(rotatable_bonds) > 0:
        ligand.rotatable_bonds = rotatable_bonds
        ligand.mask_rotate = mask_rotate_after_fixing
        ligand.mask_rotate_before_fixing = mask_rotate_before_fixing
        ligand.bond_periods = bond_periods
        ligand.init_tor = np.zeros(
            ligand.rotatable_bonds.shape[0], dtype=np.float32)
    else:
        ligand.rotatable_bonds = np.array([], dtype=np.int32)
        ligand.mask_rotate = np.array([], dtype=np.int32)
        ligand.mask_rotate_before_fixing = np.array([], dtype=np.int32)
        ligand.init_tor = np.array([], dtype=np.float32)

    ligand.t = None
    ligands[f'conf{conf_id}'] = ligand
    return ligands


def randomize_ligand_with_preds(ligand: Ligand, tr_mean: float = 0., tr_std: float = 5., with_preds: bool = False):
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
        tr = np.random.normal(tr_mean, tr_std, 3).astype(
            np.float32).reshape(1, 3)

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
    ligand.pos = np.copy(pos)

    # Time is randomized from Uniform[0, 1]:
    ligand.t = torch.rand(1)
    if ligand.rmsd is None:
        ligand.rmsd = torch.zeros(1)


def set_ligand_data_from_preds(ligand: Ligand):
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
    pred_pos = np.copy(ligand.predicted_pos)

    ligand.init_tr = ligand.pred_tr.reshape(1, 3)
    ligand.pos = pred_pos

    # Time is randomized from Uniform[0, 1]:
    ligand.t = torch.rand(1)
    if ligand.rmsd is None:
        ligand.rmsd = torch.zeros(1)


def randomize_complex(complex: Complex, tr_mean: float, tr_std: float,
                      use_pred_ligand_transforms: bool = False,
                      use_predicted_tr_only: bool = True):
    # 1. Rotate complex
    apply_random_rotation_inplace(complex)

    # 5. Compute ligand gt values
    complex.set_ground_truth_values()

    # 7. Randomize ligand for NN input
    if use_pred_ligand_transforms:
        if use_predicted_tr_only:
            randomize_ligand_with_preds(complex.ligand, with_preds=True)
        else:
            set_ligand_data_from_preds(complex.ligand)
    else:
        randomize_ligand_with_preds(
            complex.ligand, tr_mean=tr_mean, tr_std=tr_std, with_preds=False)
    return complex


class PDBBind(Dataset):
    def __init__(self, data_dir, split_path, esm_embeddings_path, sequences_path,
                 tr_std=1., tr_mean=0., cache_path='data/cache',
                 predicted_ligand_transforms_path=None, dataset_type='pdbbind',
                 add_all_atom_pos=False, use_predicted_tr_only=True,
                 data_dir_conf=None, n_preds_to_use=1, use_all_chains=False,
                 min_lig_size=0):
        self.data_dir = data_dir
        self.n_preds_to_use = n_preds_to_use
        self.n_confs_to_use = min(10, self.n_preds_to_use)
        self.data_dir_conf = data_dir_conf
        self.esm_embeddings_path = esm_embeddings_path
        self.sequences_path = sequences_path
        self.cache_path = cache_path
        self.dataset_type = dataset_type
        self.tr_std = tr_std
        self.tr_mean = tr_mean
        self.use_pred_ligand_transforms = predicted_ligand_transforms_path is not None
        self.add_all_atom_pos = add_all_atom_pos
        self.use_predicted_tr_only = use_predicted_tr_only
        self.use_all_chains = use_all_chains
        self.min_lig_size = min_lig_size
        self.split_path = split_path
        self.full_cache_path = self._get_cache_folder_path()

        # TODO keep 0 for padding
        aa_list = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
                   'R', 'S', 'T', 'V', 'W', 'Y']
        self.aa_mapping = {aa: i for i, aa in enumerate(aa_list)}

        # loads data to self.complexes list:
        if os.path.exists(self.full_cache_path):
            self._load_from_cache()
        else:
            os.makedirs(self.full_cache_path, exist_ok=True)
            self._preprocess_and_save_to_cache()

        if self.dataset_type.endswith('_conf'):
            self._set_all_conformer_proteins()

        # save orig_pos_before_augm for each ligand
        complexes = []
        for complex in self.complexes:
            complex.ligand.orig_pos_before_augm = np.copy(complex.ligand.pos)
            complexes.append(complex)
        self.complexes = complexes

        if self.dataset_type.endswith('_conf'):
            self._explode_ligand_conformers(n_preds_to_use)
        else:
            self.complexes = self.complexes * n_preds_to_use

        if self.use_pred_ligand_transforms:
            self._set_predicted_ligand_transforms(
                predicted_ligand_transforms_path, n_preds_to_use)

    def reset_predicted_ligand_transforms(self, predicted_ligand_transforms_path, n_preds_to_use):
        self.use_pred_ligand_transforms = True
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
                name2protein[complex.name.split('_conf')[0]] = complex.protein

        new_complexes = []
        for complex in self.complexes:
            if not complex.name.endswith('_conf0'):
                complex.protein = name2protein[complex.name.split('_conf')[0]]
            new_complexes.append(complex)
        self.complexes = new_complexes

    def _set_predicted_ligand_transforms(self, predicted_ligand_transforms_path, n_preds_to_use):
        self.predicted_ligand_transforms = np.load(
            predicted_ligand_transforms_path, allow_pickle=True)[0]
        self.n_repeats = 1
        n_preds_to_use = min(n_preds_to_use, len(
            self.predicted_ligand_transforms[self.complexes[0].name]))
        self.complexes = [
            complex for complex in self.complexes if complex.name in self.predicted_ligand_transforms]

        # initialize extended complexes
        extended_complexes = []
        for complex in tqdm(self.complexes, desc='Setting predicted ligand transforms...'):
            for i in range(n_preds_to_use):
                extended_complex = copy.deepcopy(complex)
                pred_data = self.predicted_ligand_transforms[complex.name][i]
                extended_complex.ligand.pred_tr = pred_data['tr_pred_init'] + \
                    pred_data['full_protein_center'] - \
                    extended_complex.protein.full_protein_center

                pred_pos = pred_data['transformed_orig'] + pred_data['full_protein_center'] - \
                    extended_complex.protein.full_protein_center
                if not self.use_predicted_tr_only:
                    extended_complex.ligand.predicted_pos = pred_pos

                extended_complexes.append(extended_complex)
        self.complexes = extended_complexes

    def __len__(self):
        return len(self.complexes)

    def __get_nonrand_item__(self, idx):
        complex = copy.deepcopy(self.complexes[idx])
        return complex

    def __getitem__(self, idx):
        complex = self.__get_nonrand_item__(idx)
        complex = randomize_complex(complex=complex,
                                    tr_mean=self.tr_mean, tr_std=self.tr_std,
                                    use_pred_ligand_transforms=self.use_pred_ligand_transforms,
                                    use_predicted_tr_only=self.use_predicted_tr_only)
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
        print('LLM embeddings are loaded.', flush=True)
        return lm_embeddings_chains_all, sequence_chains_all, tokenized_sequence_chains_all

    def _process_complex(self, complex_names, sequences_to_embeddings):
        try:
            return self._get_complex(complex_names, sequences_to_embeddings)
        except Exception as e:
            print(f"Error processing {complex_names}: {e}")
            return None

    def _preprocess_and_save_to_cache(self):
        if self.split_path is not None and os.path.exists(self.split_path):
            print(
                f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')
            # Get names of complexes:
            with open(self.split_path, 'r') as file:
                lines = file.readlines()
                complex_names_all = [line.rstrip() for line in lines]
        else:
            complex_names_all = [name for name in os.listdir(
                self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        print(f'Loading {len(complex_names_all)} complexes.')

        # Load embeddings:
        lm_embeddings_chains_all, sequence_chains_all, tokenized_sequence_chains_all = self._load_embeddings(self.esm_embeddings_path,
                                                                                                             self.sequences_path, complex_names_all)

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
                if processed_complexes is not None:
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
            print(f"Data successfully saved to {filepath}!")
        except IOError as e:
            print(f"Error saving data to {filepath}: {e}!")

    def _load_from_cache(self):
        filepath = os.path.join(self.full_cache_path, 'complexes.pkl')
        try:
            with open(filepath, 'rb') as f:
                self.complexes = pickle.load(f)
            print(f"Data successfully loaded from {filepath}!")
        except IOError as e:
            print(f"Error loading data from {filepath}: {e}!")

    def _get_complex(self, complex_names, sequences_to_embeddings):
        try:
            rec_model = parse_receptor(
                complex_names[0], self.data_dir, self.dataset_type)
        except Exception as e:
            print(f'Skipping {complex_names[0]} because of the error:')
            print(e)
            return [], []

        complexes = []
        failed_indices = []
        for name in complex_names:
            print('complex', name)

            if self.dataset_type == 'pdbbind_conf':
                orig_ligs = read_mols(self.data_dir, name, remove_hs=False)
            elif self.dataset_type == 'posebusters_conf' or self.dataset_type == 'astex_conf' or self.dataset_type == 'any_conf':
                orig_ligs = [read_molecule(os.path.join(
                    self.data_dir, name, f'{name}_ligand.sdf'), remove_hs=False, sanitize=True)]
            elif self.dataset_type == 'dockgen_full_conf':
                orig_ligs = [read_molecule(os.path.join(
                    self.data_dir, name, f'{name}_ligand.pdb'), remove_hs=False, sanitize=True)]
            else:
                raise ValueError(f'Unknown dataset type: {self.dataset_type}')

            orig_ligs = [split_molecule(
                lig_mol, min_lig_size=self.min_lig_size) for lig_mol in orig_ligs]
            orig_ligs = [
                lig_mol for lig_mol_list in orig_ligs for lig_mol in lig_mol_list if lig_mol is not None]
            assert len(
                orig_ligs) == 1, f'Expected 1 ligand, got {len(orig_ligs)}'

            fname_with_confs = os.path.join(
                self.data_dir_conf, f'{name}_conf.sdf')
            save_multiple_confs(
                orig_ligs[0], fname_with_confs, num_conformers=self.n_confs_to_use)
            ligs = [read_sdf_with_multiple_confs(
                fname_with_confs, remove_hs=False, sanitize=True)]

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
                    c_alpha_coords_list, lm_embeddings_list, sequences_list, chain_lengths, full_coords, full_atom_names = extract_receptor_structure_prody(
                        copy.deepcopy(rec_model), sequences_to_embeddings)

                    # positions are positions of C-alpha, other positions are not used
                    if not self.add_all_atom_pos:
                        full_coords = None
                        full_atom_names = None
                    protein = Protein(x=lm_embeddings_list, pos=c_alpha_coords_list,
                                      seq=sequences_list, all_atom_pos=full_coords, all_atom_names=full_atom_names)
                    protein_center = protein.pos.mean(axis=0).reshape(1, 3)
                    protein.pos -= protein_center
                    protein.full_protein_center = protein_center
                    protein.chain_lengths = chain_lengths
                    # Process ligand:
                    parse_rotbonds = True
                    for conf_id, lig_mol in enumerate(lig_mol_list):
                        ligand = get_ligand_without_randomization(
                            lig_mol, protein_center)['conf0']
                        if parse_rotbonds:
                            ligand_with_bonds = copy.deepcopy(ligand)
                            cur_ligand = ligand
                        else:
                            cur_ligand = copy.deepcopy(ligand_with_bonds)
                            cur_ligand.pos = copy.deepcopy(ligand.pos)
                            cur_ligand.x = copy.deepcopy(ligand.x)
                            cur_ligand.final_tr = copy.deepcopy(
                                ligand.final_tr)
                            cur_ligand.orig_mol = copy.deepcopy(
                                ligand.orig_mol)

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
                        complexes.append(complex)

                except Exception as e:
                    print(f'Skipping {name} because of the error:')
                    print(e)
                    failed_indices.append(lig_idx)
                    continue

        return complexes


class PDBBindWithSortedBatching(Dataset):
    def __init__(self, dataset, batch_limit, data_collator):
        self.dataset = dataset
        self.batch_limit = batch_limit
        self.data_collator = data_collator
        self._form_batches(batch_limit)

    def reset_predicted_ligand_transforms(self, predicted_ligand_transforms_path, n_preds_to_use):
        self.dataset.reset_predicted_ligand_transforms(
            predicted_ligand_transforms_path, n_preds_to_use)
        self._form_batches(self.batch_limit)

    def _init_sorted_indices(self):
        protein_lengths = np.array(
            [complex.protein.pos.shape[0] for complex in self.dataset.complexes])
        ligand_lengths = np.array([complex.ligand.pos.shape[0]
                                  for complex in self.dataset.complexes])
        sorted_indices = np.lexsort((ligand_lengths, protein_lengths))
        return protein_lengths + ligand_lengths, sorted_indices

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
                                        tr_mean=self.dataset.tr_mean, tr_std=self.dataset.tr_std,
                                        use_pred_ligand_transforms=self.dataset.use_pred_ligand_transforms,
                                        use_predicted_tr_only=self.dataset.use_predicted_tr_only if hasattr(self.dataset, 'use_predicted_tr_only') else True)
            batch_complexes.append(complex)
        return self.data_collator(batch_complexes)


def apply_random_rotation_inplace(complex):
    aug_rot = R.random().as_matrix().astype(np.float32)

    complex.ligand.pos = complex.ligand.pos @ aug_rot.T
    if complex.ligand.pred_tr is not None:
        complex.ligand.pred_tr = complex.ligand.pred_tr @ aug_rot.T
    if complex.ligand.predicted_pos is not None:
        complex.ligand.predicted_pos = complex.ligand.predicted_pos @ aug_rot.T
    complex.protein.pos = complex.protein.pos @ aug_rot.T
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
    init_tor = torch.cat(
        [torch.from_numpy(complex.ligand.init_tor) for complex in batch])
    final_tr = torch.cat(
        [torch.from_numpy(complex.ligand.final_tr) for complex in batch])

    try:
        all_atom_pos = [torch.from_numpy(
            complex.protein.all_atom_pos) for complex in batch]
        all_atom_names = [complex.protein.all_atom_names for complex in batch]
    except:
        all_atom_pos = None
        all_atom_names = None

    num_rotatable_bonds = torch.tensor(
        [len(complex.ligand.rotatable_bonds) for complex in batch], dtype=torch.long)
    t = torch.cat([complex.ligand.t for complex in batch])
    rmsd = torch.cat([complex.ligand.rmsd for complex in batch])
    names = [complex.name for complex in batch]
    orig_augm_rot = torch.cat(
        [torch.from_numpy(complex.original_augm_rot[None, :]) for complex in batch])
    try:
        full_protein_center = torch.cat(
            [torch.from_numpy(complex.protein.full_protein_center) for complex in batch])
    except:
        full_protein_center = None

    try:
        pred_tr = torch.cat(
            [torch.from_numpy(complex.ligand.pred_tr) for complex in batch])
    except:
        pred_tr = None

    # Pad ligand sequences
    lig_x_padded = pad_sequence(lig_xs, batch_first=True, padding_value=0.0)
    lig_pos_padded = pad_sequence(
        lig_positions, batch_first=True, padding_value=0.0)
    try:
        lig_orig_pos_padded = pad_sequence(
            lig_orig_positions, batch_first=True, padding_value=0.0)
    except:
        lig_orig_pos_padded = None
        print('Warning: orig_pos are not defined!')

    try:
        lig_orig_pos_before_augm_padded = pad_sequence(lig_orig_positions_before_augm,
                                                       batch_first=True, padding_value=0.0)
    except:
        lig_orig_pos_before_augm_padded = None
        print('Warning: orig_pos_before_augm are not defined!')

    # Pad protein sequences
    protein_x_padded = pad_sequence(
        protein_xs, batch_first=True, padding_value=0.0)
    protein_pos_padded = pad_sequence(
        protein_positions, batch_first=True, padding_value=0.0)
    protein_seq_padded = pad_sequence(
        protein_sequences, batch_first=True, padding_value=0.0)

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

    # Create ComplexBatch
    batch = ComplexBatch(
        ligand=LigandBatch(
            x=lig_x_padded,
            pos=lig_pos_padded,
            orig_pos=lig_orig_pos_padded,
            orig_pos_before_augm=lig_orig_pos_before_augm_padded,
            random_pos=lig_pos_padded.clone(),
            mask_rotate=mask_rotate,
            init_tr=init_tr,
            init_tor=init_tor,
            final_tr=final_tr,
            pred_tr=pred_tr,
            num_atoms=num_atoms,
            bond_periods=bond_periods,
            tor_ptr=tor_ptr,
            rotatable_bonds=rotatable_bonds,
            num_rotatable_bonds=num_rotatable_bonds,
            t=t,
            rmsd=rmsd,
            is_padded_mask=is_padded_mask_ligand,
            orig_mols=orig_mols,
        ),
        protein=ProteinBatch(x=protein_x_padded, pos=protein_pos_padded,
                             seq=protein_seq_padded,
                             is_padded_mask=is_padded_mask_protein,
                             full_protein_center=full_protein_center,
                             all_atom_pos=all_atom_pos,
                             all_atom_names=all_atom_names),
        names=names,
        original_augm_rot=orig_augm_rot,
    )

    return {"batch": batch, "labels": batch.ligand.rmsd}
