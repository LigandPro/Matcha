import os
import numpy as np
import copy
from tqdm import tqdm
from collections import defaultdict
from yaml import safe_load
from posebusters import PoseBusters
import posebusters
from matcha.utils.spyrmsd import TimeoutException, time_limit
from matcha.utils.paths import get_protein_path


def get_posebusters_tests_updated(predictions, dataset_name, dataset_data_dir, posebusters_config='redock'):
    docking_only_tests = ['mol_pred_loaded', 'mol_cond_loaded', 'sanitization', 'inchi_convertible',
                          'all_atoms_connected', 'bond_lengths', 'bond_angles', 'internal_steric_clash',
                          'aromatic_ring_flatness', 'non-aromatic_ring_non-flatness', 'double_bond_flatness',
                          'internal_energy', 'protein-ligand_maximum_distance', 'minimum_distance_to_protein',
                          'minimum_distance_to_organic_cofactors', 'minimum_distance_to_inorganic_cofactors',
                          'minimum_distance_to_waters', 'volume_overlap_with_protein',
                          'volume_overlap_with_organic_cofactors', 'volume_overlap_with_inorganic_cofactors',
                          'volume_overlap_with_waters']
    if posebusters_config == 'redock':
        redock_extra_tests = ['double_bond_stereochemistry', 'mol_true_loaded', 'molecular_bonds',
                              'molecular_formula', 'rmsd_≤_2å', 'tetrahedral_chirality']
    else:
        redock_extra_tests = []

    buster = PoseBusters(config=posebusters_config,
                         max_workers=0, chunk_size=None)

    config_path = os.path.join(os.path.dirname(
        posebusters.__file__), 'config', f'{posebusters_config}.yml')
    with open(config_path, encoding="utf-8") as config_file:
        no_energy_config = safe_load(config_file)
    no_energy_config['modules'] = no_energy_config['modules'][:9] + \
        no_energy_config['modules'][10:]
    if posebusters_config == 'redock':
        no_energy_config['modules'] = no_energy_config['modules'][:-1]
    no_energy_buster = PoseBusters(
        config=no_energy_config, max_workers=0, chunk_size=None)

    new_predictions = defaultdict(list)

    for uid, uid_data in tqdm(predictions.items(), desc='Running PoseBusters'):
        new_uid_data = {}
        for field in set(uid_data.keys()) - set(['sample_metrics']):
            new_uid_data[field] = copy.deepcopy(uid_data[field])
        new_uid_data['sample_metrics'] = []

        orig_mol = uid_data['orig_mol']

        try:
            init_samples = copy.deepcopy(uid_data['sample_metrics'])
            if posebusters_config == 'redock':
                true_positions = uid_data['true_pos']
                true_mol = copy.deepcopy(orig_mol)
                true_mol.GetConformer().SetPositions(true_positions.astype(np.float64))
            else:
                true_mol = None

            protein_path = get_protein_path(
                uid, dataset_name, dataset_data_dir)

            pred_mols = []
            new_sample_metrics = []
            samples = []
            for i, sample in enumerate(init_samples):
                pred_positions = sample['pred_pos']
                mol = copy.deepcopy(orig_mol)
                mol.GetConformer().SetPositions(pred_positions.astype(np.float64))
                mol.SetProp("_Name", f'{uid}_{i}')
                pred_mols.append(mol)
                samples.append(copy.deepcopy(sample))

            if len(pred_mols) == 0:
                new_uid_data['sample_metrics'] = init_samples
                new_predictions[uid] = new_uid_data
                continue

            try:
                with time_limit(100000):
                    results = buster.bust(
                        mol_pred=pred_mols,
                        mol_true=true_mol,
                        mol_cond=protein_path,
                        full_report=True,
                    )
                    posebusters_computation_failed = False
            except TimeoutException:
                print(f'PoseBusters computation failed for {uid}')
                results = no_energy_buster.bust(
                    mol_pred=pred_mols,
                    mol_true=true_mol,
                    mol_cond=protein_path,
                )
                results['rmsd_≤_2å'] = False
                results['internal_energy'] = False
                posebusters_computation_failed = True

            results = results.reset_index()
            results['conf_idx'] = results['molecule'].apply(
                lambda x: int(x.split('_')[-1]))
            results = results.sort_values('conf_idx', ascending=True)

            posebusters_filters = results[docking_only_tests +
                                          redock_extra_tests].values
            docking_filters_passed_count = results[docking_only_tests].sum(
                1).values
            if posebusters_config == 'redock':
                redock_passed_extra_tests = results[redock_extra_tests].sum(
                    1).values
            else:
                redock_passed_extra_tests = -np.ones(len(samples))
            all_tests_passed = docking_filters_passed_count + redock_passed_extra_tests

            for i, sample in enumerate(samples):
                new_sample = copy.deepcopy(sample)
                new_sample['posebusters_filters'] = posebusters_filters[i]
                new_sample['posebusters_filters_passed_count'] = docking_filters_passed_count[i]
                new_sample['all_posebusters_filters_passed_count'] = all_tests_passed[i]
                new_sample['posebusters_computation_failed'] = posebusters_computation_failed
                new_sample_metrics.append(new_sample)

        except Exception as e:
            print(f'PoseBusters computation failed for {uid}')
            print(e)

            new_sample_metrics = []
            for i, sample in enumerate(samples):
                new_sample = copy.deepcopy(sample)
                new_sample['posebusters_filters'] = np.zeros(
                    len(docking_only_tests + redock_extra_tests)).astype(bool)
                new_sample['posebusters_filters_passed_count'] = 0
                new_sample['all_posebusters_filters_passed_count'] = 0
                new_sample['posebusters_computation_failed'] = True
                new_sample_metrics.append(new_sample)

        new_uid_data['sample_metrics'] = new_sample_metrics
        new_predictions[uid] = new_uid_data

    return new_predictions
