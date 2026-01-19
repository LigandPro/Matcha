import copy
from tqdm import tqdm
import numpy as np
import pandas as pd

from matcha.utils.spyrmsd import get_symmetry_rmsd_with_isomorphisms, TimeoutException


def get_best_results_by_score(all_results, score_name):
    filtered_results = {}

    for uid in all_results.keys():
        metrics = all_results[uid]
        if score_name == 'random':
            best_index = 0
        else:
            scores = np.array([metr[score_name]
                               for metr in metrics['sample_metrics']])
            best_index = np.argmin(scores)

        filtered_results[uid] = metrics['sample_metrics'][best_index]
    return filtered_results


def filter_results_by_posebusters(full_results, use_separate_samples=True):
    for uid in full_results.keys():
        if use_separate_samples:
            samples = full_results[uid]['sample_metrics']
        else:
            samples = full_results[uid]

        pb_filters_name = 'posebusters_filters_passed_count'

        scores = np.array([sample[pb_filters_name] for sample in samples])
        best_score = max(scores)
        filtered_samples = [
            sample for sample in samples if sample[pb_filters_name] == best_score]
        if use_separate_samples:
            full_results[uid]['sample_metrics'] = filtered_samples
        else:
            full_results[uid] = filtered_samples
    return full_results


def filter_results_by_fast(full_results, use_separate_samples=True):
    for uid in full_results.keys():
        if use_separate_samples:
            samples = full_results[uid]['sample_metrics']
        else:
            samples = full_results[uid]

        try:
            scores = np.array(
                [sample['posebusters_filters_passed_count_fast'] for sample in samples])
            best_score = max(scores)
            filtered_samples = [
                sample for sample in samples if sample['posebusters_filters_passed_count_fast'] == best_score]

        except KeyError as e:
            filtered_samples = samples

        if use_separate_samples:
            full_results[uid]['sample_metrics'] = filtered_samples
        else:
            full_results[uid] = filtered_samples
    return full_results


def filter_empty_results_and_keep_necessary_ids(full_results, use_separate_samples=True, ids_to_keep=None):
    if ids_to_keep is not None:
        all_pred_uids = set([key.split('_mol')[0]
                            for key in full_results.keys()])
        uids_to_pop = [f'{uid}_mol0' for uid in sorted(
            all_pred_uids - set(ids_to_keep))]
    else:
        uids_to_pop = []

    if len(uids_to_pop) > 0:
        print(f'Pop {len(uids_to_pop)} uids')

    for uid in full_results.keys():
        if len(full_results[uid]) == 0:
            print(f'{uid} has no valid samples')
            uids_to_pop.append(uid)
            continue

        if use_separate_samples:
            samples = full_results[uid]['sample_metrics']
        else:
            samples = full_results[uid]

        if len(samples) == 0:
            print(f'{uid} has no valid samples')
            uids_to_pop.append(uid)
            continue

    for uid in uids_to_pop:
        full_results.pop(uid)

    return full_results


def get_final_results_for_df(full_results, score_names, score_name_prefix='', posebusters_filter=False,
                             fast_filter=False, ids_to_keep=None):
    def get_row(results, score_name, full_score_name, posebusters_filter):
        scored_results = get_best_results_by_score(results, score_name)

        rmsds = np.array([item['rmsd'] for item in scored_results.values()])
        sym_rmsds = np.array([item['symm_rmsd']
                             for item in scored_results.values()])
        tr_errs = np.array([item['tr_err']
                           for item in scored_results.values()])

        row = {
            'ranking': full_score_name,
            'RMSD < 2A': (rmsds <= 2).mean(),
            'RMSD < 5A': (rmsds <= 5).mean(),
            'avg RMSD': rmsds.mean(),
            'median RMSD': np.median(rmsds),
            'SymRMSD < 2A': (sym_rmsds <= 2).mean(),
            'SymRMSD < 5A': (sym_rmsds <= 5).mean(),
            'avg SymRMSD': sym_rmsds.mean(),
            'median SymRMSD': np.median(sym_rmsds),
            'avg tr_err': tr_errs.mean(),
            'median tr_err': np.median(tr_errs),
            'tr_err < 1A': (tr_errs <= 1).mean(),
            'num_samples': len(scored_results.values()),
        }

        if posebusters_filter:
            posebusters_all = np.array([item['all_posebusters_filters_passed_count']
                                        for item in scored_results.values()])
            row['SymRMSD < 2A & PB valid'] = np.logical_and(
                sym_rmsds < 2, posebusters_all == 27).mean()
        return row, scored_results

    rows_list = []
    all_scored_results = {}

    full_results = filter_empty_results_and_keep_necessary_ids(
        full_results, use_separate_samples=True, ids_to_keep=ids_to_keep)

    if posebusters_filter:
        filtered_results_posebusters = filter_results_by_posebusters(
            copy.deepcopy(full_results))

    if fast_filter:
        filtered_results_fast = filter_results_by_fast(
            copy.deepcopy(full_results))

    for score_name in score_names:
        full_score_name = f'{score_name_prefix}{score_name}'

        row, scored_results = get_row(
            full_results, score_name, full_score_name, posebusters_filter=posebusters_filter)
        all_scored_results[full_score_name] = scored_results
        rows_list.append(row)

        if posebusters_filter:
            real_score_name = f'{full_score_name}_posebusters'
            row, scored_results = get_row(filtered_results_posebusters, score_name, real_score_name,
                                          posebusters_filter=posebusters_filter)
            all_scored_results[real_score_name] = scored_results
            rows_list.append(row)

        if fast_filter:
            real_score_name = f'{full_score_name}_fast'
            row, scored_results = get_row(filtered_results_fast, score_name, real_score_name,
                                          posebusters_filter=posebusters_filter)
            all_scored_results[real_score_name] = scored_results
            rows_list.append(row)

    return rows_list, all_scored_results


def add_score_results(all_rmsds_new, score_res, score_name, n_samples=None):
    extended_results = {}
    for uid in tqdm(all_rmsds_new.keys(), desc='Adding score results'):
        new_samples = []
        for i in range(len(all_rmsds_new[uid])):
            sample = all_rmsds_new[uid][i]
            sample_scores = np.array(score_res[f'{uid}_{i}'])
            nan_mask = np.isnan(sample_scores).sum(axis=1).astype(bool)
            if nan_mask.sum() > 0:
                if score_name == 'mult':
                    sample_scores[nan_mask, 2] = 6.
                    sample_scores[nan_mask, 0] = 0.
                    sample_scores[nan_mask, 1] = 0.
                elif score_name == 'bin':
                    sample_scores[nan_mask] = 0.
                elif score_name == 'reg':
                    sample_scores[nan_mask] = 50.

            sample_scores = -sample_scores
            if n_samples is None:
                n_samples = len(sample_scores)
            mean_scores = np.mean(sample_scores[:n_samples], axis=0)

            for idx in range(len(mean_scores)):
                sample[f'{score_name}_{idx}'] = mean_scores[idx]

            new_samples.append(sample)
        extended_results[uid] = new_samples
    return extended_results


def construct_output_dict(preds, dataset):
    output_dict = {}

    for complex in dataset.complexes:
        uid_full = complex.name
        uid_real = uid_full.split('_conf')[0]

        preds_list = preds[uid_full]
        if len(preds_list) == 0:
            continue

        if uid_real not in output_dict:
            output_dict[uid_real] = {
                'sample_metrics': [],
                'orig_mol': complex.ligand.orig_mol,
            }
        samples = []
        for pred in preds_list:
            sample = {
                'pred_pos': pred['transformed_orig'] + pred['full_protein_center'].reshape(1, 3),
                'error_estimate_0': pred['error_estimate_0'],
            }
            samples.append(sample)
        output_dict[uid_real]['sample_metrics'].extend(samples)
    return output_dict


def get_simple_metrics_df(all_real_rmsds, compute_symm_rmsd, mol2isomorphisms, score_names):
    full_results = {}
    skipped_samples = []
    for uid, samples in tqdm(all_real_rmsds.items(), desc='Computing metrics'):
        samples_results = []
        failed_symm_rmsd_count = 0

        true_pos = samples[0]['true_pos']
        for idx in range(len(samples)):
            pred_pos = samples[idx]['transformed_orig']

            if true_pos.shape[0] != pred_pos.shape[0]:
                error_msg = (
                    f'{uid}_{idx:<8} Atom count mismatch: '
                    f'true_pos={true_pos.shape[0]} atoms, pred_pos={pred_pos.shape[0]} atoms. '
                    f'This typically indicates hydrogens in true structure.'
                )
                print(error_msg)
                skipped_samples.append({
                    'uid': uid,
                    'idx': idx,
                    'true_atoms': int(true_pos.shape[0]),
                    'pred_atoms': int(pred_pos.shape[0]),
                })
                continue

            tr_pred = pred_pos.mean(axis=0)
            tr_true = true_pos.mean(axis=0)
            tr_err = np.linalg.norm(tr_pred - tr_true)

            rmsd = np.sqrt(
                ((true_pos - pred_pos) ** 2).sum(axis=1).sum() / true_pos.shape[0])
            if compute_symm_rmsd and failed_symm_rmsd_count < 3:  # compute symmetry rmsd
                try:
                    mol2iso = mol2isomorphisms.get(uid.split('_conf')[0])
                    if mol2iso is None:
                        symm_rmsd = rmsd
                        failed_symm_rmsd_count += 1
                    else:
                        symm_rmsd = get_symmetry_rmsd_with_isomorphisms(
                            true_pos, pred_pos, mol2iso)
                except TimeoutException:
                    symm_rmsd = rmsd
                    failed_symm_rmsd_count += 1
            else:
                symm_rmsd = rmsd

            results = {
                'tr_pred': tr_pred,
                'tr_err': float(tr_err),
                'symm_rmsd': float(symm_rmsd),
                'rmsd': float(rmsd),
                'pred_pos': pred_pos,
            }
            for score_name in set(score_names) - {'random', 'symm_rmsd'}:
                results[score_name] = float(samples[idx][score_name])

            samples_results.append(results)
        samples_results_dict = {
            'sample_metrics': samples_results,
            'true_pos': true_pos,
            'orig_mol': samples[0]['orig_mol'],
        }
        if len(samples_results_dict['sample_metrics']) > 0:
            full_results[uid] = samples_results_dict
        else:
            print(f'{uid} has no valid samples')
            print(
                f'{uid} true_pos.shape[0] != pred_pos.shape[0]', true_pos.shape, pred_pos.shape)

    if len(full_results) != len(all_real_rmsds):
        print('Initial length of test_names', len(all_real_rmsds))
        print('Length of full_results', len(full_results))

        # Detailed summary
        print('\n' + '='*80)
        print('METRICS COMPUTATION SUMMARY')
        print('='*80)
        total_samples = sum(len(samples) for samples in all_real_rmsds.values())
        total_processed = sum(len(result['sample_metrics']) for result in full_results.values())
        print(f'Total samples: {total_samples}')
        print(f'Successfully processed: {total_processed}')
        print(f'Skipped: {total_samples - total_processed}')

        if skipped_samples:
            print(f'\nAtom count mismatches: {len(skipped_samples)}')
            print('Common causes:')
            print('  - True molecule SDF contains hydrogens, predictions do not')
            print('  - Different protonation states')
            print('  - Molecule preprocessing inconsistency')
            print('\nFirst 5 examples:')
            for i, sample in enumerate(skipped_samples[:5]):
                print(f'  {i+1}. {sample["uid"]} (sample {sample["idx"]}): '
                      f'{sample["true_atoms"]} atoms (true) vs {sample["pred_atoms"]} atoms (pred)')
            if len(skipped_samples) > 5:
                print(f'  ... and {len(skipped_samples) - 5} more')
            print('\nRecommendation: Ensure SDF files contain only heavy atoms (no hydrogens)')
        print('='*80 + '\n')

    rows_list, all_scored_results = get_final_results_for_df(
        full_results, score_names)
    return pd.DataFrame(rows_list), all_scored_results, full_results
