import os
from tqdm import tqdm
import shutil
from argparse import ArgumentParser
from omegaconf import OmegaConf
from collections import defaultdict

import numpy as np
from rdkit.Chem import RemoveAllHs, SDWriter

from matcha.utils.alignment import (filter_protein_chains_by_ligand_distance, align_to_binding_site,
                                    restore_atom_order, align_to_binding_site_by_pocket)
from matcha.utils.preprocessing import read_molecule
from matcha.utils.paths import get_protein_path, get_ligand_path, get_dataset_path
from matcha.utils.log import get_logger

logger = get_logger(__name__)


def process_single_prediction(ref_protein_path, ref_lig_path, pred_folder, uid, has_pred_proteins, tmp_folder):
    try:
        tmp_folder = os.path.join(tmp_folder, uid)
        os.makedirs(tmp_folder, exist_ok=True)

        ref_protein_path_cropped = os.path.join(
            tmp_folder, 'ref_protein_cropped.pdb')
        chains = filter_protein_chains_by_ligand_distance(
            ref_protein_path, ref_lig_path, ref_protein_path_cropped)
        if len(chains) == 0:
            return None, None

        pred_ligand_path = os.path.join(pred_folder, 'lig_0.sdf')
        if has_pred_proteins:
            pred_protein_path = os.path.join(pred_folder, 'prot.pdb')
        else:
            pred_protein_path = ref_protein_path

        if has_pred_proteins:
            pred_prot_path_for_alignment = pred_protein_path
            ref_prot_path_for_alignment = ref_protein_path_cropped
            # Align protein pockets
            aligned_ligand_pdb = os.path.join(
                tmp_folder, f"aligned_ligand.pdb")
            _ = align_to_binding_site(
                predicted_protein=pred_prot_path_for_alignment,
                reference_protein=ref_prot_path_for_alignment,
                predicted_ligand=pred_ligand_path,
                reference_ligand=ref_lig_path,
                aligned_ligand_path=aligned_ligand_pdb,
                aligned_protein_path=os.path.join(
                    tmp_folder, f"aligned_protein.pdb"),
            )
        else:
            aligned_ligand_pdb = pred_ligand_path

        pred_mol = read_molecule(aligned_ligand_pdb)
        mol_true = read_molecule(ref_lig_path)

        try:
            pred_mol = RemoveAllHs(pred_mol, sanitize=True)
        except Exception as e:
            logger.error(f'Failed to remove hydrogens for predicted ligand {uid}: {e}')
            pred_mol = RemoveAllHs(pred_mol, sanitize=False)

        try:
            mol_true = RemoveAllHs(mol_true, sanitize=True)
        except Exception as e:
            logger.error(f'Failed to remove hydrogens for true ligand {uid}: {e}')
            mol_true = RemoveAllHs(mol_true, sanitize=False)

        pred_mol_restored = restore_atom_order(mol_true, pred_mol)
        shutil.rmtree(tmp_folder)
        pred_pos = pred_mol_restored.GetConformer().GetPositions()

    except Exception as e:
        logger.error(f'failed {ref_lig_path}: {e}')
        pred_pos = None
        pred_mol_restored = None
    return pred_pos, pred_mol_restored


def process_single_prediction_pocket_alignment(ref_protein_path, ref_lig_path, pred_folder, uid, has_pred_proteins, tmp_folder):

    try:
        tmp_folder = os.path.join(tmp_folder, uid)
        os.makedirs(tmp_folder, exist_ok=True)

        ref_protein_path_cropped = os.path.join(
            tmp_folder, 'ref_protein_cropped.pdb')
        filter_protein_chains_by_ligand_distance(
            ref_protein_path, ref_lig_path, ref_protein_path_cropped)

        pred_ligand_path = os.path.join(pred_folder, 'lig_0.sdf')
        if has_pred_proteins:
            pred_protein_path = os.path.join(pred_folder, 'prot.pdb')
        else:
            pred_protein_path = ref_protein_path

        pred_protein_path_cropped = os.path.join(
            tmp_folder, 'pred_protein_cropped.pdb')
        chain_ids_list = filter_protein_chains_by_ligand_distance(pred_protein_path, pred_ligand_path,
                                                                  pred_protein_path_cropped, return_all=True)

        best_rmsd = 10000
        best_chain = chain_ids_list[0]
        for chain_id in chain_ids_list:
            pred_prot_path_for_alignment = os.path.join(
                tmp_folder, f'pred_protein_cropped_{chain_id}.pdb')
            ref_prot_path_for_alignment = ref_protein_path_cropped
            # Align protein pockets
            aligned_ligand_pdb = os.path.join(
                tmp_folder, f"aligned_ligand_{chain_id}.pdb")
            try:
                pocket_rms = align_to_binding_site_by_pocket(
                    predicted_protein=pred_prot_path_for_alignment,
                    reference_protein=ref_prot_path_for_alignment,
                    predicted_ligand=pred_ligand_path,
                    reference_ligand=ref_lig_path,
                    aligned_ligand_path=aligned_ligand_pdb,
                    aligned_protein_path=os.path.join(
                        tmp_folder, f"aligned_protein.pdb"),
                )
            except Exception as e:
                logger.error(
                    f'Alignment failed for {uid} chain {chain_id} (out of {chain_ids_list})')
                continue
            if pocket_rms < best_rmsd:
                best_rmsd = pocket_rms
                best_chain = chain_id
        aligned_ligand_pdb = os.path.join(
            tmp_folder, f"aligned_ligand_{best_chain}.pdb")

        pred_mol = read_molecule(aligned_ligand_pdb)
        mol_true = read_molecule(ref_lig_path)

        try:
            pred_mol = RemoveAllHs(pred_mol, sanitize=True)
        except Exception as e:
            logger.error(f'Failed to remove hydrogens for predicted ligand {uid}: {e}')
            pred_mol = RemoveAllHs(pred_mol, sanitize=False)
        
        try:
            mol_true = RemoveAllHs(mol_true, sanitize=True)
        except Exception as e:
            logger.error(f'Failed to remove hydrogens for true ligand {uid}: {e}')
            mol_true = RemoveAllHs(mol_true, sanitize=False)

        pred_mol_restored = restore_atom_order(mol_true, pred_mol)
        shutil.rmtree(tmp_folder)
        pred_pos = pred_mol_restored.GetConformer().GetPositions()

    except Exception as e:
        logger.error(f'failed {ref_lig_path}: {e}')
        pred_pos = None
        pred_mol_restored = None
    return pred_pos, pred_mol_restored


def process_all_predictions(preds_path, dataset_name, has_pred_proteins, 
                            alignment_type, tmp_folder, method_path, conf):
    dataset_path = get_dataset_path(dataset_name, conf)
    uids = os.listdir(preds_path)
    if dataset_name == 'dockgen':
        real_dataset_name = 'dockgen_full'
    else:
        real_dataset_name = dataset_name
    best_preds_path = os.path.join(method_path, f'{real_dataset_name}_conf', 'best_base_predictions')
    os.makedirs(best_preds_path, exist_ok=True)

    predicted_positions = defaultdict(dict)
    for i, uid in enumerate(tqdm(uids, desc=f"Processing {dataset_name} ({len(uids)} complexes) for {os.path.basename(method_path)}")):
        if os.path.exists(os.path.join(preds_path, uid)):
            ref_protein_path = get_protein_path(
                uid, dataset_name, dataset_path)

            ref_lig_path = get_ligand_path(uid, dataset_name, dataset_path)
            if dataset_name == 'pdbbind':
                mol_tmp = read_molecule(ref_lig_path, remove_hs=False, sanitize=True, strict=True)
                if mol_tmp is None:
                    ref_lig_path = ref_lig_path.replace('.mol2', '.sdf')
                    logger.warning(f'Failed to read mol2 file {ref_lig_path}. Trying to read sdf file instead.')

            # parse top-1 prediction
            for conf_id in [0]:
                if alignment_type == 'base':
                    pred_pos, pred_mol = process_single_prediction(
                        ref_protein_path=ref_protein_path,
                        ref_lig_path=ref_lig_path,
                        pred_folder=os.path.join(
                            preds_path, uid, f'conf_{conf_id}'),
                        uid=uid,
                        has_pred_proteins=has_pred_proteins,
                        tmp_folder=tmp_folder,
                    )
                elif alignment_type == 'pocket':
                    pred_pos, pred_mol = process_single_prediction_pocket_alignment(
                        ref_protein_path=ref_protein_path,
                        ref_lig_path=ref_lig_path,
                        pred_folder=os.path.join(
                            preds_path, uid, f'conf_{conf_id}'),
                        uid=uid,
                        has_pred_proteins=has_pred_proteins,
                        tmp_folder=tmp_folder,
                    )
                else:
                    raise ValueError(
                        f'Unknown alignment_type: {alignment_type}')
                if pred_pos is not None:
                    # Save aligned molecule to method_path folder
                    aligned_mol_path = os.path.join(best_preds_path, f'{uid}.sdf')
                    try:
                        writer = SDWriter(str(aligned_mol_path))
                        writer.write(pred_mol)
                    except Exception as e:
                        writer = SDWriter(str(aligned_mol_path))
                        writer.SetKekulize(False)
                        try:
                            writer.write(pred_mol)
                        except Exception as e:
                            logger.error(f'Error processing {uid}: {e}')
                            writer.close()
                            continue
                    writer.close()

                    new_sample = {
                        'error_estimate_0': 0,
                        'pred_pos': pred_pos,
                    }
                    if f'{uid}_mol0' not in predicted_positions:
                        predicted_positions[f'{uid}_mol0'] = {'sample_metrics': [new_sample]}
                    else:
                        predicted_positions[f'{uid}_mol0']['sample_metrics'].append(new_sample)
        else:
            logger.error(f'No prediction for {uid}')

    return predicted_positions


def compute_aligned_rmsd_for_dataset(conf, initial_preds_path, methods_data, dataset_names, alignment_type='base'):

    for dataset_name in dataset_names:
        run_names = []
        for method_name, has_pred_proteins in methods_data.items():
            run_name = f'{method_name}_{alignment_type}'
            method_path = os.path.join(conf.inference_results_folder, run_name)
            os.makedirs(method_path, exist_ok=True)
            tmp_folder = os.path.join(method_path, 'tmp_data')
            os.makedirs(tmp_folder, exist_ok=True)
            final_predictions_path = os.path.join(
                method_path, f'{dataset_name}_final_preds_fast_metrics.npy')

            logger.info(f"Processing {dataset_name} for {method_name} with {alignment_type} alignment (has_pred_proteins: {has_pred_proteins})")

            predicted_positions = process_all_predictions(
                preds_path=os.path.join(
                    initial_preds_path, method_name, dataset_name),
                dataset_name=dataset_name,
                has_pred_proteins=has_pred_proteins,
                alignment_type=alignment_type,
                tmp_folder=tmp_folder,
                method_path=method_path,
                conf=conf,
            )
            np.save(final_predictions_path, [predicted_positions])
            shutil.rmtree(tmp_folder)
            run_names.append(run_name)
    return run_names


if __name__ == '__main__':
    parser = ArgumentParser(description="Read file form Command line.")
    parser.add_argument("-p", "--paths-config", dest="paths_config_filename",
                        required=True, help="config file with paths")
    parser.add_argument("-a", "--alignment-type", dest="alignment_type",
                        required=True, help="alignment type", default='base')
    parser.add_argument("--init-preds-path", dest="initial_preds_path",
                        required=True, help="initial predictions path")
    args = parser.parse_args()

    # Load main model config
    conf = OmegaConf.load(args.paths_config_filename)
    args = parser.parse_args()

    # set True to use predicted proteins, False to use reference proteins
    methods_data = {
        'af3': True,
        'diffdock': False,
    }
    dataset_names = ['astex', 'dockgen', 'pdbbind', 'posebusters']ё

    logger.info(f'Computing aligned RMSD for {dataset_names} with {args.alignment_type} alignment')
    logger.info(f'Initial predictions path: {args.initial_preds_path}')
    logger.info(f'Methods data: {methods_data}')
    run_names = compute_aligned_rmsd_for_dataset(
        conf=conf,
        initial_preds_path=args.initial_preds_path, 
        methods_data=methods_data, 
        dataset_names=dataset_names, 
        alignment_type=args.alignment_type)
    logger.info(f'Run names: {run_names}')

