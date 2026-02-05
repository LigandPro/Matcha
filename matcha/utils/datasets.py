import os
import torch
import omegaconf
from matcha.dataset.pdbbind import (
    PDBBind, PDBBindWithSortedBatching)
from matcha.utils.paths import get_sequences_path, get_esm_embeddings_path
from matcha.utils.log import get_logger

logger = get_logger(__name__)


def get_datasets(conf, splits,
                 complex_collate_fn=None,
                 predicted_ligand_transforms_path=None, use_predicted_tr_only=True,
                 is_train_dataset=True,
                 n_preds_to_use=1, use_all_chains=None, stage_num=None):

    all_datasets = {}
    use_sorted_batching = conf.get(
        'use_sorted_batching', False)
    randomize_bond_neighbors = conf.get('randomize_bond_neighbors', True)

    if use_all_chains is None:
        use_all_chains = conf.get('use_all_chains', False)

    for split in splits:
        dataset_list = conf.test_dataset_types
        if split == 'train':
            add_all_atom_pos = True
            min_lig_size = 7
        elif split == 'val':
            add_all_atom_pos = True
            min_lig_size = 7
        elif split == 'test':
            add_all_atom_pos = True
            min_lig_size = 0

        split_datasets = []
        for dataset_type in dataset_list:
            sequences_path = get_sequences_path(dataset_type, conf, split)
            esm_emb_path = get_esm_embeddings_path(dataset_type, conf, split)

            test_split_path = None
            train_split_path = None
            val_split_path = None
            if dataset_type == 'pdbbind' or dataset_type == 'pdbbind_conf':
                data_dir = conf.pdbbind_data_dir
                train_split_path = conf.pdbbind_split_train
                val_split_path = conf.pdbbind_split_val
                test_split_path = conf.pdbbind_split_test
            elif dataset_type == 'dockgen' or dataset_type == 'dockgen_full' or dataset_type == 'dockgen_full_conf':
                data_dir = conf.dockgen_data_dir
                val_split_path = conf.dockgen_split_val
                if dataset_type == 'dockgen_full' or dataset_type == 'dockgen_full_conf':
                    test_split_path = conf.dockgen_split_test_full
                else:
                    test_split_path = conf.dockgen_split_test
            elif dataset_type.startswith('posebusters'):
                data_dir = conf.posebusters_data_dir
                test_split_path = conf.posebusters_split_test
            elif dataset_type.startswith('astex'):
                data_dir = conf.astex_data_dir
                test_split_path = conf.astex_split_test
            elif dataset_type.startswith('any'):
                data_dir = conf.any_data_dir
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            if dataset_type.endswith('_conf'):
                data_dir_conf = os.path.join(conf.data_folder, f'{dataset_type}_conformers')
                os.makedirs(data_dir_conf, exist_ok=True)
            else:
                data_dir_conf = None

            if split == 'train':
                split_path = train_split_path
            elif split == 'val':
                split_path = val_split_path
            elif split == 'test':
                split_path = test_split_path

            split_dataset = PDBBind(
                data_dir=data_dir,
                split_path=split_path,
                tr_std=conf.tr_std,
                esm_embeddings_path=esm_emb_path,
                sequences_path=sequences_path,
                cache_path=conf.cache_path,
                num_dataset_workers=1,
                std_protein_pos=conf.std_protein_pos,
                std_lig_pos=conf.std_lig_pos,
                ligand_mask_ratio=conf.ligand_mask_ratio,
                protein_mask_ratio=conf.protein_mask_ratio,
                esm_emb_noise_std=conf.esm_emb_noise_std,
                dataset_type=dataset_type,
                predicted_ligand_transforms_path=predicted_ligand_transforms_path,
                add_all_atom_pos=add_all_atom_pos,
                min_lig_size=min_lig_size,
                use_predicted_tr_only=use_predicted_tr_only,
                randomize_bond_neighbors=randomize_bond_neighbors,
                data_dir_conf=data_dir_conf,
                is_train_dataset=is_train_dataset,
                n_preds_to_use=n_preds_to_use,
                use_all_chains=use_all_chains,
                stage_num=stage_num,
                n_confs_override=conf.get('n_confs_override', None),
            )

            if use_sorted_batching:
                split_dataset = PDBBindWithSortedBatching(dataset=split_dataset, batch_limit=conf.batch_limit,
                                                          data_collator=complex_collate_fn)
            split_datasets.append(split_dataset)

        if use_sorted_batching:
            all_datasets[split] = {dataset.dataset.dataset_type: dataset for dataset in split_datasets}
        else:
            all_datasets[split] = {dataset.dataset_type: dataset for dataset in split_datasets}

    return all_datasets
