import os
import torch
from matcha.dataset.pdbbind import PDBBind, PDBBindWithSortedBatching
from matcha.dataset.pdbbind_scoring import PDBBindForScoringInference
from matcha.utils.paths import get_sequences_path, get_esm_embeddings_path, get_dataset_path


def get_datasets(conf, splits, return_separately=False,
                 complex_collate_fn=None, predicted_complex_positions_path=None,
                 predicted_ligand_transforms_path=None, use_predicted_tr_only=True,
                 n_preds_to_use=1, use_all_chains=None):
    all_datasets = {}
    use_sorted_batching = conf.get('use_sorted_batching', False)

    if use_all_chains is None:
        use_all_chains = conf.get('use_all_chains', False)

    for split in splits:
        dataset_list = conf.test_dataset_types
        add_all_atom_pos = True

        split_datasets = []
        for dataset_type in dataset_list:
            sequences_path = get_sequences_path(dataset_type, conf)
            esm_emb_path = get_esm_embeddings_path(dataset_type, conf)
            data_dir = get_dataset_path(dataset_type, conf)
            test_split_path = None
            if dataset_type == 'pdbbind_conf':
                test_split_path = conf.pdbbind_split_test
            elif dataset_type == 'posebusters_conf':
                test_split_path = conf.posebusters_split_test

            data_dir_conf = os.path.join(
                conf.data_folder, f'{dataset_type}_conformers')
            os.makedirs(data_dir_conf, exist_ok=True)
            split_path = test_split_path

            if predicted_complex_positions_path is not None:
                # scoring inference
                split_dataset = PDBBindForScoringInference(
                    data_dir=data_dir,
                    split_path=split_path,
                    tr_std=1.,
                    esm_embeddings_path=esm_emb_path,
                    sequences_path=sequences_path,
                    cache_path=conf.cache_path,
                    predicted_complex_positions_path=predicted_complex_positions_path,
                    dataset_type=dataset_type,
                    add_all_atom_pos=add_all_atom_pos,
                    n_preds_to_use=n_preds_to_use,
                    use_all_chains=use_all_chains,
                )
            else:
                split_dataset = PDBBind(
                    data_dir=data_dir,
                    split_path=split_path,
                    tr_std=conf.tr_std,
                    esm_embeddings_path=esm_emb_path,
                    sequences_path=sequences_path,
                    cache_path=conf.cache_path,
                    dataset_type=dataset_type,
                    predicted_ligand_transforms_path=predicted_ligand_transforms_path,
                    add_all_atom_pos=add_all_atom_pos,
                    use_predicted_tr_only=use_predicted_tr_only,
                    data_dir_conf=data_dir_conf,
                    n_preds_to_use=n_preds_to_use,
                    use_all_chains=use_all_chains,
                    n_confs_override=conf.get('n_confs_override', None),
                )
            if use_sorted_batching:
                split_dataset = PDBBindWithSortedBatching(dataset=split_dataset, batch_limit=conf.batch_limit,
                                                          data_collator=complex_collate_fn)
            split_datasets.append(split_dataset)

        for dataset in split_datasets:
            if use_sorted_batching:
                print(split, dataset.dataset.dataset_type, len(dataset),
                      len(dataset.dataset.complexes), 'sorted batching')
            else:
                print(split, dataset.dataset_type, len(dataset))

        if return_separately:
            if use_sorted_batching:
                all_datasets[split] = {
                    dataset.dataset.dataset_type: dataset for dataset in split_datasets}
            else:
                all_datasets[split] = {
                    dataset.dataset_type: dataset for dataset in split_datasets}
        else:
            if len(split_datasets) > 1:
                combined_dataset = torch.utils.data.ConcatDataset(
                    split_datasets)
                print(split, len(combined_dataset))
            else:
                combined_dataset = split_datasets[0]

            all_datasets[split] = combined_dataset

    return all_datasets
