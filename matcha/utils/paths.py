import os


def get_dataset_path(dataset_name, conf):
    if dataset_name.startswith('astex'):
        return conf.astex_data_dir
    if dataset_name.startswith('posebusters'):
        return conf.posebusters_data_dir
    if dataset_name.startswith('pdbbind'):
        return conf.pdbbind_data_dir
    if dataset_name.startswith('dockgen'):
        return conf.dockgen_data_dir
    if dataset_name.startswith('any'):
        return conf.any_data_dir


def get_protein_path(uid, dataset_name, dataset_data_dir):
    uid = uid.split('_mol')[0]
    if dataset_name.startswith('astex') or dataset_name.startswith('posebusters') or dataset_name.startswith('any'):
        rec_path = os.path.join(dataset_data_dir, uid, f'{uid}_protein.pdb')
    elif dataset_name.startswith('pdbbind') or dataset_name.startswith('dockgen'):
        rec_path = os.path.join(dataset_data_dir, uid,
                                f'{uid}_protein_processed.pdb')
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return rec_path


def get_ligand_path(uid, dataset_name, dataset_data_dir):
    uid = uid.split('_mol')[0]
    if dataset_name.startswith('astex') or dataset_name.startswith('posebusters') or \
            dataset_name.startswith('any') or dataset_name.startswith('pdbbind'):
        lig_path = os.path.join(dataset_data_dir, uid, f'{uid}_ligand.sdf')
    elif dataset_name.startswith('dockgen'):
        lig_path = os.path.join(dataset_data_dir, uid, f'{uid}_ligand.pdb')
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return lig_path


def get_sequences_path(dataset_name, conf):
    return os.path.join(conf.data_folder, f'{dataset_name}_id2seq_new.json')


def get_esm_embeddings_path(dataset_name, conf):
    return os.path.join(conf.data_folder, f'{dataset_name}_esm_embeddings.pt')
