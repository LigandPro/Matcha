import os
from tqdm import tqdm
import numpy as np
import gc

from deli import save_json, load
import prody
from prody import confProDy
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from matcha.utils.paths import (get_dataset_path, get_protein_path,
                                get_sequences_path, get_esm_embeddings_path)
from matcha.utils.log import get_logger

confProDy(verbosity='none')
logger = get_logger(__name__)


def get_structure_from_file(file_path):
    rec = prody.parsePDB(file_path)
    seq = rec.ca.getSequence()

    res_chain_ids = rec.ca.getChids()
    res_seg_ids = rec.ca.getSegnames()
    res_chain_ids = np.asarray(
        [s + c for s, c in zip(res_seg_ids, res_chain_ids)])
    chain_ids = np.unique(res_chain_ids)
    seq = np.array([s for s in seq])

    chain_sequences = []
    for i, id in enumerate(chain_ids):
        chain_mask = res_chain_ids == id
        chain_seq = ''.join(seq[chain_mask])
        chain_sequences.append(chain_seq)
    return chain_sequences


def compute_sequences(conf):
    os.makedirs(conf.data_folder, exist_ok=True)
    split_data = {
        'test': conf.test_dataset_types,
    }

    for split, dataset_names in split_data.items():
        for dataset_name in dataset_names:
            logger.info(f'Computing sequences for {dataset_name} on {split} split')
            dataset_data_dir = get_dataset_path(dataset_name, conf)
            logger.info(f'dataset_data_dir: {dataset_data_dir}')
            save_id2seq_path = get_sequences_path(dataset_name, conf)
            logger.info(f'save_id2seq_path: {save_id2seq_path}')
            names = [name for name in os.listdir(
                dataset_data_dir) if not name.startswith('.')]

            id2seq = {}
            bad_ids = []
            for name in tqdm(names, desc=f'Preparing {dataset_name} sequences'):
                dataset_name_real = dataset_name
                real_name = name
                rec_path = get_protein_path(name, dataset_name_real, dataset_data_dir, 
                                            crop_mol=False)
                try:
                    l = get_structure_from_file(rec_path)
                except Exception as e:
                    bad_ids.append(name)
                    continue

                for i, seq in enumerate(l):
                    id2seq[f'{real_name}_chain_{i}'] = seq

            logger.info(f'{split}, {dataset_name} has {len(bad_ids)} bad IDs')
            logger.info(f'total chains: {len(id2seq)}')
            save_json(id2seq, save_id2seq_path)
            logger.info(f'Saved sequences to {save_id2seq_path}')
            logger.info("")


def get_tokens(seqs, tokenizer):
    # batch_encode_plus was deprecated in transformers; use __call__ instead
    encoded = tokenizer(seqs, padding=False, truncation=True)
    tokens = encoded['input_ids']
    return tokens


def get_embeddings_residue(tokens, esm_model, device):
    embeddings = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(tokens, desc='Computing ESM embeddings')):
            if not i % 1000 and i != 0:
                torch.cuda.empty_cache()
                gc.collect()
            batch = torch.tensor(batch).to(device)
            batch = batch[None, :]
            res = esm_model(batch, output_hidden_states=True)[
                'hidden_states'][-1]
            embeddings.append(res[0, 1:-1].cpu())
    return embeddings


def save_dataset_embeddings(dataset_sequence_path, save_emb_path, model, tokenizer, device,
                            reduce_to_unique_sequences=False):

    all_data = load(dataset_sequence_path)
    logger.info('Sequences loaded')

    if reduce_to_unique_sequences:
        logger.info('Reducing to unique sequences')
        logger.info(f'Number of sequences: {len(all_data)}')
        prepared_sequences = list(
            set([''.join(seq) for seq in all_data.values()]))
        logger.info(f'Number of unique sequences: {len(prepared_sequences)}')
    else:
        prepared_sequences = [''.join(seq) for seq in all_data.values()]

    tokens = get_tokens(prepared_sequences, tokenizer)
    embeddings = get_embeddings_residue(
        tokens=tokens, esm_model=model, device=device)

    if reduce_to_unique_sequences:
        names = prepared_sequences
    else:
        names = all_data.keys()

    logger.info(f'Number of protein chains: {len(names)}')
    save_data = {name: emb for name, emb in zip(names, embeddings)}
    torch.save(save_data, save_emb_path)


def compute_esm_embeddings(conf, model_type='hf_esm_12'):
    reduce_to_unique_sequences = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Available device: {device}')

    if model_type == 'hf_esm_6':
        model_checkpoint = 'facebook/esm2_t6_8M_UR50D'
    elif model_type == 'hf_esm_12':
        model_checkpoint = 'facebook/esm2_t12_35M_UR50D'
    elif model_type == 'hf_esm_33':
        model_checkpoint = 'facebook/esm2_t33_650M_UR50D'
    else:
        logger.error(f'Model {model_type} not found')
        raise ValueError(f'Model {model_type} not found')

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model.eval()
    model.to(device=device)
    logger.info('ESM model loaded')

    num_params_trainable = 0
    num_params_all = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params_trainable += int(
                torch.prod(torch.tensor(param.data.shape)))
        num_params_all += int(torch.prod(torch.tensor(param.data.shape)))
    logger.info(f'Trainable parameters: {num_params_trainable}')
    logger.info(f'All parameters: {num_params_all}')

    split_data = {
        'test': conf.test_dataset_types,
    }

    for split, dataset_names in split_data.items():
        for dataset_name in dataset_names:
            dataset_sequence_path = get_sequences_path(dataset_name, conf, split)
            save_emb_path = get_esm_embeddings_path(dataset_name, conf, split)

            save_dataset_embeddings(dataset_sequence_path, save_emb_path, model=model, tokenizer=tokenizer, device=device,
                                    reduce_to_unique_sequences=reduce_to_unique_sequences)
            logger.info(f'Saved embeddings to {save_emb_path}')
            logger.info("")
