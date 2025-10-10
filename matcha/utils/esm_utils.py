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

confProDy(verbosity='none')


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

    for dataset_name in conf.test_dataset_types:
        dataset_data_dir = get_dataset_path(dataset_name, conf)
        save_id2seq_path = get_sequences_path(dataset_name, conf)
        names = [name for name in os.listdir(
            dataset_data_dir) if not name.startswith('.')]

        id2seq = {}
        bad_ids = []
        for name in tqdm(names, desc=f'Preparing {dataset_name} sequences'):
            rec_path = get_protein_path(name, dataset_name, dataset_data_dir)
            try:
                l = get_structure_from_file(rec_path)
            except Exception as e:
                bad_ids.append(name)
                continue

            for i, seq in enumerate(l):
                id2seq[f'{name}_chain_{i}'] = seq

        print(dataset_name, 'has', len(bad_ids), 'bad IDs')
        save_json(id2seq, save_id2seq_path)
        print(f'Saved sequences to {save_id2seq_path}')
        print()


def get_tokens(seqs, tokenizer):
    tokens = tokenizer.batch_encode_plus(seqs)['input_ids']
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
    print('Sequences loaded')

    if reduce_to_unique_sequences:
        print('Reducing to unique sequences')
        print('Number of sequences:', len(all_data))
        prepared_sequences = list(
            set([''.join(seq) for seq in all_data.values()]))
        print('Number of unique sequences:', len(prepared_sequences))
    else:
        prepared_sequences = [''.join(seq) for seq in all_data.values()]

    tokens = get_tokens(prepared_sequences, tokenizer)
    embeddings = get_embeddings_residue(
        tokens=tokens, esm_model=model, device=device)

    if reduce_to_unique_sequences:
        names = prepared_sequences
    else:
        names = all_data.keys()

    print('Number of protein chains:', len(names))
    save_data = {name: emb for name, emb in zip(names, embeddings)}
    torch.save(save_data, save_emb_path)


def compute_esm_embeddings(conf):
    reduce_to_unique_sequences = False
    model_type = 'hf_esm_12'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Available device:', device)

    if model_type == 'hf_esm_6':
        model_checkpoint = 'facebook/esm2_t6_8M_UR50D'
    elif model_type == 'hf_esm_12':
        model_checkpoint = 'facebook/esm2_t12_35M_UR50D'
    elif model_type == 'hf_esm_33':
        model_checkpoint = 'facebook/esm2_t33_650M_UR50D'
    else:
        print(f'Model {model_type} not found')

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model.eval()
    model.to(device=device)
    print('Model loaded')

    num_params_trainable = 0
    num_params_all = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params_trainable += int(
                torch.prod(torch.tensor(param.data.shape)))
        num_params_all += int(torch.prod(torch.tensor(param.data.shape)))
    print('Trainable parameters:', num_params_trainable)
    print('All parameters:', num_params_all)

    for dataset_name in conf.test_dataset_types:
        dataset_sequence_path = get_sequences_path(dataset_name, conf)
        save_emb_path = get_esm_embeddings_path(dataset_name, conf)

        save_dataset_embeddings(dataset_sequence_path, save_emb_path, model=model, tokenizer=tokenizer, device=device,
                                reduce_to_unique_sequences=reduce_to_unique_sequences)
        print(f'Saved embeddings to {save_emb_path}')
        print()
