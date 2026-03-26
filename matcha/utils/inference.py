import os
import copy
import time
import safetensors
from tqdm import tqdm
import torch
from matcha.utils.rotation import expm_SO3
from matcha.utils.transforms import (apply_tr_rot_changes_to_batch_inplace, apply_tor_changes_to_batch_inplace,
                                       get_torsion_angles, find_rigid_alignment)
from matcha.utils.log import get_logger

logger = get_logger(__name__)

def load_from_checkpoint(model, checkpoint_path, strict=True):
    checkpoint_path = os.path.expanduser(checkpoint_path)
    state_dict = safetensors.torch.load_file(os.path.join(
        checkpoint_path, 'model.safetensors'), device="cpu")
    model.load_state_dict(state_dict, strict=strict)
    return model


def euler(model, batch, device, num_steps=20):
    cur_batch = copy.copy(batch)
    if hasattr(cur_batch, "ligand"):
        cur_batch.ligand = copy.copy(cur_batch.ligand)
    if hasattr(cur_batch, "protein"):
        cur_batch.protein = copy.copy(cur_batch.protein)
    if cur_batch.ligand.pos.device.type != torch.device(device).type:
        cur_batch = cur_batch.to(device)

    # Clone only the tensors that are updated in-place by the solver.
    cur_batch.ligand.pos = cur_batch.ligand.pos.clone()
    cur_batch.ligand.t = cur_batch.ligand.t.clone()

    h = 1. / num_steps
    batch_size = len(cur_batch)
    R_eye = torch.eye(3, device=device).repeat(batch_size, 1, 1)
    R_agg = cur_batch.ligand.init_rot
    tr = cur_batch.ligand.init_tr
    tor = cur_batch.ligand.init_tor
    cur_batch.ligand.t = torch.zeros_like(cur_batch.ligand.t)

    pos_hist = []
    trajectory = None
    tor_agg = torch.zeros_like(tor)
    for _ in range(num_steps):
        dtr, drot, dtor, _ = model.forward_step(cur_batch)

        tr = tr + h * dtr
        if dtor is not None:
            tor = h * dtor
            tor_agg += tor
        if drot is not None:
            R = expm_SO3(drot, h)
        else:
            R = R_eye

        apply_tor_changes_to_batch_inplace(cur_batch, tor, is_reverse_order=False)
        apply_tr_rot_changes_to_batch_inplace(cur_batch, tr, R)

        cur_batch.ligand.t += h
        R_agg = torch.bmm(R, R_agg)

    tr_agg = tr
    return cur_batch, tr_agg, R_agg, tor_agg, pos_hist, trajectory


def run_evaluation(dataloader, num_steps, solver, model, max_batches=None, batch_timings=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics_dict = {}
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Docking inference")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_start_time = time.perf_counter()
            batch = batch['batch'].to(device, non_blocking=True)
            batch_size = len(batch)
            optimized, tr_agg, R_agg, tor_agg, pos_hist, _ = solver(
                model, batch, device=device, num_steps=num_steps)

            # compute RMSD in case of zero-padded batches
            max_num_atoms = optimized.ligand.pos.shape[1]
            atom_idx = torch.arange(max_num_atoms, device=device)
            pad_mask = atom_idx.unsqueeze(0) >= optimized.ligand.num_atoms.unsqueeze(1)
            optimized.ligand.pos.masked_fill_(pad_mask.unsqueeze(-1), 0.)
            for i in range(len(pos_hist)):
                pos_hist[i].masked_fill_(pad_mask.unsqueeze(-1), 0.)

            # Handle cases where solver returns None for aggregated values
            if tr_agg is None or R_agg is None or tor_agg is None:
                # Skip alignment and use identity transformations
                tr_agg = torch.zeros(batch_size, 3, device=device)
                R_agg = torch.eye(3, device=device).repeat(batch_size, 1, 1)
                tor_agg = torch.zeros_like(batch.ligand.init_tor, device=device)
            else:
                # Normal alignment process
                tr_aligned = torch.zeros_like(tr_agg, device=device)
                rot_aligned = torch.eye(3, device=device).repeat(
                    tr_agg.shape[0], 1, 1)
                apply_tor_changes_to_batch_inplace(
                    batch, tor_agg, is_reverse_order=False)
                for i in range(len(optimized.ligand.pos)):
                    pos_pred = batch.ligand.pos[i, :optimized.ligand.num_atoms[i]]
                    pos_true = optimized.ligand.pos[i, :optimized.ligand.num_atoms[i]]

                    rot, tr = find_rigid_alignment(pos_pred, pos_true)
                    tr_aligned[i] = tr
                    rot_aligned[i] = rot

                apply_tr_rot_changes_to_batch_inplace(
                    batch, tr_aligned, rot_aligned)
                tr_agg = tr_aligned
                R_agg = rot_aligned

            tr_agg_init_coord = torch.bmm(
                tr_agg[:, None, :], optimized.original_augm_rot)[:, 0]

            transformed_orig = torch.einsum(
                'bij,bjk->bik', optimized.ligand.pos, optimized.original_augm_rot)
            transformed_orig.masked_fill_(pad_mask.unsqueeze(-1), 0.)

            all_names = batch.names
            num_atoms_list = optimized.ligand.num_atoms.detach().cpu().tolist()
            num_rot_bonds_list = optimized.ligand.num_rotatable_bonds.detach().cpu().tolist()
            orig_pos_before_augm_cpu = optimized.ligand.orig_pos_before_augm.detach().cpu().numpy()
            transformed_orig_cpu = transformed_orig.detach().cpu().numpy()
            tr_agg_init_coord_cpu = tr_agg_init_coord.detach().cpu().numpy()
            rot_pred_cpu = R_agg.detach().cpu().numpy()
            rot_augm_cpu = optimized.original_augm_rot.detach().cpu().numpy()
            full_protein_center_cpu = optimized.protein.full_protein_center.detach().cpu().numpy()
            rot_bonds_ext_cpu = {
                'start': optimized.ligand.rotatable_bonds_ext.start.detach().cpu().numpy(),
                'end': optimized.ligand.rotatable_bonds_ext.end.detach().cpu().numpy(),
                'neighbor_of_start': optimized.ligand.rotatable_bonds_ext.neighbor_of_start.detach().cpu().numpy(),
                'neighbor_of_end': optimized.ligand.rotatable_bonds_ext.neighbor_of_end.detach().cpu().numpy(),
                'bond_periods': optimized.ligand.rotatable_bonds_ext.bond_periods.detach().cpu().numpy(),
            }

            for full_idx, name in enumerate(all_names):
                sample_num_atoms = num_atoms_list[full_idx]
                sample_num_rot_bonds = num_rot_bonds_list[full_idx]

                complex_metrics = {}
                complex_metrics['orig_pos_before_augm'] = orig_pos_before_augm_cpu[full_idx, :sample_num_atoms]
                complex_metrics['transformed_orig'] = transformed_orig_cpu[full_idx, :sample_num_atoms]
                complex_metrics['tr_pred_init'] = tr_agg_init_coord_cpu[full_idx]
                complex_metrics['rot_pred'] = rot_pred_cpu[full_idx]
                complex_metrics['rot_augm'] = rot_augm_cpu[full_idx]
                complex_metrics['full_protein_center'] = full_protein_center_cpu[full_idx]

                bond_properties_for_angles = {
                    'start': rot_bonds_ext_cpu['start'][full_idx, :sample_num_rot_bonds],
                    'end': rot_bonds_ext_cpu['end'][full_idx, :sample_num_rot_bonds],
                    'neighbor_of_start': rot_bonds_ext_cpu['neighbor_of_start'][full_idx, :sample_num_rot_bonds],
                    'neighbor_of_end': rot_bonds_ext_cpu['neighbor_of_end'][full_idx, :sample_num_rot_bonds],
                    'bond_periods': rot_bonds_ext_cpu['bond_periods'][full_idx, :sample_num_rot_bonds],
                }

                torsion_angles_pred = get_torsion_angles(
                    transformed_orig_cpu[full_idx, :sample_num_atoms],
                    bond_atoms_for_angles=bond_properties_for_angles,
                )
                complex_metrics['torsion_angles_pred'] = torsion_angles_pred

                metrics_dict.setdefault(name, []).append(complex_metrics)

            if batch_timings is not None:
                batch_timings.append(time.perf_counter() - batch_start_time)
    return metrics_dict
