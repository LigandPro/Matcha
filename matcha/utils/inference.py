import os
import numpy as np
import copy
from copy import deepcopy
import safetensors
from tqdm import tqdm
import torch
from matcha.utils.rotation import expm_SO3
from matcha.utils.transforms import (apply_tr_rot_changes_to_batch_inplace, apply_tor_changes_to_batch_inplace,
                                       get_torsion_angles, find_rigid_alignment)
from matcha.utils.log import get_logger

logger = get_logger(__name__)


def _to_numpy_or_none(value):
    if value is None:
        return None
    return value.detach().cpu().numpy().copy()


def _batched_l2_norm(value):
    if value is None:
        return None
    flat = value.reshape(value.shape[0], -1)
    return torch.linalg.norm(flat, dim=1).detach().cpu().numpy().copy()

def load_from_checkpoint(model, checkpoint_path, strict=True):
    checkpoint_path = os.path.expanduser(checkpoint_path)
    state_dict = safetensors.torch.load_file(os.path.join(
        checkpoint_path, 'model.safetensors'), device="cpu")
    model.load_state_dict(state_dict, strict=strict)
    return model


def euler(model, batch, device, num_steps=20):
    cur_batch = deepcopy(batch).to(device)
    h = 1. / num_steps
    batch_size = len(cur_batch)
    R_eye = torch.eye(3, device=device).repeat(batch_size, 1, 1)
    R_agg = cur_batch.ligand.init_rot
    tr = cur_batch.ligand.init_tr
    tor = cur_batch.ligand.init_tor
    cur_batch.ligand.t = torch.zeros_like(cur_batch.ligand.t)

    pos_hist = []
    trajectory = [cur_batch.ligand.pos.detach().cpu().numpy().copy()]
    tor_agg = torch.zeros_like(tor)
    trajectory_details = [{
        "frame": 0,
        "step": 0,
        "time": 0.0,
        "delta_translation": np.zeros((batch_size, 3), dtype=np.float32),
        "delta_rotation": np.zeros((batch_size, 3), dtype=np.float32),
        "delta_torsion": np.zeros_like(tor_agg.detach().cpu().numpy(), dtype=np.float32),
        "delta_translation_norm": np.zeros(batch_size, dtype=np.float32),
        "delta_rotation_norm": np.zeros(batch_size, dtype=np.float32),
        "delta_torsion_norm": np.zeros(batch_size, dtype=np.float32),
        "translation": tr.detach().cpu().numpy().copy(),
        "rotation": R_agg.detach().cpu().numpy().copy(),
        "torsion": tor_agg.detach().cpu().numpy().copy(),
        "translation_norm": _batched_l2_norm(tr),
        "torsion_norm": _batched_l2_norm(tor_agg),
    }]
    for step in range(num_steps):
        with torch.no_grad():
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

            pos_hist.append(cur_batch.ligand.pos.detach().cpu().numpy().copy())
            trajectory.append(cur_batch.ligand.pos.detach().cpu().numpy().copy())
            trajectory_details.append({
                "frame": step + 1,
                "step": step + 1,
                "time": float(cur_batch.ligand.t[0].item()),
                "delta_translation": _to_numpy_or_none(dtr),
                "delta_rotation": _to_numpy_or_none(drot),
                "delta_torsion": _to_numpy_or_none(dtor),
                "delta_translation_norm": _batched_l2_norm(dtr),
                "delta_rotation_norm": _batched_l2_norm(drot),
                "delta_torsion_norm": _batched_l2_norm(dtor),
                "translation": _to_numpy_or_none(tr),
                "rotation": _to_numpy_or_none(R_agg),
                "torsion": _to_numpy_or_none(tor_agg),
                "translation_norm": _batched_l2_norm(tr),
                "torsion_norm": _batched_l2_norm(tor_agg),
            })

    tr_agg = tr
    return cur_batch, tr_agg, R_agg, tor_agg, pos_hist, trajectory, trajectory_details


def run_evaluation(
    dataloader,
    num_steps,
    solver,
    model,
    device=None,
    compute_torsion_angles_pred=True,
    solver_kwargs=None,
    capture_trajectory=False,
):
    del compute_torsion_angles_pred

    def revert_augm(batch):
        batch.ligand.pos[:] = torch.einsum(
            'bij,bjk->bik', batch.ligand.pos, batch.original_augm_rot)
        for batch_idx, num_atoms in enumerate(batch.ligand.num_atoms):
            batch.ligand.pos[batch_idx, num_atoms:] = 0.
        return batch.ligand.pos

    if solver_kwargs is None:
        solver_kwargs = {}
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics_dict = {}
    for batch in tqdm(dataloader, desc="Docking inference"):
        batch = batch['batch']
        batch_size = len(batch)
        solver_output = solver(model, batch, device=device, num_steps=num_steps, **solver_kwargs)
        if len(solver_output) == 6:
            optimized, tr_agg, R_agg, tor_agg, pos_hist, trajectory = solver_output
            trajectory_details = None
        else:
            optimized, tr_agg, R_agg, tor_agg, pos_hist, trajectory, trajectory_details = solver_output

        # compute RMSD in case of zero-padded batches
        num_batch_atoms = (~batch.ligand.is_padded_mask).sum(dim=1).to(device)
        num_atoms_cpu = optimized.ligand.num_atoms.detach().cpu().numpy()
        for batch_idx, num_atoms in enumerate(optimized.ligand.num_atoms):
            optimized.ligand.pos[batch_idx, num_atoms:] = 0.
            for i in range(len(pos_hist)):
                pos_hist[i][batch_idx, num_atoms:] = 0.

        trajectory_frames = None
        if capture_trajectory:
            original_augm_rot = optimized.original_augm_rot.detach().cpu().numpy()
            trajectory_frames = []
            for frame in trajectory:
                reverted = np.einsum('bij,bjk->bik', frame, original_augm_rot)
                for batch_idx, num_atoms in enumerate(num_atoms_cpu):
                    reverted[batch_idx, int(num_atoms):] = 0.0
                trajectory_frames.append(reverted)

        # Handle cases where solver returns None for aggregated values
        if tr_agg is None or R_agg is None or tor_agg is None:
            # Skip alignment and use identity transformations
            tr_agg = torch.zeros(batch_size, 3, device=device)
            R_agg = torch.eye(3, device=device).repeat(batch_size, 1, 1)
            tor_agg = torch.zeros_like(batch.ligand.init_tor, device=device)
            aligned_batch = copy.deepcopy(batch).to(device)
        else:
            # Normal alignment process
            aligned_batch = copy.deepcopy(batch).to(device)
            tr_aligned = torch.zeros_like(tr_agg, device=device)
            rot_aligned = torch.eye(3, device=device).repeat(
                tr_agg.shape[0], 1, 1)
            apply_tor_changes_to_batch_inplace(
                aligned_batch, tor_agg, is_reverse_order=False)
            for i in range(len(optimized.ligand.pos)):
                pos_pred = aligned_batch.ligand.pos[i, :optimized.ligand.num_atoms[i]]
                pos_true = optimized.ligand.pos[i, :optimized.ligand.num_atoms[i]]

                rot, tr = find_rigid_alignment(pos_pred, pos_true)
                tr_aligned[i] = tr
                rot_aligned[i] = rot

            apply_tr_rot_changes_to_batch_inplace(
                aligned_batch, tr_aligned, rot_aligned)
            tr_agg = tr_aligned
            R_agg = rot_aligned

        # Handle tr_agg_init_coord computation
        tr_agg_init_coord = torch.bmm(
            (tr_agg)[:, None, :], optimized.original_augm_rot)[:, 0]

        init_batch = copy.deepcopy(batch).to(device)
        init_batch.ligand.pos = optimized.ligand.pos.clone().to(device)
        transformed_orig = revert_augm(init_batch)

        tr_true_init = torch.bmm((optimized.ligand.final_tr)[:, None, :], optimized.original_augm_rot)[:, 0]

        all_names = batch.names
        tor_true = optimized.ligand.final_tor.cpu().numpy()

        compute_metrics = True
        tor_pred = tor_agg.cpu().numpy()

        for full_idx, name in enumerate(all_names):
            batch_idx = full_idx % len(batch.names)
            complex_metrics = {}
            complex_metrics['orig_pos_before_augm'] = optimized.ligand.orig_pos_before_augm[batch_idx, :optimized.ligand.num_atoms[batch_idx]].cpu().numpy()
            complex_metrics['transformed_orig'] = transformed_orig[full_idx, :optimized.ligand.num_atoms[batch_idx]].cpu().numpy()

            # Handle cases where aggregated values might be None
            if tr_agg is not None:
                complex_metrics['tr_pred_init'] = tr_agg_init_coord[full_idx].cpu().numpy()
            else:
                complex_metrics['tr_pred_init'] = np.zeros(3)

            if R_agg is not None:
                complex_metrics['rot_pred'] = R_agg[full_idx].cpu().numpy()
            else:
                complex_metrics['rot_pred'] = np.eye(3)
            complex_metrics['rot_augm'] = optimized.original_augm_rot[batch_idx].cpu().numpy()
            complex_metrics['full_protein_center'] = optimized.protein.full_protein_center[batch_idx].cpu().numpy()
            if trajectory_frames is not None:
                sample_trajectory = []
                for frame_index, frame in enumerate(trajectory_frames):
                    frame_entry = {
                        'positions': np.copy(frame[batch_idx, :int(num_atoms_cpu[batch_idx])]),
                    }
                    if trajectory_details is not None and frame_index < len(trajectory_details):
                        detail = trajectory_details[frame_index]
                        frame_entry.update({
                            'frame': int(detail['frame']),
                            'step': int(detail['step']),
                            'time': float(detail['time']),
                            'delta_translation': np.copy(detail['delta_translation'][batch_idx]) if detail['delta_translation'] is not None else None,
                            'delta_rotation': np.copy(detail['delta_rotation'][batch_idx]) if detail['delta_rotation'] is not None else None,
                            'delta_torsion': np.copy(detail['delta_torsion'][batch_idx]) if detail['delta_torsion'] is not None else None,
                            'delta_translation_norm': float(detail['delta_translation_norm'][batch_idx]) if detail['delta_translation_norm'] is not None else None,
                            'delta_rotation_norm': float(detail['delta_rotation_norm'][batch_idx]) if detail['delta_rotation_norm'] is not None else None,
                            'delta_torsion_norm': float(detail['delta_torsion_norm'][batch_idx]) if detail['delta_torsion_norm'] is not None else None,
                            'translation': np.copy(detail['translation'][batch_idx]) if detail['translation'] is not None else None,
                            'rotation': np.copy(detail['rotation'][batch_idx]) if detail['rotation'] is not None else None,
                            'torsion': np.copy(detail['torsion'][batch_idx]) if detail['torsion'] is not None else None,
                            'translation_norm': float(detail['translation_norm'][batch_idx]) if detail['translation_norm'] is not None else None,
                            'torsion_norm': float(detail['torsion_norm'][batch_idx]) if detail['torsion_norm'] is not None else None,
                        })
                    sample_trajectory.append(frame_entry)
                complex_metrics['trajectory'] = sample_trajectory

            # compute torsion angles
            num_rotatable = int(optimized.ligand.num_rotatable_bonds[batch_idx])
            if num_rotatable > 0:
                bond_properties_for_angles = {}
                bond_properties_for_angles['start'] = optimized.ligand.rotatable_bonds_ext.start[batch_idx,
                                                                                                 :optimized.ligand.num_rotatable_bonds[batch_idx]]
                bond_properties_for_angles['end'] = optimized.ligand.rotatable_bonds_ext.end[batch_idx,
                                                                                             :optimized.ligand.num_rotatable_bonds[batch_idx]]
                bond_properties_for_angles['neighbor_of_start'] = optimized.ligand.rotatable_bonds_ext.neighbor_of_start[batch_idx,
                                                                                                                         :optimized.ligand.num_rotatable_bonds[batch_idx]]
                bond_properties_for_angles['neighbor_of_end'] = optimized.ligand.rotatable_bonds_ext.neighbor_of_end[batch_idx,
                                                                                                                     :optimized.ligand.num_rotatable_bonds[batch_idx]]
                bond_properties_for_angles['bond_periods'] = optimized.ligand.rotatable_bonds_ext.bond_periods[batch_idx,
                                                                                                               :optimized.ligand.num_rotatable_bonds[batch_idx]]

                torsion_angles_pred = get_torsion_angles(torch.from_numpy(np.copy(complex_metrics['transformed_orig'])).to(device),
                                                         bond_atoms_for_angles=bond_properties_for_angles)
                complex_metrics['torsion_angles_pred'] = torsion_angles_pred.cpu().numpy()
            else:
                complex_metrics['torsion_angles_pred'] = np.zeros(0, dtype=np.float32)

            metrics_dict[name] = metrics_dict.get(name, []) + [complex_metrics]
    return metrics_dict
