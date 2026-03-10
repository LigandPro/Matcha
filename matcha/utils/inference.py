import copy
import inspect
import os

import numpy as np
import safetensors
import torch
from tqdm import tqdm

from matcha.utils.log import get_logger
from matcha.utils.rotation import expm_SO3
from matcha.utils.transforms import (apply_tr_rot_changes_to_batch_inplace, apply_tor_changes_to_batch_inplace,
                                       get_torsion_angles, find_rigid_alignment, RigidAlignmentError)

logger = get_logger(__name__)

def load_from_checkpoint(model, checkpoint_path, strict=True):
    checkpoint_path = os.path.expanduser(checkpoint_path)
    state_dict = safetensors.torch.load_file(os.path.join(
        checkpoint_path, 'model.safetensors'), device="cpu")
    model.load_state_dict(state_dict, strict=strict)
    return model


def euler(
    model,
    batch,
    device,
    num_steps=20,
    record_history: bool = True,
    record_trajectory: bool = True,
    forward_step_fn=None,
    copy_batch: bool = True,
):
    cur_batch = copy.deepcopy(batch) if copy_batch else batch
    if copy_batch:
        cur_batch = cur_batch.to(device)
    h = 1. / num_steps
    batch_size = len(cur_batch)
    R_eye = torch.eye(3, device=device).repeat(batch_size, 1, 1)
    R_agg = cur_batch.ligand.init_rot
    tr = cur_batch.ligand.init_tr
    tor = cur_batch.ligand.init_tor
    cur_batch.ligand.t = torch.zeros_like(cur_batch.ligand.t)

    pos_hist = []
    trajectory = []
    if record_trajectory:
        trajectory = [cur_batch.ligand.pos.detach().cpu().numpy().copy()]
    tor_agg = torch.zeros_like(tor)
    for step in range(num_steps):
        with torch.no_grad():
            step_fn = forward_step_fn or model.forward_step
            dtr, drot, dtor, _ = step_fn(cur_batch)

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

            if record_history:
                pos_hist.append(cur_batch.ligand.pos.detach().cpu().numpy().copy())
            if record_trajectory:
                trajectory.append(cur_batch.ligand.pos.detach().cpu().numpy().copy())

    tr_agg = tr
    return cur_batch, tr_agg, R_agg, tor_agg, pos_hist, trajectory


def run_evaluation(
    dataloader,
    num_steps,
    solver,
    model,
    progress_callback=None,
    current_stage=None,
    device: str | None = None,
    compute_torsion_angles_pred: bool = False,
    solver_kwargs: dict | None = None,
    profiler=None,
):
    def revert_augm(batch):
        batch.ligand.pos[:] = torch.einsum(
            'bij,bjk->bik', batch.ligand.pos, batch.original_augm_rot)
        for idx, num_atoms in enumerate(batch.ligand.num_atoms):
            batch.ligand.pos[idx, num_atoms:] = 0.
        return batch.ligand.pos

    if device is None:
        from matcha.utils.device import resolve_device
        device = resolve_device()
    metrics_dict = {}
    total_batches = len(dataloader)
    solver_signature = inspect.signature(solver)
    solver_supports_copy_batch = "copy_batch" in solver_signature.parameters
    use_non_blocking = torch.cuda.is_available() and str(device).startswith("cuda")

    for loader_idx, batch in enumerate(tqdm(dataloader, desc="Docking inference")):
        batch = batch["batch"]
        batch_size = len(batch)
        solver_kwargs_local = dict(solver_kwargs or {})

        if solver_supports_copy_batch and hasattr(batch, "clone_structure"):
            solver_kwargs_local.pop("copy_batch", None)
            solver_batch = batch.clone_structure().to(device, non_blocking=use_non_blocking)
            optimized, tr_agg, R_agg, tor_agg, _, _ = solver(
                model,
                solver_batch,
                device=device,
                num_steps=num_steps,
                copy_batch=False,
                **solver_kwargs_local,
            )
            aligned_batch = batch.clone_structure().to(device, non_blocking=use_non_blocking)
            init_batch = batch.clone_structure().to(device, non_blocking=use_non_blocking)
        else:
            optimized, tr_agg, R_agg, tor_agg, _, _ = solver(
                model, batch, device=device, num_steps=num_steps, **solver_kwargs_local
            )
            aligned_batch = None
            init_batch = None

        for elem_idx, num_atoms in enumerate(optimized.ligand.num_atoms):
            optimized.ligand.pos[elem_idx, num_atoms:] = 0.

        # Alignment process
        if tr_agg is None or R_agg is None or tor_agg is None:
            tr_agg = torch.zeros(batch_size, 3, device=device)
            R_agg = torch.eye(3, device=device).repeat(batch_size, 1, 1)
            tor_agg = torch.zeros_like(batch.ligand.init_tor, device=device)
            if aligned_batch is None:
                aligned_batch = copy.deepcopy(batch).to(device)
        else:
            if aligned_batch is None:
                aligned_batch = copy.deepcopy(batch).to(device)
            tr_aligned = torch.zeros_like(tr_agg, device=device)
            rot_aligned = torch.eye(3, device=device).repeat(
                tr_agg.shape[0], 1, 1)
            apply_tor_changes_to_batch_inplace(
                aligned_batch, tor_agg, is_reverse_order=False)
            for i in range(len(optimized.ligand.pos)):
                pos_pred = aligned_batch.ligand.pos[i, :optimized.ligand.num_atoms[i]]
                pos_true = optimized.ligand.pos[i, :optimized.ligand.num_atoms[i]]

                try:
                    rot, tr = find_rigid_alignment(pos_pred, pos_true)
                except RigidAlignmentError as exc:
                    raise RigidAlignmentError(
                        f"Rigid alignment failed for inference target {batch.names[i]}: {exc}"
                    ) from exc
                tr_aligned[i] = tr
                rot_aligned[i] = rot

            apply_tr_rot_changes_to_batch_inplace(
                aligned_batch, tr_aligned, rot_aligned)
            tr_agg = tr_aligned
            R_agg = rot_aligned

        # Handle tr_agg_init_coord computation
        tr_agg_init_coord = torch.bmm(
            (tr_agg)[:, None, :], optimized.original_augm_rot)[:, 0]

        if init_batch is None:
            init_batch = copy.deepcopy(batch).to(device)
        init_batch.ligand.pos = optimized.ligand.pos.clone().to(device)
        transformed_orig = revert_augm(init_batch)

        all_names = batch.names

        for full_idx, name in enumerate(all_names):
            sample_idx = full_idx % len(batch.names)
            complex_metrics = {}
            complex_metrics['orig_pos_before_augm'] = optimized.ligand.orig_pos_before_augm[sample_idx, :optimized.ligand.num_atoms[sample_idx]].cpu().numpy()
            complex_metrics['transformed_orig'] = transformed_orig[full_idx, :optimized.ligand.num_atoms[sample_idx]].cpu().numpy()

            complex_metrics['tr_pred_init'] = tr_agg_init_coord[full_idx].cpu().numpy()
            complex_metrics['rot_pred'] = R_agg[full_idx].cpu().numpy()
            complex_metrics['rot_augm'] = optimized.original_augm_rot[sample_idx].cpu().numpy()
            complex_metrics['full_protein_center'] = optimized.protein.full_protein_center[sample_idx].cpu().numpy()

            # compute torsion angles
            if compute_torsion_angles_pred:
                rot_bonds = optimized.ligand.rotatable_bonds_ext
                n_rot = optimized.ligand.num_rotatable_bonds[sample_idx]
                bond_properties_for_angles = {
                    'start': rot_bonds.start[sample_idx, :n_rot],
                    'end': rot_bonds.end[sample_idx, :n_rot],
                    'neighbor_of_start': rot_bonds.neighbor_of_start[sample_idx, :n_rot],
                    'neighbor_of_end': rot_bonds.neighbor_of_end[sample_idx, :n_rot],
                    'bond_periods': rot_bonds.bond_periods[sample_idx, :n_rot],
                }
                torsion_angles_pred = get_torsion_angles(
                    torch.from_numpy(np.copy(complex_metrics['transformed_orig'])).to(device),
                    bond_atoms_for_angles=bond_properties_for_angles,
                )
                complex_metrics['torsion_angles_pred'] = torsion_angles_pred.cpu().numpy()

            metrics_dict.setdefault(name, []).append(complex_metrics)

        # Update progress for TUI
        if progress_callback is not None and current_stage is not None:
            progress_percent = int((loader_idx + 1) / total_batches * 100)
            progress_callback('stage_progress', current_stage, None, None, progress_percent)

        if profiler is not None:
            profiler.step()

    return metrics_dict
