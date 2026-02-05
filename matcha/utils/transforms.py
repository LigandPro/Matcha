import numpy as np
import torch


def compute_batch_ligand_centers(batch):
    """
    Compute the mean positions of ligands in a batch.

    Parameters:
    ----------
    batch : ComplexBatch

    Returns:
    -------
    torch.Tensor
        Mean positions of the ligands in the batch. Shape: (batch_size, 3)
    """
    # We assume that zero-padding is maintained
    ligand_centers = batch.ligand.pos.sum(
        axis=1) / batch.ligand.num_atoms[:, None]
    return ligand_centers


def rotvec_to_rotmat(rotvec):
    """
    Converts a batch of rotation vectors to rotation matrices using the Rodrigues' rotation formula.

    Args:
        rotvec (torch.Tensor): A tensor of shape (batch_size, 3) representing a batch of rotation vectors.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 3, 3) representing a batch of rotation matrices.
    """
    # Compute the norm (theta) for each rotation vector
    theta = torch.norm(rotvec, dim=-1, keepdim=True)  # Shape: (batch_size, 1)
    n = torch.nn.functional.normalize(rotvec, dim=-1)

    # Extract n1, n2, n3 from normalized vectors
    n1, n2, n3 = n[0], n[1], n[2]

    # Precompute trigonometric terms
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    one_minus_cos_theta = 1 - cos_theta

    # Compute the rotation matrix using the formula
    r11 = cos_theta + n1**2 * one_minus_cos_theta
    r12 = n1 * n2 * one_minus_cos_theta - n3 * sin_theta
    r13 = n1 * n3 * one_minus_cos_theta + n2 * sin_theta

    r21 = n1 * n2 * one_minus_cos_theta + n3 * sin_theta
    r22 = cos_theta + n2**2 * one_minus_cos_theta
    r23 = n2 * n3 * one_minus_cos_theta - n1 * sin_theta

    r31 = n1 * n3 * one_minus_cos_theta - n2 * sin_theta
    r32 = n2 * n3 * one_minus_cos_theta + n1 * sin_theta
    r33 = cos_theta + n3**2 * one_minus_cos_theta

    # Stack the components to form the rotation matrix
    rotation_matrix = torch.stack(
        [r11, r12, r13, r21, r22, r23, r31, r32, r33], dim=-1)
    rotation_matrix = rotation_matrix.view(
        3, 3)  # Reshape to (batch_size, 3, 3)

    return rotation_matrix


def apply_tor_changes_to_pos(pos, rotatable_bonds, mask_rotate, torsion_updates, is_reverse_order,
                             shift_center_back=True):
    """
    Apply torsion updates to the positions of atoms in a sample in-place.

    Parameters:
    ----------
    pos : Union[np.ndarray, torch.Tensor]
        The positions of atoms in the sample, shape (num_atoms, 3).
    rotatable_bonds : Union[np.ndarray, torch.Tensor]
        Rotatable bonds in the sample, shape (num_rotatable_bonds, 2). Each bond is represented by
        two indices: (atom1, atom2).
    mask_rotate : Union[np.ndarray, torch.Tensor]
        Mask indicating which atoms to rotate for each bond, shape (num_rotatable_bonds, num_atoms).
    torsion_updates : Union[np.ndarray, torch.Tensor]
        Torsion updates to apply to each rotatable bond, shape (num_rotatable_bonds,).

    Returns:
    -------
    Union[np.ndarray, torch.Tensor]
        The updated positions of atoms in the sample.
    """
    is_torch = isinstance(pos, torch.Tensor)

    if len(rotatable_bonds) == 0:
        return pos

    if rotatable_bonds.shape[1] != 2:
        raise ValueError('A wrong format of rotational bonds array!')

    num_rotatable_bonds = rotatable_bonds.shape[0]

    if is_reverse_order:
        range_for_rot_bonds = range(num_rotatable_bonds - 1, -1, -1)
    else:
        range_for_rot_bonds = range(num_rotatable_bonds)

    # compute initial ligand center
    pos_mean = pos.mean(0)[None, :]

    for idx_rot_bond in range_for_rot_bonds:
        u = rotatable_bonds[idx_rot_bond, 0]
        v = rotatable_bonds[idx_rot_bond, 1]
        # convention: positive rotation if pointing inwards
        rot_vec = pos[u] - pos[v]

        if is_torch:
            # Rotate v:
            rot_vec = rot_vec * \
                torsion_updates[idx_rot_bond] / torch.linalg.norm(rot_vec)
            rot_mat = rotvec_to_rotmat(rot_vec)
            mask = mask_rotate[idx_rot_bond].bool()
            pos[mask] = (pos[mask] - pos[v]) @ rot_mat.T + pos[v]
        else:
            # Rotate v:
            rot_vec = rot_vec * \
                torsion_updates[idx_rot_bond] / np.linalg.norm(rot_vec)
            rot_mat = rotvec_to_rotmat(torch.tensor(
                rot_vec, dtype=torch.float)).numpy()
            mask = mask_rotate[idx_rot_bond].astype(bool)
            pos[mask] = (pos[mask] - pos[v]) @ rot_mat.T + pos[v]

    # shift to the initial center
    if shift_center_back:
        pos = pos - pos.mean(0)[None, :] + pos_mean
    return pos


def apply_tor_changes_to_batch_inplace(batch, tor, is_reverse_order):
    """
    Apply torsion updates to each ligand in the batch.

    Parameters:
    ----------
    batch : Batch
        The batch containing ligand information.
    tor : np.ndarray
        Torsion updates to apply to each rotatable bond, shape (num_rotatable_bonds,).

    Returns:
    -------
    Batch
        The batch with updated positions of atoms.
    """
    left_rot_bond_idx = 0
    # TODO: vectorize
    for idx, mask_rotate in enumerate(batch.ligand.mask_rotate):
        pos = batch.ligand.pos[idx, :batch.ligand.num_atoms[idx], :]
        right_rot_bond_idx = left_rot_bond_idx + \
            batch.ligand.num_rotatable_bonds[idx]
        rotatable_bonds = batch.ligand.rotatable_bonds[left_rot_bond_idx:right_rot_bond_idx]
        torsion_updates = tor[left_rot_bond_idx:right_rot_bond_idx]
        left_rot_bond_idx = right_rot_bond_idx
        pos = apply_tor_changes_to_pos(pos, rotatable_bonds, mask_rotate, torsion_updates,
                                       is_reverse_order=is_reverse_order)
        batch.ligand.pos[idx, :batch.ligand.num_atoms[idx], :] = pos
    return


def apply_tr_rot_changes_to_batch_inplace(batch, tr, rot):
    batch_size = tr.shape[0]
    pos_mean = compute_batch_ligand_centers(batch)
    '''
    Here we do not add pos_mean, because tr is the new center of mass!
    So, new_pos = (pos - pos_mean) @ rot.T + tr
    '''
    batch.ligand.pos[:] = torch.einsum('bij,bkj->bik', batch.ligand.pos - pos_mean[:, None, :],
                                       rot) + tr.reshape(batch_size, 1, 3)
    # TODO: vectorize
    for batch_idx, num_atoms in enumerate(batch.ligand.num_atoms):
        batch.ligand.pos[batch_idx, num_atoms:] = 0.

    return


def compute_signed_dihedral_angles_vectorized(positions):
    """
    Compute signed dihedral angles for multiple sets of atoms in a vectorized way.

    Parameters:
    -----------
    positions: np.ndarray or torch.Tensor, shape (B, 4, 3)
        Positions of B sets of 4 atoms each: (n0, start, end, n1) in 3D space.

    Returns:
    --------
    angles: np.ndarray or torch.Tensor, shape (B,)
        The signed dihedral angles in radians.
    """
    # Extract the positions of each atom in each set
    p0 = positions[:, 0]  # n0 positions
    p1 = positions[:, 1]  # start positions
    p2 = positions[:, 2]  # end positions
    p3 = positions[:, 3]  # n1 positions

    # Calculate bond vectors
    b1 = p1 - p0  # n0 -> start
    b2 = p2 - p1  # start -> end
    b3 = p3 - p2  # end -> n1

    # Normalize b2
    if isinstance(positions, torch.Tensor):
        b2_normalized = b2 / torch.norm(b2, dim=1, keepdim=True)

        # Calculate normal vectors to the planes
        n1 = torch.cross(b1, b2, dim=1)
        n1_normalized = n1 / torch.norm(n1, dim=1, keepdim=True)

        n2 = torch.cross(b2, b3, dim=1)
        n2_normalized = n2 / torch.norm(n2, dim=1, keepdim=True)

        # Calculate the orthogonal vector to n1 in the plane defined by b2
        m1 = torch.cross(n1_normalized, b2_normalized, dim=1)

        # Calculate cosine and sine
        x = torch.sum(n1_normalized * n2_normalized, dim=1)
        y = torch.sum(m1 * n2_normalized, dim=1)

        # Calculate dihedral angle
        angles = torch.atan2(y, x)
    else:
        b2_normalized = b2 / np.linalg.norm(b2, axis=1, keepdims=True)

        # Calculate normal vectors to the planes
        n1 = np.cross(b1, b2)
        n1_normalized = n1 / np.linalg.norm(n1, axis=1, keepdims=True)

        n2 = np.cross(b2, b3)
        n2_normalized = n2 / np.linalg.norm(n2, axis=1, keepdims=True)

        # Calculate the orthogonal vector to n1 in the plane defined by b2
        m1 = np.cross(n1_normalized, b2_normalized)

        # Calculate cosine and sine
        x = np.sum(n1_normalized * n2_normalized, axis=1)
        y = np.sum(m1 * n2_normalized, axis=1)

        # Calculate dihedral angle
        angles = np.arctan2(y, x)

    return -angles


def get_torsion_angles(pos, bond_atoms_for_angles):
    # Create a batch of all atom quartets for dihedral calculations
    n0 = bond_atoms_for_angles['neighbor_of_start']
    start = bond_atoms_for_angles['start']
    end = bond_atoms_for_angles['end']
    n1 = bond_atoms_for_angles['neighbor_of_end']

    # Stack the positions to form a batch
    if isinstance(pos, torch.Tensor):
        atom_quartets = torch.stack([
            pos[n0], pos[start], pos[end], pos[n1]
        ], dim=1)
    else:
        atom_quartets = np.stack([
            pos[n0], pos[start], pos[end], pos[n1]
        ], axis=1)

    # Calculate all dihedral angles at once
    angles = compute_signed_dihedral_angles_vectorized(atom_quartets)

    # Fix the angles based on bond periods
    angles = fix_torsion_angles(angles, bond_atoms_for_angles['bond_periods'])
    return angles


def fix_torsion_angles(angles, bond_periods):
    return (angles + bond_periods / 2) % bond_periods - bond_periods / 2


def get_bond_properties_for_angles(rotatable_bonds_ext):
    bond_properties_for_angles = {}
    bond_properties_for_angles['start'] = rotatable_bonds_ext.start
    bond_properties_for_angles['end'] = rotatable_bonds_ext.end
    bond_properties_for_angles['neighbor_of_start'] = rotatable_bonds_ext.neighbor_of_start
    bond_properties_for_angles['neighbor_of_end'] = rotatable_bonds_ext.neighbor_of_end
    bond_properties_for_angles['bond_periods'] = rotatable_bonds_ext.bond_periods
    return bond_properties_for_angles


def find_rigid_alignment(pos_a, pos_b):
    """
    Borrowed and slightly modified from
    https://gist.github.com/bougui505/23eb8a39d7a601399edc7534b28de3d4

    Outputs rot and tr (with fixed tor components)
    """
    a_mean = pos_a.mean(0)
    b_mean = pos_b.mean(0)
    a_centered = pos_a - a_mean
    b_centered = pos_b - b_mean
    # Covariance matrix
    cov_mat = a_centered.T @ b_centered
    if isinstance(pos_a, torch.Tensor):
        U, _, Vt = torch.linalg.svd(cov_mat)
        V = Vt.T
        det = torch.linalg.det(V @ U.T)
    else:
        U, _, Vt = np.linalg.svd(cov_mat)
        V = Vt.T
        det = np.linalg.det(V @ U.T)

    # Ensure proper rotation by checking determinant
    if det < 0:
        V[:, -1] = -V[:, -1]  # Flip the last column of V
    # Rotation matrix (now guaranteed to be proper rotation)
    rot = V @ U.T
    # Translation vector
    tr = b_mean
    return rot, tr
