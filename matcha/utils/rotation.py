import torch


def skew_symmetric(omega):
    """
    Convert a batch of angular velocities to skew-symmetric matrices.
    omega: tensor of shape (batch_size, 3)
    Returns a tensor of shape (batch_size, 3, 3)
    """
    batch_size = omega.shape[0]
    Omega = torch.zeros((batch_size, 3, 3), device=omega.device)
    Omega[:, 0, 1] = -omega[:, 2]
    Omega[:, 0, 2] = omega[:, 1]
    Omega[:, 1, 0] = omega[:, 2]
    Omega[:, 1, 2] = -omega[:, 0]
    Omega[:, 2, 0] = -omega[:, 1]
    Omega[:, 2, 1] = omega[:, 0]
    return Omega


def expm_SO3(Omega, dt):
    """
    Compute the matrix exponential expm(Omega * dt) for a batch of Omega matrices using Rodrigues' formula.
    Omega: tensor of shape (batch_size, 3)
    dt: scalar time step
    Returns a tensor of shape (batch_size, 3, 3)
    """
    theta = torch.norm(Omega, dim=1)[:, None, None]
    Omega_hat = skew_symmetric(Omega)
    Omega_normalized = Omega_hat / (theta + 1e-8)

    I = torch.eye(3, device=Omega.device).unsqueeze(
        0)  # Identity matrix of shape (1, 3, 3)
    A = Omega_normalized * torch.sin(theta * dt)
    B = torch.bmm(Omega_normalized, Omega_normalized) * \
        (1 - torch.cos(theta * dt))

    expm_Omega_dt = I + A + B
    return expm_Omega_dt
