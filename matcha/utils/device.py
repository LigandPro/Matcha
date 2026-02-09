"""Centralized device resolution: CUDA > MPS > CPU."""

import torch


def resolve_device(requested: str | None = None) -> str:
    """Return the best available torch device string.

    Priority: explicit request > CUDA > MPS (Apple Metal) > CPU.
    """
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
