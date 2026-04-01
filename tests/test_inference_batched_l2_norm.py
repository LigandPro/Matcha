import numpy as np
import torch

from matcha.utils.inference import _batched_l2_norm


def test_batched_l2_norm_handles_empty_flat_torsion_tensor():
    value = torch.zeros(0, dtype=torch.float32)

    result = _batched_l2_norm(value)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (0,)


def test_batched_l2_norm_handles_empty_per_batch_torsion_tensor():
    value = torch.zeros(3, 0, dtype=torch.float32)

    result = _batched_l2_norm(value)

    np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))
