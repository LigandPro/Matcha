import numpy as np
import pytest


def test_find_rigid_alignment_rejects_non_finite_input():
    from matcha.utils.transforms import RigidAlignmentError, find_rigid_alignment

    pos_a = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    pos_b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    with pytest.raises(RigidAlignmentError, match="non-finite numpy coordinates"):
        find_rigid_alignment(pos_a, pos_b)


def test_find_rigid_alignment_svd_fails_fast(monkeypatch):
    from matcha.utils import transforms

    calls = {"n": 0}

    def flaky_svd(*args, **kwargs):
        calls["n"] += 1
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "svd", flaky_svd)

    pos_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    pos_b = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)

    with pytest.raises(transforms.RigidAlignmentError, match="SVD failed for numpy input"):
        transforms.find_rigid_alignment(pos_a, pos_b)

    assert calls["n"] == 1
