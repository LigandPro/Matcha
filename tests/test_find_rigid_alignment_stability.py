import numpy as np


def test_find_rigid_alignment_nan_fallback():
    from matcha.utils.transforms import find_rigid_alignment

    pos_a = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    pos_b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    rot, tr = find_rigid_alignment(pos_a, pos_b)
    assert rot.shape == (3, 3)
    assert tr.shape == (3,)
    assert np.isfinite(rot).all()
    assert np.isfinite(tr).all()
    assert np.allclose(rot, np.eye(3, dtype=np.float32))


def test_find_rigid_alignment_svd_retry(monkeypatch):
    from matcha.utils import transforms

    original_svd = np.linalg.svd
    calls = {"n": 0}

    def flaky_svd(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise np.linalg.LinAlgError("SVD did not converge")
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", flaky_svd)

    pos_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    pos_b = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)

    rot, tr = transforms.find_rigid_alignment(pos_a, pos_b)
    assert calls["n"] == 2
    assert rot.shape == (3, 3)
    assert tr.shape == (3,)
    assert np.isfinite(rot).all()
    assert np.isfinite(tr).all()

