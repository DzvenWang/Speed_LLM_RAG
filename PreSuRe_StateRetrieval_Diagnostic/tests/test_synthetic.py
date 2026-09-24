import numpy as np
from srd.transport import raw_copy, mean_residual, zscore, delta_anchor, restoration_scope_mask


def _q():
    x = np.array([[1., 2.], [0., 3.], [3., 0.], [4., 0.]], np.float32)
    obs = np.array([[1, 1], [0, 1], [1, 0], [1, 0]], bool)
    return {"x_obs": x, "history_observed": obs, "restorable": ~obs}


def _c():
    x = np.array([[10., 20.], [11., 21.], [12., 22.], [13., 23.]], np.float32)
    return {"x_full": x, "full_valid": np.ones_like(x, bool)}


def main():
    q, c = _q(), _c()
    cfg = {"transport": {"eps": 1e-6, "scale_clip": [0.5, 2.0], "restoration_scope": "recent_suffix", "recent_steps": 2}}
    sm = restoration_scope_mask((4, 2), cfg)
    assert sm[:2].sum() == 0 and sm[2:].all()
    for fn in [raw_copy, mean_residual, zscore, delta_anchor]:
        x, f = fn(q, c, cfg)
        assert x.shape == (4, 2)
        # Only the two recent restorable cells can be modified.
        assert f.sum() == 2
        assert not f[:2].any()
        assert np.isfinite(x).all()
    print("synthetic recent-scope transport tests PASS")


if __name__ == "__main__":
    main()
