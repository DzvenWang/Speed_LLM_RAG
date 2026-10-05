from __future__ import annotations

import numpy as np


def broadcast_mask(mask, target_shape, *, name: str = "mask") -> np.ndarray:
    """Return a boolean mask broadcast to target_shape.

    Supported common forms for a target [Q,P,N]:
      - [Q,P,N] : already horizon-specific
      - [Q,N]   : node-level mask, repeated across horizon P
      - [Q,1,N] : singleton horizon, broadcast across P

    More generally, standard NumPy broadcasting is accepted after the
    explicit [Q,N] -> [Q,1,N] promotion.
    """
    arr = np.asarray(mask, dtype=np.bool_)
    target_shape = tuple(int(x) for x in target_shape)
    if arr.shape == target_shape:
        return arr

    if len(target_shape) == 3 and arr.ndim == 2:
        # Base_SupportBench direct_mask is query x node, while forecast
        # metrics are query x horizon x node. direct_mask applies to every
        # forecast horizon for that query/node pair.
        if arr.shape[0] == target_shape[0] and arr.shape[1] == target_shape[2]:
            arr = arr[:, None, :]

    try:
        return np.broadcast_to(arr, target_shape)
    except ValueError as exc:
        raise ValueError(
            f"{name} shape {np.asarray(mask).shape} cannot broadcast to target shape {target_shape}"
        ) from exc
