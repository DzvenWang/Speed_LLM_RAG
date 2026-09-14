from __future__ import annotations

from typing import Any

import numpy as np
import torch


LEAF_GRAPH_COMPAT_REVISION = "leaf_graph_ops_r1"


def symmetric_norm_adj(adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Symmetric adjacency normalization used by LEAF's graph branch.

    Implements the normalization stated in the LEAF paper:
        D^{-1/2} A D^{-1/2}

    Supports ``[..., N, N]`` dense tensors and preserves device/dtype.
    The function deliberately does *not* add self loops; frozen LEAF code
    constructs the spatio-temporal/self-loop adjacency before calling it.
    """
    if not torch.is_tensor(adj):
        raise TypeError(f"adj must be a torch.Tensor, got {type(adj)!r}")
    if adj.ndim < 2 or adj.shape[-1] != adj.shape[-2]:
        raise ValueError(f"adj must end in square [N,N], got {tuple(adj.shape)}")

    degree = adj.sum(dim=-1)
    # Official LEAF graph construction is non-negative.  Clamp only guards
    # isolated/zero-degree rows and floating-point underflow.
    inv_sqrt = degree.clamp_min(float(eps)).pow(-0.5)
    return inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)


def install_leaf_graph_compat(graph_module: Any, leaf_args: Any, adjacency: np.ndarray) -> dict:
    """Install adapter-side compatibility for the frozen LEAF snapshot.

    The released source references ``norm_adj`` without defining it and the
    HypergraphBranch references ``args.predefined_adjs`` without assigning it.
    We keep ``vendor/LEAF`` immutable and inject the missing runtime symbols.

    ``predefined_adjs`` has no released definition in the pinned source.  The
    external bridge therefore uses the only benchmark road adjacency available
    to LEAF for both expected slots.  This is an explicit compatibility
    fallback, not claimed to reproduce an undocumented upstream preprocessing
    choice.
    """
    adj = np.asarray(adjacency, dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"LEAF adjacency must be square, got {adj.shape}")

    graph_module.norm_adj = symmetric_norm_adj

    # Frozen hypergraph.py indexes exactly two predefined supports.
    leaf_args.predefined_adjs = [adj.copy(), adj.copy()]

    return {
        "revision": LEAF_GRAPH_COMPAT_REVISION,
        "norm_adj": "paper_D^-1/2_A_D^-1/2",
        "norm_adj_injected_into": "vendor/LEAF/model/graph.py runtime module",
        "predefined_adjs_policy": "two_copies_of_shared_benchmark_road_adjacency",
        "predefined_adjs_count": 2,
        "vendor_modified": False,
    }
