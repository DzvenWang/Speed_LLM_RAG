from __future__ import annotations

"""Minimal Torch-Spatiotemporal compatibility surface for ImputeFormer.

The locked ImputeFormer model imports only::

    from tsl.nn import utils
    from tsl.nn.blocks.encoders import MLP

On Gadi, importing the bundled full TSL package can traverse optional PyG
modules and load an incompatible ``torch_scatter`` binary.  This shim exposes
only the two interfaces required by ImputeFormer without modifying vendor
sources or the Python environment.

The MLP implementation follows Torch-Spatiotemporal's public MLP definition:
stacked Dense(input -> hidden, activation, dropout) blocks followed by an
optional linear readout.  For the ImputeFormer configuration used here,
``activation='relu'`` and ``dropout=0``.
"""

import sys
import types
from typing import Optional

import torch
from torch import nn


def maybe_cat_exog(x: torch.Tensor,
                   u: Optional[torch.Tensor],
                   dim: int = -1) -> torch.Tensor:
    """Concatenate optional exogenous variables using TSL-compatible shapes."""
    if u is None:
        return x

    # TSL accepts node-shared exogenous inputs shaped [B, T, F_u] for
    # node-wise tensors [B, T, N, F_x].  Expand over nodes when necessary.
    if u.ndim == x.ndim - 1 and x.ndim >= 3:
        node_dim = x.ndim - 2
        u = u.unsqueeze(node_dim)
        shape = list(u.shape)
        shape[node_dim] = x.shape[node_dim]
        u = u.expand(*shape)

    if u.ndim != x.ndim:
        raise RuntimeError(
            f'ImputeFormer TSL shim cannot concatenate x.ndim={x.ndim} '
            f'and u.ndim={u.ndim}'
        )
    return torch.cat([x, u], dim=dim)


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'gelu':
        return nn.GELU()
    if name in ('silu', 'swish'):
        return nn.SiLU()
    if name == 'elu':
        return nn.ELU()
    if name in ('identity', 'none', 'linear'):
        return nn.Identity()
    raise ValueError(f'Unsupported activation for ImputeFormer TSL shim: {name}')


class Dense(nn.Module):
    """TSL-style dense layer: Linear -> activation -> optional dropout."""

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.activation = _activation(activation)
        self.dropout = nn.Dropout(dropout) if float(dropout) > 0 else nn.Identity()

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(x)))


class MLP(nn.Module):
    """API-compatible subset of ``tsl.nn.blocks.encoders.MLP``."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: Optional[int] = None,
                 exog_size: Optional[int] = None,
                 n_layers: int = 1,
                 activation: str = 'relu',
                 dropout: float = 0.0):
        super().__init__()
        if exog_size is not None:
            input_size += int(exog_size)
        self.mlp = nn.Sequential(*[
            Dense(
                input_size=input_size if i == 0 else hidden_size,
                output_size=hidden_size,
                activation=activation,
                dropout=dropout,
            )
            for i in range(int(n_layers))
        ])
        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.readout = None

    def reset_parameters(self) -> None:
        for module in self.mlp:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        if self.readout is not None:
            self.readout.reset_parameters()

    def forward(self,
                x: torch.Tensor,
                u: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = maybe_cat_exog(x, u)
        out = self.mlp(x)
        if self.readout is not None:
            return self.readout(out)
        return out


def install_imputeformer_tsl_shim() -> dict:
    """Install a minimal ``tsl`` module tree into ``sys.modules``.

    This function is intentionally idempotent.  It must run before importing
    ``imputeformer.models`` so Python never executes the full bundled TSL
    package and therefore never touches the incompatible torch_scatter wheel.
    """
    # Remove partial/broken TSL imports left by a previous failed attempt.
    for key in list(sys.modules):
        if key == 'tsl' or key.startswith('tsl.'):
            del sys.modules[key]

    tsl_mod = types.ModuleType('tsl')
    tsl_mod.__path__ = []
    nn_mod = types.ModuleType('tsl.nn')
    nn_mod.__path__ = []
    utils_mod = types.ModuleType('tsl.nn.utils')
    utils_mod.maybe_cat_exog = maybe_cat_exog

    blocks_mod = types.ModuleType('tsl.nn.blocks')
    blocks_mod.__path__ = []
    enc_mod = types.ModuleType('tsl.nn.blocks.encoders')
    enc_mod.__path__ = []
    enc_mod.MLP = MLP

    layers_mod = types.ModuleType('tsl.nn.layers')
    layers_mod.__path__ = []
    base_mod = types.ModuleType('tsl.nn.layers.base')
    base_mod.Dense = Dense

    tsl_mod.nn = nn_mod
    nn_mod.utils = utils_mod
    nn_mod.blocks = blocks_mod
    nn_mod.layers = layers_mod
    blocks_mod.encoders = enc_mod
    layers_mod.base = base_mod

    modules = {
        'tsl': tsl_mod,
        'tsl.nn': nn_mod,
        'tsl.nn.utils': utils_mod,
        'tsl.nn.blocks': blocks_mod,
        'tsl.nn.blocks.encoders': enc_mod,
        'tsl.nn.layers': layers_mod,
        'tsl.nn.layers.base': base_mod,
    }
    sys.modules.update(modules)
    return {
        'mode': 'minimal_tsl_compat_shim',
        'provided': sorted(modules),
        'torch_scatter_required': False,
    }
