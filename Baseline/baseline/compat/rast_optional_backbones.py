from __future__ import annotations

import sys
import types
import torch.nn as nn


class _UnavailableOfficialBackbone(nn.Module):
    _backbone_name = "unknown"

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError(
            f"RAST attempted to instantiate optional backbone {self._backbone_name}. "
            "This Baseline compatibility layer is valid only for the native RAST "
            "external runner, where pre_train_path/pre_train_model_name are unset. "
            "Do not use this shim for an official pretrained-STGNN RAST experiment."
        )


class STID(_UnavailableOfficialBackbone):
    _backbone_name = "STID"


class AGCRN(_UnavailableOfficialBackbone):
    _backbone_name = "AGCRN"


def _pkg(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
    return mod


def _unused_basicts(*args, **kwargs):
    raise RuntimeError(
        "RAST native external runner unexpectedly called BasicTS utility code. "
        "BasicTS/EasyTorch is intentionally not part of the TFP runtime because "
        "native RAST does not use an official pretrained STID/AGCRN backbone."
    )


def install_rast_optional_backbone_shim() -> str:
    """Install deterministic STID/AGCRN import placeholders.

    Frozen RAST eagerly imports these optional classes even when
    ``pre_train_path`` is None.  We deliberately do not try to import a random
    system/upstream ``baselines`` package first, because doing so can pull the
    full BasicTS/EasyTorch stack into an experiment that never uses it.
    """
    baselines = _pkg("baselines")

    stid_pkg = _pkg("baselines.STID")
    stid_arch_pkg = _pkg("baselines.STID.arch")
    stid_mod = types.ModuleType("baselines.STID.arch.stid_arch")
    stid_mod.STID = STID
    sys.modules[stid_mod.__name__] = stid_mod
    baselines.STID = stid_pkg
    stid_pkg.arch = stid_arch_pkg
    stid_arch_pkg.stid_arch = stid_mod

    agcrn_pkg = _pkg("baselines.AGCRN")
    agcrn_arch_pkg = _pkg("baselines.AGCRN.arch")
    agcrn_mod = types.ModuleType("baselines.AGCRN.arch.agcrn_arch")
    agcrn_mod.AGCRN = AGCRN
    sys.modules[agcrn_mod.__name__] = agcrn_mod
    baselines.AGCRN = agcrn_pkg
    agcrn_pkg.arch = agcrn_arch_pkg
    agcrn_arch_pkg.agcrn_arch = agcrn_mod

    return "native_stub"


def install_rast_basicts_shim() -> str:
    """Expose only the BasicTS names imported by RAST model-config helpers.

    Importing ``vendor/RAST/basicts`` normally executes ``basicts/__init__.py``
    which imports ``launcher.py`` and therefore EasyTorch.  In the native RAST
    external runner, model-config helpers are imported but never executed,
    because no pretrained STID/AGCRN checkpoint is requested.  These guarded
    placeholders therefore preserve the frozen RAST source while preventing a
    dead optional dependency from aborting import.
    """
    basicts = _pkg("basicts")
    utils = types.ModuleType("basicts.utils")
    utils.get_regular_settings = _unused_basicts
    utils.load_adj = _unused_basicts
    sys.modules["basicts.utils"] = utils
    basicts.utils = utils
    return "native_stub"


def install_rast_native_import_shims() -> dict:
    """Install all import-only compatibility modules required by native RAST."""
    return {
        "optional_backbones": install_rast_optional_backbone_shim(),
        "basicts": install_rast_basicts_shim(),
    }
