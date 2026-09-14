#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.compat.rast_optional_backbones import install_rast_native_import_shims

modes = install_rast_native_import_shims()

from baselines.STID.arch.stid_arch import STID
from baselines.AGCRN.arch.agcrn_arch import AGCRN
from basicts.utils import get_regular_settings, load_adj

assert STID.__name__ == "STID"
assert AGCRN.__name__ == "AGCRN"
assert callable(get_regular_settings) and callable(load_adj)
assert modes == {"optional_backbones": "native_stub", "basicts": "native_stub"}

# Guarded BasicTS functions must fail loudly if native RAST unexpectedly calls them.
try:
    get_regular_settings("PEMS04")
except RuntimeError as e:
    assert "unexpectedly called BasicTS" in str(e)
else:
    raise AssertionError("BasicTS guard did not trigger")

# Full RAST import is attempted only when FAISS is available in this runtime.
try:
    import faiss  # noqa: F401
except Exception:
    print(f"RAST NATIVE IMPORT SHIMS PASS modes={modes}; full import skipped (faiss unavailable here)")
else:
    from baseline.runners.rast_runner import import_model
    RAST = import_model()
    assert RAST.__name__ == "RAST"
    assert "easytorch" not in sys.modules
    print(f"RAST IMPORT CONTRACT SMOKE PASS modes={modes}")
