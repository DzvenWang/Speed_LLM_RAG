#!/usr/bin/env python3
from __future__ import annotations
import importlib, importlib.util, json
if importlib.util.find_spec('faiss') is None:
    raise SystemExit('RAST_RUNTIME_FAIL: module faiss missing. Run: bash scripts/setup_rast_faiss.sh')
faiss=importlib.import_module('faiss')
need=['IndexFlatIP','IndexHNSWFlat','normalize_L2']; miss=[x for x in need if not hasattr(faiss,x)]
if miss: raise SystemExit(f'RAST_RUNTIME_FAIL: faiss missing symbols {miss}')
info={'version':getattr(faiss,'__version__','unknown'),'module':getattr(faiss,'__file__',None),'gpu_symbols':{k:hasattr(faiss,k) for k in ['StandardGpuResources','GpuIndexFlatIP','index_cpu_to_gpu']}}
print(json.dumps(info,indent=2,sort_keys=True)); print('RAST FAISS RUNTIME PASS')

# Formal-path native import contract (kept here so the release needs no tests/ tree).
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from baseline.compat.rast_optional_backbones import install_rast_native_import_shims
modes = install_rast_native_import_shims()
from baselines.STID.arch.stid_arch import STID
from baselines.AGCRN.arch.agcrn_arch import AGCRN
from basicts.utils import get_regular_settings, load_adj
assert STID.__name__ == 'STID' and AGCRN.__name__ == 'AGCRN'
assert callable(get_regular_settings) and callable(load_adj)
assert modes == {'optional_backbones': 'native_stub', 'basicts': 'native_stub'}
try:
    get_regular_settings('PEMS04')
except RuntimeError as e:
    assert 'unexpectedly called BasicTS' in str(e)
else:
    raise AssertionError('BasicTS guard did not trigger')
from baseline.runners.rast_runner import import_model
RAST = import_model()
assert RAST.__name__ == 'RAST'
assert 'easytorch' not in sys.modules
print(f'RAST IMPORT CONTRACT PASS modes={modes}')
