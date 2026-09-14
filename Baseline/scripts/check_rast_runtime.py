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
