from pathlib import Path
from types import SimpleNamespace
import sys, tempfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from baseline.time_features import resolve_time_features, assert_resolved_time_features


def bench(root,T):
    return SimpleNamespace(traffic=np.zeros((T,2),np.float32),cfg={'forecast':{'freq_minutes':5}},sg_dir=root)

T=7*288
with tempfile.TemporaryDirectory() as td:
    root=Path(td); (root/'core').mkdir(); ts=pd.date_range('2026-01-05',periods=T,freq='5min')
    pd.DataFrame({'index':np.arange(T),'timestamp':ts.astype(str)}).to_csv(root/'core/time_index.csv',index=False)
    r=resolve_time_features(bench(root,T),allow_index_fallback=False); a=assert_resolved_time_features(r)
    assert r.column=='timestamp' and r.parser=='datetime_string'
    assert a['tod_unique']==288 and a['dow_unique']==7
with tempfile.TemporaryDirectory() as td:
    root=Path(td); (root/'core').mkdir(); pd.DataFrame({'index':np.arange(T)}).to_csv(root/'core/time_index.csv',index=False)
    b=bench(root,T)
    try: resolve_time_features(b,allow_index_fallback=False)
    except RuntimeError: pass
    else: raise AssertionError('numeric row index was accepted as datetime')
    r=resolve_time_features(b,allow_index_fallback=True,index_origin_dow=0); a=assert_resolved_time_features(r)
    assert r.source=='index_modulo_fallback'
    assert a['tod_unique']==288 and a['dow_unique']==7
    assert a['tod_step_consistency']==1.0 and a['dow_rollover_consistency']==1.0
print('BASELINE TIME FEATURE CONTRACT SMOKE PASS')
