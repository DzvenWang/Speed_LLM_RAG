"""Regression coverage for failure-prone result aggregation cases."""
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_zero_stable_candidates_does_not_open_stla(monkeypatch, tmp_path):
    repo=tmp_path/'repo'
    (repo/'Backbone/configs').mkdir(parents=True)
    (repo/'Backbone/configs/backbone.yaml').write_text('model: STLAformer\n')
    cached=repo/'cache';cached.mkdir()
    (cached/'cache_manifest.json').write_text('{}')
    ckpt=repo/'model.pt';ckpt.write_bytes(b'diagnostic placeholder')
    backbone=types.ModuleType('backbone');backbone.__path__=[]
    sc=types.ModuleType('backbone.shared_cache');sc.cache_path=lambda c,stage:cached
    ut=types.ModuleType('backbone.utils');ut.load_yaml=lambda p:{}
    monkeypatch.setitem(sys.modules,'backbone',backbone)
    monkeypatch.setitem(sys.modules,'backbone.shared_cache',sc)
    monkeypatch.setitem(sys.modules,'backbone.utils',ut)
    spec=importlib.util.spec_from_file_location('run_diagnostic_test',ROOT/'scripts/run_diagnostic.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=repo/'diagnostic';out.mkdir()
    frame=pd.DataFrame({'stable_direction_lag':[False],'best_lag':[0]})
    result=module.infer_gain(repo,'speed',None,frame,ckpt,
                             SimpleNamespace(dataset='TraffiDent_Net_06'),out,{})
    assert result['confirmed_count']==0
    assert result['validation_confirmation'] is False
    assert 'No stable' in result['validation_skipped_reason']
    csv=pd.read_csv(out/'stla_incremental_gain_edges.csv')
    assert csv.empty and {'src','dst','lag','fully_confirmed'} <= set(csv.columns)
