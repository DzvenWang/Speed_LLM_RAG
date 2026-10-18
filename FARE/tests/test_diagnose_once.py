import json, subprocess, sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fare.diagnostics_once.data import load_shard,InvalidDiagnosticInput
from fare.diagnostics_once.engine import evaluate_shard


def make_shard(p,target='speed',s=40,family='entry_random',n=10,leak=False,rand=False):
    rng=np.random.default_rng(10)
    y=rng.normal(size=(n,12,5))
    no=y+0.7
    full=y+0.4
    extra={'mask_ratio':np.linspace(.1,.9,n),'demand':np.linspace(.2,.8,n),
           'alpha':np.ones(n)*0.8,'null_weight':np.ones(n)*.7,
           'future_start':np.arange(50,50+n),
           'memory_last_future_end':np.arange(49,49+n)+(2 if leak else 0)}
    data={'y_true':y,'valid_mask':np.ones_like(y,dtype=bool),
          'query_id':np.arange(n),
          'metadata_json':np.array(json.dumps({'target':target,'severity':s,'failure_family':family,'split':'val','metrics_space':'original_units','checkpoint':'best.pt','checkpoint_sha256':'f'*64,'seed':10})),
          'pred__full':full,'pred__no_evidence':no,'pred__null_evidence':no.copy(),
          'pred__random':y+0.65,'pred__similarity_only':y+0.55,
          'pred__hidden_only':y+0.44,'pred__forecast_only':y+0.67,
          'pred__no_gate':y+0.5,'pred__shuffled':y+.6,
          'pred__wrong_evidence':y+.8,'pred__no_mask_replay':y+.53,
          'pred_candidates':np.stack([y+.1,y+.3,y+.65],axis=1)}
    data.update(extra)
    if rand: data['pred__null_evidence'][0,0,0]+=0.3
    np.savez_compressed(p,**data)


def test_metrics_and_oracle(tmp_path):
    path=tmp_path/'shard.npz';make_shard(path)
    d=load_shard(path)
    r=evaluate_shard(d,n_boot=40)
    assert abs(r['modes']['full']['MAE']-.4)<1e-8
    assert r['oracle_best_candidate_gain']>.5
    assert r['oracle_capture_ratio']>0
    assert r['null_path_equivalent_1e-5'] is True
    assert r['harm_rate']==0
    assert r['candidate_count']==3

def test_no_overlap(tmp_path):
    path=tmp_path/'bad.npz';make_shard(path,leak=True)
    with pytest.raises(InvalidDiagnosticInput,match='future leakage'):load_shard(path)

def test_null_bug_detected(tmp_path):
    path=tmp_path/'bad.npz';make_shard(path,rand=True)
    r=evaluate_shard(load_shard(path),n_boot=40)
    assert r['null_path_equivalent_1e-5'] is False

def test_no_test_split(tmp_path):
    path=tmp_path/'shard.npz';make_shard(path)
    with np.load(path) as x:d={k:x[k] for k in x.files}
    md=json.loads(str(d['metadata_json'].item()));md['split']='test';d['metadata_json']=np.array(json.dumps(md))
    np.savez_compressed(path,**d)
    with pytest.raises(InvalidDiagnosticInput,match='split=val'):load_shard(path)

def test_cli_end_to_end(tmp_path):
    inp=tmp_path/'inputs';inp.mkdir()
    out=tmp_path/'out'
    make_shard(inp/'one.npz')
    cfg=tmp_path/'cfg.json';cfg.write_text(json.dumps({'expected_targets':['speed'],'expected_severities':[40],'expected_families':['entry_random'],'expected_family_count':1}))
    cmd=[sys.executable,str(ROOT/'scripts/diagnose_once.py'), '--config',str(cfg), '--input-dir',str(inp),'--output-dir',str(out),'--strict-coverage','--strict-oracle','--seed','10']
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    assert p.returncode==0,p.stdout+'\n'+p.stderr
    assert (out/'diagnostic_report.md').is_file()
    assert (out/'overall_comparison.csv').is_file()
    assert (out/'module_contribution.csv').is_file()
    assert (out/'diagnostic_report.json').is_file()
