#!/usr/bin/env python3
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TMP=ROOT/'tests'/'_synthetic_runtime'
CFG=TMP/'synthetic.yaml'
if not CFG.exists():
    raise SystemExit('Run tests/run_synthetic_smoke.py first')
os.environ['PYTHONPATH']=str(ROOT)+(':'+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
sys.path.insert(0,str(ROOT))
from supportbench.dataset import SupportBench
from supportbench.config import load_config

cfg=load_config(CFG); b=SupportBench('TraffiDent_Net_06',config=CFG)
derived=TMP/'derived_smoke'; derived.mkdir(parents=True,exist_ok=True)
views_file=b.sg_dir/'protocols'/'resolved'/'gap2_val_development.csv'
pool=derived/'val_pool.npz'
subprocess.check_call([sys.executable,str(ROOT/'derived'/'build_candidate_pool.py'),'--config',str(CFG),'--subgraph','TraffiDent_Net_06','--views-file',str(views_file),'--k','8','--output',str(pool)],cwd=ROOT,env=os.environ)
P=np.load(pool,allow_pickle=False)
opp=derived/'opportunity.csv'
subprocess.check_call([sys.executable,str(ROOT/'derived'/'build_support_opportunity.py'),'--config',str(CFG),'--subgraph','TraffiDent_Net_06','--candidate-pool',str(pool),'--fit-calibrator','--output',str(opp)],cwd=ROOT,env=os.environ)
cal=opp.with_suffix('.calibrator.json'); assert cal.exists()

# Mining pool is deliberately larger and contains the dedicated diverse bit.
mining=derived/'mining_pool.npz'
subprocess.check_call([sys.executable,str(ROOT/'derived'/'build_training_mining_pool.py'),'--config',str(CFG),'--subgraph','TraffiDent_Net_06','--views-file',str(views_file),'--memory-budget','development','--max-queries','10','--k','16','--per-view-k','8','--diverse-k','4','--output',str(mining)],cwd=ROOT,env=os.environ)
M=np.load(mining,allow_pickle=False); assert M['candidate_episode_id'].shape==(10,16); assert int(M['diverse_candidate_quota'])==4

# Inference-safe simple baseline: per-node observed-history mean with global fallback.
preds=[]
for vid in P['view_id']:
    s=b.sample(str(vid)); x=s.history_full[...,b.speed_idx]; m=s.observation_mask[...,b.speed_idx]
    global_mean=float(x[m].mean()) if np.any(m) else 0.0
    node=[]
    for n in range(b.N):
        vals=x[:,n][m[:,n]]; node.append(float(vals.mean()) if len(vals) else global_mean)
    preds.append(np.repeat(np.asarray(node,np.float32)[None,:],b.P,axis=0))
qpred=derived/'query_pred.npz'; np.savez_compressed(qpred,view_id=P['view_id'],pred=np.asarray(preds,np.float32))
mpreds=[]
for eid in range(len(b.memory_starts)):
    ep=b.memory_episode(eid); last=ep['history_speed'][-1]; mpreds.append(np.repeat(last[None,:],b.P,axis=0))
mpred=derived/'memory_pred.npz'; np.savez_compressed(mpred,episode_id=np.arange(len(mpreds),dtype=np.int64),pred=np.asarray(mpreds,np.float32))
alpha=derived/'alpha.json'
subprocess.check_call([sys.executable,str(ROOT/'derived'/'select_support_alpha.py'),'--config',str(CFG),'--subgraph','TraffiDent_Net_06','--candidate-pool',str(pool),'--query-predictions',str(qpred),'--memory-predictions',str(mpred),'--output',str(alpha)],cwd=ROOT,env=os.environ)
utility=derived/'utility.npz'
subprocess.check_call([sys.executable,str(ROOT/'derived'/'build_support_utility.py'),'--config',str(CFG),'--subgraph','TraffiDent_Net_06','--candidate-pool',str(pool),'--query-predictions',str(qpred),'--memory-predictions',str(mpred),'--alpha-config',str(alpha),'--output',str(utility)],cwd=ROOT,env=os.environ)
U=np.load(utility,allow_pickle=False)
for key in ['gain_all','gain_masked','gain_support_dependent','gain_h3','gain_h6','gain_h12','safe_support','support_dependent_target_count']:
    assert key in U.files, key
assert np.allclose(U['forecast_gain'],U['gain_all'],equal_nan=True)

pairs=derived/'pairs.csv'
subprocess.check_call([sys.executable,str(ROOT/'derived'/'build_reasoning_pairs.py'),'--utility',str(utility),'--output',str(pairs)],cwd=ROOT,env=os.environ)
contrast=derived/'contrastive_inputs.jsonl'; contrast_labels=derived/'contrastive_labels.jsonl'
subprocess.check_call([sys.executable,str(ROOT/'derived'/'build_contrastive_explanation.py'),'--config',str(CFG),'--subgraph','TraffiDent_Net_06','--reasoning-pairs',str(pairs),'--max-pairs','10','--output',str(contrast),'--labels-output',str(contrast_labels)],cwd=ROOT,env=os.environ)
records=derived/'records.jsonl'
subprocess.check_call([sys.executable,str(ROOT/'derived'/'build_reasoning_records.py'),'--config',str(CFG),'--subgraph','TraffiDent_Net_06','--utility',str(utility),'--max-records','50','--output',str(records)],cwd=ROOT,env=os.environ)
inputs=derived/'explanation_inputs.jsonl'; labels=derived/'explanation_labels.jsonl'
subprocess.check_call([sys.executable,str(ROOT/'derived'/'build_explanation_inputs.py'),'--config',str(CFG),'--subgraph','TraffiDent_Net_06','--candidate-pool',str(pool),'--max-pairs','25','--output',str(inputs)],cwd=ROOT,env=os.environ)
subprocess.check_call([sys.executable,str(ROOT/'derived'/'build_explanation_labels.py'),'--config',str(CFG),'--utility',str(utility),'--max-pairs','25','--output',str(labels)],cwd=ROOT,env=os.environ)
headroom=derived/'headroom.json'
subprocess.check_call([sys.executable,str(ROOT/'derived'/'analyze_rag_headroom.py'),'--utility',str(utility),'--ks','4','8','--output',str(headroom)],cwd=ROOT,env=os.environ)

first=json.loads(records.read_text().splitlines()[0]); ein=json.loads(inputs.read_text().splitlines()[0]); elab=json.loads(labels.read_text().splitlines()[0])
assert first['evidence']['query_anchors'] and first['evidence']['candidate_anchors']
assert 'historical_upstream_to_target' in first['evidence']['candidate_anchors'][0]
assert 'predictive_neighbor_lag_minutes' in first['evidence']['query_anchors'][0]
assert first['label']['primary_gain_key']=='gain_support_dependent'
assert 'decision_target' not in ein and 'primary_gain' not in ein and 'gain_support_dependent' not in ein
assert 'primary_gain' in elab
assert contrast.exists() and contrast_labels.exists() and headroom.exists()
print('DERIVED SMOKE PASS',derived)
print('selected alpha',json.loads(alpha.read_text())['selected_alpha'])
print('reasoning primary gain',first['label']['primary_gain_key'])
print('explanation input/label separation PASS')
