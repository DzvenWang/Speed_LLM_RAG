#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, subprocess
import numpy as np
import pandas as pd
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline'); sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml, write_json
from baseline.run_identity import load_cache_identity
from baseline.shared_cache import cache_path


def interval_overlap_pairs(memory_starts, query_starts, span):
    """Count potential overlapping [start,start+span) memory/query interval pairs."""
    m=np.sort(np.asarray(memory_starts,dtype=np.int64)); q=np.asarray(query_starts,dtype=np.int64)
    total=0; affected=0
    for s in q:
        lo=np.searchsorted(m,s-span+1,side='left'); hi=np.searchsorted(m,s+span-1,side='right')
        c=int(max(0,hi-lo)); total+=c; affected+=int(c>0)
    return int(total),int(affected)

p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['development','formal'],default='development'); a=p.parse_args()
cfg=load_yaml(ROOT/'configs/baseline.yaml'); subprocess.check_call([cfg['project']['python'],str(ROOT/'scripts/validate_cache.py'),'--stage',a.stage])
cache=cache_path(cfg,a.stage); ident=load_cache_identity(cache,None)
tr=pd.read_csv(cache/'train_manifest.csv'); va=pd.read_csv(cache/'val_manifest.csv')
train_vid=set(tr.view_id.astype(str)); val_vid=set(va.view_id.astype(str)); train_gid=set(tr.view_group_id.astype(str)); val_gid=set(va.view_group_id.astype(str))
span=int(cfg['experiment']['history'])+int(cfg['experiment']['horizon'])
pairs,affected=interval_overlap_pairs(tr.window_start,va.window_start,span)
report={**ident,'artifact_type':'retrieval_leakage_audit','memory_split':'train','query_split':'val','train_rows':len(tr),'val_rows':len(va),
 'view_id_overlap':len(train_vid&val_vid),'view_group_overlap':len(train_gid&val_gid),'query_future_allowed':bool(cfg['protocol']['current_query_future_allowed_at_inference']),
 'retrieval_memory_policy':cfg['protocol']['retrieval_memory'],'retrieval_memory_signal':cfg['protocol'].get('retrieval_memory_signal'),
 'test_cache_present':(cache/'test_manifest.csv').exists(),'interval_span_steps':span,
 'potential_train_memory_val_query_interval_overlap_pairs':pairs,'val_queries_with_potential_interval_overlap':affected,
 'external_runner_interval_exclusion_required':bool(pairs>0),
 'external_runner_policy':'RAFT/SARAF exclude all Train-memory windows whose [history+future] interval overlaps each Train/Val query interval.'}
# Potential temporal overlap does not fail the cache: external retrieval runners explicitly mask those candidates.
report['pass']=report['view_id_overlap']==0 and report['view_group_overlap']==0 and not report['query_future_allowed'] and report['retrieval_memory_policy']=='train_only' and not report['test_cache_present']
out=Path(cfg['project']['exp_root'])/'_audit'/f'retrieval_leakage_{a.stage}.json'; write_json(out,report)
print(json.dumps(report,indent=2,sort_keys=True)); print('wrote',out)
if not report['pass']: raise SystemExit(2)
print('RETRIEVAL LEAKAGE AUDIT PASS')
