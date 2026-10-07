from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from fare.context import FAREContext
from fare.cache import CacheSplit


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--target',choices=['speed','flow'],required=True); ap.add_argument('--run-mode',choices=['formal','smoke'],default='formal'); a=ap.parse_args(); ctx=FAREContext(a.target,run_mode=a.run_mode)
    tr=CacheSplit(ctx.formal_cache,'train'); va=CacheSplit(ctx.formal_cache,'val')
    assert tr.x_obs.shape[1:]==(12,int(ctx.bench.N)); assert va.x_obs.shape[1:]==(12,int(ctx.bench.N))
    # Runtime dependency audit: no historical FARE projects and no Baseline imports.
    forbidden=('STLA_DASR','Support_Oracle','from baseline','import baseline')
    offenders=[]
    for p in (ROOT/'fare').rglob('*.py'):
        s=p.read_text(encoding='utf-8')
        for token in forbidden:
            if token in s: offenders.append(f'{p}:{token}')
    if offenders: raise RuntimeError('Forbidden dependency references: '+str(offenders))
    print('FARE_PREFLIGHT_PASS',flush=True); print('version 0.2.1',flush=True); print('target',ctx.target,flush=True); print('dataset',ctx.dataset,'N',ctx.bench.N,flush=True); print('formal_cache',ctx.formal_cache,flush=True); print('backbone_checkpoint',ctx.checkpoint,flush=True); print('checkpoint_sha256',ctx.checkpoint_sha256[:12],flush=True); print('run_mode',ctx.run_mode,flush=True); print('dependency_policy Base_SupportBench + Backbone only; Baseline comparison-only; no prior FARE dependency',flush=True); print('experts context + structural_tmcp + learned_retrieval + backbone_aware; no MoE router in v0.2',flush=True); print('restoration_policy every expert restores every artificial_missing cell; observed/original-invalid immutable',flush=True)
if __name__=='__main__': main()
