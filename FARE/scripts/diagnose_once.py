#!/usr/bin/env python3
"""Run a single, validation-only, paired FARE diagnostic, no training.

Integrates directly with native FARE v4.0.3 checkpoints and its original source.
Shards are generated once per target inside a single GPU job; no retraining.
"""
from __future__ import annotations
import argparse,hashlib,importlib,inspect,json,os,sys,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from fare.diagnostics_once.data import load_shard,sha256_file,InvalidDiagnosticInput
from fare.diagnostics_once.engine import evaluate_shard
from fare.diagnostics_once.report import report,save_json

def _adapter(spec):
    if not spec or ':' not in spec:raise ValueError('--adapter must be module:function, e.g., my_fare_export:export_once')
    module,function=spec.split(':',1)
    mod=importlib.import_module(module)
    if not bool(getattr(mod, 'FARE_DIAGNOSTIC_ADAPTER_READY', False)):
        raise ValueError(f'{module} has not declared FARE_DIAGNOSTIC_ADAPTER_READY=True; refusing GPU submission')
    func=getattr(mod,function)
    if not callable(func):raise ValueError(f'{spec} not callable')
    required=('target','checkpoint','output_dir','severities','failure_families','max_queries_per_group','seed','split','config')
    sig=inspect.signature(func)
    if not all(k in sig.parameters or any(v.kind==inspect.Parameter.VAR_KEYWORD for v in sig.parameters.values()) for k in required):
        raise ValueError('exporter signature must accept: '+', '.join(required))
    return func

def _config(a):
    if a.config:
        with open(a.config,encoding='utf-8') as f:c=json.load(f)
    else:c={}
    if not isinstance(c,dict):raise ValueError('JSON config must be an object')
    c.setdefault('expected_targets',['speed','flow'])
    c.setdefault('split','val')
    c.setdefault('allow_exploratory_test',False)
    c.setdefault('expected_severities',[20,40,60,90])
    c.setdefault('expected_families',[])
    return c

def preflight(a,c):
    failures=[];checks=[]
    try:out=Path(a.output_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True);checks.append('output directory writable')
    except Exception as e:failures.append('output directory not writable: '+str(e))
    if c['split']=='test' and not c.get('allow_exploratory_test',False):
        failures.append('Test-split exploratory audit needs config.allow_exploratory_test=true; Test labels cannot be used to choose architecture without qualification')
    if c['split'] not in ('val','test'):failures.append('split must be val or test')
    inp=Path(a.input_dir).expanduser().resolve()
    npzs=sorted(inp.rglob('*.npz')) if inp.exists() else []
    if not npzs and not a.adapter:failures.append('No NPZ export shards found, and no --adapter provided. No model inference can happen.')
    if a.adapter:
        try:_adapter(a.adapter);checks.append('adapter importable')
        except Exception as e:failures.append('adapter import error: '+repr(e))
        for t in c['expected_targets']:
            p=a.speed_checkpoint if t=='speed' else a.flow_checkpoint
            if not p or not Path(p).is_file():failures.append(f'{t} checkpoint file missing: {p}')
            else:checks.append(t+' checkpoint exists')
    if npzs:
        sample_size=min(2,len(npzs))
        for f in npzs[:sample_size]:
            try:load_shard(f,max_queries=2,allowed_splits=(c['split'],));checks.append('NPZ contract valid: '+f.name)
            except Exception as e:failures.append('NPZ invalid '+str(f)+': '+str(e))
    status={'ok':not failures,'checks':checks,'failures':failures,'existing_input_shards':len(npzs),'adapter':a.adapter,'input':str(inp)}
    Path(a.output_dir).mkdir(parents=True,exist_ok=True)
    save_json(Path(a.output_dir)/'preflight.json',status)
    for x in checks:print('[PREFLIGHT PASS]',x,flush=True)
    for x in failures:print('[PREFLIGHT FAIL]',x,flush=True)
    return status

def run(a,c):
    status=preflight(a,c)
    if not status['ok']:raise RuntimeError('Preflight failed; refusing to spend GPU time')
    inp=Path(a.input_dir).resolve();inp.mkdir(parents=True,exist_ok=True)
    if a.adapter:
        f=_adapter(a.adapter)
        for t in c['expected_targets']:
            ckpt=Path(a.speed_checkpoint if t=='speed' else a.flow_checkpoint).resolve()
            print('[EXPORT BEGIN]',t,ckpt,flush=True)
            f(target=t,checkpoint=str(ckpt),output_dir=str(inp),
              severities=list(c['expected_severities']),
              failure_families=list(c['expected_families']),
              max_queries_per_group=a.max_queries_per_group, seed=a.seed,
              split=c['split'], config=c)
            print('[EXPORT END]',t,flush=True)
    paths=sorted(inp.rglob('*.npz'))
    if not paths:raise RuntimeError('Exporter generated no NPZ shards')
    manifests=[];rows=[];identity={};query_ids_by_target={}
    for i,path in enumerate(paths,1):
        d=load_shard(path,max_queries=a.max_queries_per_group,seed=a.seed,allowed_splits=(c['split'],))
        k=(d.metadata['target'],int(d.metadata['severity']),d.metadata['failure_family'])
        if k in identity: raise InvalidDiagnosticInput('Duplicate stratum '+str(k))
        identity[k]=str(path)
        if str(d.metadata.get('metrics_space','')) != 'original_units':
            raise InvalidDiagnosticInput('Export predictions must be inverse-transformed to original physical units; set metadata.metrics_space=original_units')
        if d.metadata['target'] not in c['expected_targets'] or int(d.metadata['severity']) not in c['expected_severities']:
            raise InvalidDiagnosticInput('Unexpected target/severity '+str(k))
        if c['expected_families'] and d.metadata['failure_family'] not in c['expected_families']:
            raise InvalidDiagnosticInput('Unexpected failure family '+str(k))
        if int(d.metadata['seed'])!=a.seed:raise InvalidDiagnosticInput('Seed mismatch for '+str(path))
        if d.metadata['split']!=c['split']:raise InvalidDiagnosticInput('Dataset split mismatch')
        # The NPZ may carry query ids that repeat across severity/families; this is
        # allowed and is NOT pooled as independent observations in bootstrap.
        stat=evaluate_shard(d,seed=a.seed,n_boot=a.bootstrap)
        if a.strict_modes and stat['missing_recommended_modes']:
            raise InvalidDiagnosticInput(str(k)+' missing required audit modes: '+str(stat['missing_recommended_modes']))
        if a.strict_oracle and 'oracle_best_candidate_gain' not in stat:
            raise InvalidDiagnosticInput(str(k)+' lacks pred_candidates for oracle')
        rows.append(stat)
        manifests.append({'path':str(path),'sha256':sha256_file(path),'stratum':k,'n_queries':d.n,'checkpoint':d.metadata['checkpoint'],'checkpoint_sha256':d.metadata['checkpoint_sha256']})
        print(f'[AUDIT {i}/{len(paths)}] target={k[0]} s{k[1]} family={k[2]} queries={d.n} full_mae={stat["modes"]["full"]["MAE"]:.6f}',flush=True)
    # A single checkpoint per target is essential for paired analysis.
    for t in c['expected_targets']:
        chk={z['checkpoint_sha256'] for z in manifests if z['stratum'][0]==t}
        if len(chk)>1:raise InvalidDiagnosticInput(f'Inconsistent {t} checkpoint hashes across shards: {chk}')
        expected_ckpt=a.speed_checkpoint if t=='speed' else a.flow_checkpoint
        if expected_ckpt and Path(expected_ckpt).is_file() and chk:
            actual=sha256_file(Path(expected_ckpt))
            if actual != next(iter(chk)):
                raise InvalidDiagnosticInput(f'{t} NPZ provenance digest does not match supplied frozen checkpoint')
    expected={'targets':c['expected_targets'],'severities':c['expected_severities'],'families':c['expected_families']}
    families=sorted(set(z['failure_family'] for z in rows))
    if not expected['families']:expected['families']=families
    if len(families)<int(c.get('expected_family_count',5)) and a.strict_coverage:
        raise InvalidDiagnosticInput(f'Only {len(families)} failure families found; expected at least {c.get("expected_family_count",5)}')
    missing=[(t,int(s),f) for t in expected['targets'] for s in expected['severities'] for f in expected['families'] if (t,int(s),f) not in identity] if expected['families'] else []
    if a.strict_coverage and (missing or not all(any(z['target']==t for z in rows) for t in c['expected_targets'])):
        raise InvalidDiagnosticInput('Incomplete validation coverage: missing '+str(missing))
    prov={'seed':a.seed,'input_shards':manifests,'export_adapter':a.adapter,'max_queries_per_group':a.max_queries_per_group,
          'bootstrap':a.bootstrap,'source_split':c['split'], 'architecture_selection_safe': c['split']=='val',
          'note':'No training performed; oracle uses labels from source_split and is never inference; selected-slot oracle is not global pool upper bound'}
    output=Path(a.output_dir);save_json(output/'manifest.json',prov)
    doc=report(rows,output,prov,expected)
    if missing:print('[WARNING] expected missing strata:',missing,flush=True)
    print('DIAGNOSE_ONCE_COMPLETE',str(output.resolve()),flush=True)
    return doc

def main():
    p=argparse.ArgumentParser(description='One-pass FARE v4.0.4 diagnostic, validation-only, never trains.')
    p.add_argument('--config',type=str,default=None)
    p.add_argument('--input-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--adapter',default=None,help='Native FARE exporter module:function; run once per target')
    p.add_argument('--speed-checkpoint',default=None)
    p.add_argument('--flow-checkpoint',default=None)
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--max-queries-per-group',type=int,default=64)
    p.add_argument('--bootstrap',type=int,default=400)
    p.add_argument('--strict-coverage',action='store_true',help='Fail unless all expected target × severity × family shards exist')
    p.add_argument('--strict-modes',action='store_true',help='Require all recommended mode exports')
    p.add_argument('--strict-oracle',action='store_true',help='Require candidate predictions for all shards')
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args();c=_config(a)
    if a.adapter and (not a.speed_checkpoint or not a.flow_checkpoint) and not a.preflight:
        print('[DIAGNOSE_ONCE_INFO] Both Speed/Flow checkpoints are required; use the included PBS with auto-discovery or pass both explicit paths.',flush=True)
    try:
        if a.preflight:sys.exit(0 if preflight(a,c)['ok'] else 2)
        run(a,c)
    except Exception as e:
        print('[DIAGNOSE_ONCE_FAILED]',type(e).__name__,str(e),file=sys.stderr,flush=True)
        traceback.print_exc()
        sys.exit(2)

if __name__=='__main__':main()
