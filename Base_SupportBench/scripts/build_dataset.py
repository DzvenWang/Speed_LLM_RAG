#!/usr/bin/env python3
from __future__ import annotations
import argparse, datetime as dt, os, shutil, subprocess, sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.environ.get('SPEED_LLM_RAG_ROOT', ROOT.parent)).resolve()
for p in (ROOT,):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from supportbench.config import load_config
from supportbench.dataset import SupportBench
from shared_protocol.builder import build_stage, validate_stage, validate_paired_evaluations
from experiment_common.cache import prepare_cache, validate_cache

FAMILIES=['entry_random','sensor_outage','temporal_outage','regional_outage','directional_outage']
SEVS=[0.20,0.40,0.60,0.90]

def expand(v):
    if isinstance(v,str): return os.path.expandvars(v)
    if isinstance(v,dict): return {k:expand(x) for k,x in v.items()}
    if isinstance(v,list): return [expand(x) for x in v]
    return v

def _portable(v):
    """Persist configs without binding them to the machine that built the cache."""
    if isinstance(v, str):
        root = str(PROJECT)
        if v == root:
            return '${SPEED_LLM_RAG_ROOT}'
        if v.startswith(root + os.sep) or v.startswith(root + '/') or v.startswith(root + '\\'):
            rel = v[len(root):].lstrip('/\\').replace('\\','/')
            return '${SPEED_LLM_RAG_ROOT}/' + rel
        return v
    if isinstance(v, dict): return {k:_portable(x) for k,x in v.items() if not str(k).startswith('_')}
    if isinstance(v, list): return [_portable(x) for x in v]
    return v

def dump_cfg(obj,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(_portable(obj),sort_keys=False),encoding='utf-8')
    return path

def protocol_cfg(dataset,sev,base,config_path):
    tag=f's{int(round(sev*100)):02d}'
    train=abs(sev-0.40)<1e-9
    cfg={
      'version':'2.1.0',
      'project':{
        'root':str(PROJECT),'protocol_root':str(ROOT),
        'artifact_root':str(PROJECT/'Base_SupportBench_artifacts'/'protocol'),
        'log_root':str(PROJECT/'Base_SupportBench_logs'),
        'benchmark_root':str(PROJECT/'data'/'processed'/'traffident'/'base_supportbench'),
        'python':'auto'},
      'source':{'supportbench_benchmark_version':'1.2.1'},
      'experiment':{'dataset':dataset,'history':12,'horizon':12,'severity':sev,'distribution':'id','seed':int(base['experiment']['seed']),'families':FAMILIES},
      'protocol':{
        'name':f'base_core5_{tag}','role':'backbone_baseline_base_data',
        'allowed_stages':['development','formal','evaluation'] if train else ['evaluation'],
        'paired_group_id':f'{dataset}_base_core5_multiseverity_v1','paired_reference_severity':0.40,
        'paired_evaluation_seed_offset':20,'unique_view_group':True,'family_balanced':True,
        'formal_train_policy':'exact_balanced_unique_groups','formal_train_exact_family_balance':True,
        'formal_reuses_development_val':True,'train_only_normalization':True,
        'normalized_missing_fill_value':0.0,'development_val_pool_policy':'all_eligible',
        'evaluation_pool_policy':'all_eligible','evaluation_access_policy':'frozen_manifest_direct_access'},
      'stages':{
        'development':{'train_size':int(base['experiment']['development_train_size']),'val_size':int(base['experiment']['development_val_size'])},
        'formal':{'train_size':int(base['experiment']['formal_train_size']),'val_size':int(base['experiment']['formal_val_size'])},
        'evaluation':{'test_size':int(base['experiment']['evaluation_test_size'])}}
    }
    cfg['_config_path']=str(config_path)
    return cfg

def cache_cfg(pcfg,stage):
    sev=float(pcfg['experiment']['severity']); tag=f's{int(round(sev*100)):02d}'
    train=abs(sev-.40)<1e-9
    assertions={
      'development':{'train_rows':12000,'train_unique_groups':12000,'val_rows':5000,'val_unique_groups':5000,'family_rows':{'train':2400,'val':1000}},
      'formal':{'train_rows':50000,'train_unique_groups':50000,'val_rows':5000,'val_unique_groups':5000,'family_rows':{'train':10000,'val':1000}},
      'evaluation':{'test_rows':5000,'test_unique_groups':5000,'family_rows':{'test':1000}},
    }
    return {
      'version':'1.2.1','source':{'supportbench_benchmark_version':'1.2.1'},
      'project':{'root':str(PROJECT),'cache_root':str(PROJECT/'ExperimentCache'),'log_root':str(PROJECT/'Base_SupportBench_logs'),
                 'benchmark_root':str(PROJECT/'data'/'processed'/'traffident'/'base_supportbench'),
                 'protocol_artifact_root':str(PROJECT/'Base_SupportBench_artifacts'/'protocol')},
      'experiment':{'subgraph':pcfg['experiment']['dataset'],'history':12,'horizon':12,'severity':sev,'normalized_fill_value':0.0,'families':FAMILIES},
      'protocol':{'name':pcfg['protocol']['name'],'version':'2.1.0','paired_group_id':pcfg['protocol']['paired_group_id'],
                  'allowed_stages':pcfg['protocol']['allowed_stages'],'no_local_sampling':True,'stage_assertions':assertions},
      'contract':{'cache_schema_version':3,'data_contract_revision':'traffident_experiment_cache_v1','time_feature_revision':'traffident_time_features_v1','metrics_revision':'traffident_metrics_v2','prediction_contract_revision':'traffident_prediction_v1'},
      'time_features':{'allow_index_modulo_fallback':True,'index_origin_dow':0,'expected_freq_minutes':5,'expected_steps_per_day':288}
    }

def main():
    ap=argparse.ArgumentParser(description='Build Backbone/Baseline-ready Base_SupportBench data.')
    ap.add_argument('--dataset',default='TraffiDent_Net_06',choices=[f'TraffiDent_Net_{x}' for x in ['01','02','03','04','06']])
    ap.add_argument('--config',default=str(ROOT/'configs'/'base_supportbench.yaml')); ap.add_argument('--target',choices=['speed','flow'],default='speed')
    ap.add_argument('--force',action='store_true')
    ap.add_argument('--reuse-base', action='store_true', help='Reuse existing core/windows/sensing views and rebuild only protocol + ExperimentCache.')
    a=ap.parse_args()
    os.environ['SPEED_LLM_RAG_ROOT']=str(PROJECT)
    base=load_config(a.config)
    ds=a.dataset; target=a.target
    # Resolve a target-specific build config so Speed and Flow never share benchmark/cache artifacts.
    base['source']['primary_channel']=target
    base['project']['benchmark_root']=str(PROJECT/'data'/'processed'/'traffident'/'base_supportbench'/target)
    base['project']['diagnostics_root']=str(PROJECT/'data'/'diagnostics'/'traffident'/'base_supportbench'/target)
    resolved_build_cfg=PROJECT/'Base_SupportBench_artifacts'/'resolved_configs'/target/ds/'base_supportbench.yaml'
    dump_cfg(base,resolved_build_cfg)
    a.config=str(resolved_build_cfg)
    logs=PROJECT/'Base_SupportBench_logs'; logs.mkdir(parents=True,exist_ok=True)
    stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S'); log=logs/f'BaseSupportBench.{ds}.{target}.{stamp}.live.log'
    lf=log.open('w',encoding='utf-8',buffering=1)
    def emit(*xs):
        m=' '.join(map(str,xs)); print(m,flush=True); lf.write(m+'\n'); lf.flush()
    def run(script):
        cmd=[sys.executable,'-u',str(script),'--config',str(Path(a.config).resolve()),'--subgraph',ds]
        # Child build scripts import the local supportbench package.  Make the
        # package root explicit so the same launcher works on Windows and Gadi
        # without requiring users to set PYTHONPATH manually.  Preserve any
        # caller-provided PYTHONPATH for external dependencies.
        child_env=os.environ.copy()
        old_pythonpath=child_env.get('PYTHONPATH','')
        child_env['PYTHONPATH']=str(ROOT) if not old_pythonpath else str(ROOT)+os.pathsep+old_pythonpath
        emit('RUN',' '.join(cmd)); p=subprocess.Popen(cmd,cwd=str(ROOT),env=child_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        for line in p.stdout:
            print(line,end='',flush=True); lf.write(line); lf.flush()
        rc=p.wait()
        if rc: raise subprocess.CalledProcessError(rc,cmd)
    try:
        emit('LOG_FILE',log); emit('PROJECT_ROOT',PROJECT); emit('DATASET',ds); emit('TARGET',target); emit('START_TIME',dt.datetime.now().isoformat(timespec='seconds'))
        emit('SCOPE canonical core + windows + five general sensing families + frozen protocol + ExperimentCache')
        emit('EXCLUDED support memory, predictive sensing, challenge pool, retrieval/RAG/LLM derived artifacts')
        benchmark_ds = Path(base['project']['benchmark_root'])/ds
        diagnostics_ds = Path(base['project']['diagnostics_root'])/ds
        protocol_ds = PROJECT/'Base_SupportBench_artifacts'/'protocol'/target/ds
        cache_ds = PROJECT/'ExperimentCache'/target/ds
        if a.reuse_base:
            required = [benchmark_ds/'core', benchmark_ds/'windows', benchmark_ds/'views'/'view_manifest.csv']
            missing = [str(p) for p in required if not p.exists()]
            if missing:
                raise FileNotFoundError('Cannot --reuse-base; missing completed base artifacts: ' + ', '.join(missing))
            emit('REUSE_BASE', benchmark_ds)
            for p in [protocol_ds, cache_ds]:
                if p.exists(): emit('REMOVE',p); shutil.rmtree(p)
            emit('SKIP 01_build_core.py / 02_build_windows.py / 03_build_sensing_views.py (reuse existing base)')
        else:
            if a.force:
                for p in [benchmark_ds, diagnostics_ds, protocol_ds, cache_ds]:
                    if p.exists(): emit('REMOVE',p); shutil.rmtree(p)
            for n in ['01_build_core.py','02_build_windows.py','03_build_sensing_views.py']:
                emit('==========',n,'START =========='); run(ROOT/'build'/n); emit('==========',n,'DONE ==========')
        bench=SupportBench(ds,root=base['project']['benchmark_root'])
        cfgdir=PROJECT/'Base_SupportBench_artifacts'/'resolved_configs'/target/ds
        pcfgs=[]
        for sev in SEVS:
            cp=cfgdir/f'protocol_s{int(sev*100):02d}.yaml'; pcfg=protocol_cfg(ds,sev,base,cp); pcfg['experiment']['target']=target; pcfg['protocol']['name']=f'base_core5_{target}_s{int(sev*100):02d}'; pcfg['protocol']['paired_group_id']=f'{ds}_base_core5_{target}_multiseverity_v1'; pcfg['project']['artifact_root']=str(PROJECT/'Base_SupportBench_artifacts'/'protocol'/target); pcfg['project']['benchmark_root']=base['project']['benchmark_root']; dump_cfg({k:v for k,v in pcfg.items() if k!='_config_path'},cp); pcfgs.append(pcfg)
        main40=next(c for c in pcfgs if abs(c['experiment']['severity']-.40)<1e-9)
        for st in ['development','formal']:
            emit('========== PROTOCOL',st.upper(),'s40 START =========='); build_stage(bench,main40,st,force=(a.force or a.reuse_base)); validate_stage(main40,st); emit('PROTOCOL',st,'PASS')
        # Build reference s40 evaluation first so one paired manifest is frozen, then resolve other severities.
        order=[main40]+[c for c in pcfgs if c is not main40]
        for c in order:
            tag=f"s{int(c['experiment']['severity']*100):02d}"; emit('========== PROTOCOL EVALUATION',tag,'START =========='); build_stage(bench,c,'evaluation',force=(a.force or a.reuse_base)); validate_stage(c,'evaluation'); emit('PROTOCOL evaluation',tag,'PASS')
        validate_paired_evaluations(pcfgs); emit('PAIRED EVALUATION PASS')
        # Materialize model-ready caches and persist the exact contracts consumed by Backbone/Baseline.
        for st in ['development','formal']:
            cc=cache_cfg(main40,st); cc['project']['cache_root']=str(PROJECT/'ExperimentCache'/target); cc['project']['benchmark_root']=base['project']['benchmark_root']; cc['project']['protocol_artifact_root']=str(PROJECT/'Base_SupportBench_artifacts'/'protocol'/target); cc['experiment']['target']=target; cc['experiment']['target_channel_index']=['flow','occupancy','speed'].index(target); dump_cfg(cc,cfgdir/f'cache_{st}.yaml'); emit('========== CACHE',st.upper(),'START =========='); prepare_cache(cc,st,force=(a.force or a.reuse_base),allow_locked_evaluation_materialization=True); validate_cache(cc,st,allow_locked_evaluation_validation=True); emit('CACHE',st,'PASS')
        for c in pcfgs:
            cc=cache_cfg(c,'evaluation'); cc['project']['cache_root']=str(PROJECT/'ExperimentCache'/target); cc['project']['benchmark_root']=base['project']['benchmark_root']; cc['project']['protocol_artifact_root']=str(PROJECT/'Base_SupportBench_artifacts'/'protocol'/target); cc['experiment']['target']=target; cc['experiment']['target_channel_index']=['flow','occupancy','speed'].index(target); tag=f"s{int(c['experiment']['severity']*100):02d}"; dump_cfg(cc,cfgdir/f'cache_evaluation_{tag}.yaml'); emit('========== CACHE EVALUATION',tag,'START =========='); prepare_cache(cc,'evaluation',force=(a.force or a.reuse_base),allow_locked_evaluation_materialization=True); validate_cache(cc,'evaluation',allow_locked_evaluation_validation=True); emit('CACHE evaluation',tag,'PASS')
        emit('BASE_SUPPORTBENCH BUILD PASS'); emit('END_TIME',dt.datetime.now().isoformat(timespec='seconds'))
    except Exception as e:
        emit('BASE_SUPPORTBENCH BUILD FAILED',type(e).__name__,e); emit('END_TIME',dt.datetime.now().isoformat(timespec='seconds')); raise
    finally: lf.close()
if __name__=='__main__': main()
