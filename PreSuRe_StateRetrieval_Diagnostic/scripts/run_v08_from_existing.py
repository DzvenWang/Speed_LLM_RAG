#!/usr/bin/env python3
from pathlib import Path
import argparse,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from srd.config import load_config

def ts():return time.strftime('%Y-%m-%d %H:%M:%S')
def emit(handles,text):
    print(text,end='',flush=True)
    for h in handles:h.write(text);h.flush()
def call(cmd,master,stage_log):
    cmd=[str(x) for x in cmd];stage_log=Path(stage_log);stage_log.parent.mkdir(parents=True,exist_ok=True);st=time.time();banner=f"\n[{ts()}] >> {' '.join(cmd)}\n";emit([master],banner)
    with stage_log.open('w',encoding='utf-8',errors='replace') as sl:
        sl.write(banner);sl.flush();p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        for line in p.stdout:emit([master,sl],line)
        rc=p.wait();trail=f"[{ts()}] STAGE_RETURN_CODE={rc} ELAPSED_SEC={time.time()-st:.1f}\n";emit([master,sl],trail)
        if rc:raise subprocess.CalledProcessError(rc,cmd)
def main():
    ap=argparse.ArgumentParser(description='Run v0.8 Intervention-Aware Utility from completed v0.6/v0.7 run')
    ap.add_argument('--stage',default='development',choices=['development','formal']);ap.add_argument('--run-tag',required=True);ap.add_argument('--checkpoint',default=None)
    ap.add_argument('--models',default='logistic_intervention,cost_sensitive_gate,histgb_mean_intervention,histgb_q20_intervention,risk_expected_intervention');ap.add_argument('--feature-sets',default='full')
    ap.add_argument('--scopes',default='1,3,6,12');ap.add_argument('--deep-top-k',type=int,default=20);ap.add_argument('--max-k',type=int,default=20);ap.add_argument('--protocol',default='expanding',choices=['expanding','blocked']);ap.add_argument('--cv-blocks',type=int,default=6)
    ap.add_argument('--patch-caps',default='1,2,4,6,8,12,16');ap.add_argument('--transport',default='affine_alignment');ap.add_argument('--eval-batch-size',type=int,default=64);ap.add_argument('--rebuild-v07-dataset',action='store_true');ap.add_argument('--rebuild-v08-dataset',action='store_true')
    a=ap.parse_args();
    if a.max_k>a.deep_top_k:raise ValueError('--max-k cannot exceed --deep-top-k because deep intervention features are only materialized for that candidate pool')
    cfg=load_config(ROOT);dataset=cfg['experiment']['subgraph'];run=Path(cfg['project']['exp_root'])/dataset/a.stage/a.run_tag;logs=Path(cfg['project']['log_root'])/dataset/a.stage/a.run_tag;logs.mkdir(parents=True,exist_ok=True)
    needed=[run/'forecast_utility_oracle'/'single_patch_candidate_actions.csv.gz',run/'patch_retrieval'/'patch_retrieval_topk.csv.gz',run/'forecast_utility_oracle'/'forecast_oracle_by_k.csv'];miss=[str(p) for p in needed if not p.exists()]
    if miss:raise RuntimeError(f'v0.6 run incomplete; missing {miss}')
    v07=run/'predictive_utility_selector'/'utility_action_dataset.csv.gz';v08=run/'intervention_utility_selector'/'intervention_action_dataset.csv.gz';ck=[] if not a.checkpoint else ['--checkpoint',a.checkpoint]
    master_path=logs/'v08_upgrade.log'
    with master_path.open('w',encoding='utf-8',errors='replace') as ml:
        emit([ml],f"[{ts()}] Intervention-Aware Predictive Utility v0.8.0\nRUN={run}\nLOGS={logs}\nDEEP_TOP_K={a.deep_top_k}\nMAX_K={a.max_k}\nPROTOCOL={a.protocol}\nMODELS={a.models}\nFEATURE_SETS={a.feature_sets}\n")
        try:
            if a.rebuild_v07_dataset or not v07.exists():
                call([sys.executable,ROOT/'scripts'/'07_build_predictive_utility_dataset.py','--run-dir',run,'--stage',a.stage,*ck],ml,logs/'12_v08_prepare_v07_dataset.log')
            else:emit([ml],f"[{ts()}] REUSE_V07_DATASET={v07}\n")
            rebuild_v08=bool(a.rebuild_v08_dataset or not v08.exists())
            meta_path=run/'intervention_utility_selector'/'dataset_summary.json'
            if not rebuild_v08 and meta_path.exists():
                import json
                meta=json.loads(meta_path.read_text(encoding='utf-8'))
                if int(meta.get('deep_top_k',-1))<int(a.deep_top_k) or str(meta.get('transport'))!=str(a.transport):
                    emit([ml],f"[{ts()}] V08_DATASET_CONFIG_MISMATCH existing_deep_top_k={meta.get('deep_top_k')} requested={a.deep_top_k} existing_transport={meta.get('transport')} requested_transport={a.transport}; REBUILD\n")
                    rebuild_v08=True
            if rebuild_v08:
                call([sys.executable,ROOT/'scripts'/'11_build_intervention_utility_dataset.py','--run-dir',run,'--stage',a.stage,*ck,'--transport',a.transport,'--deep-top-k',str(a.deep_top_k),'--eval-batch-size',str(a.eval_batch_size)],ml,logs/'13_v08_build_intervention_dataset.log')
            else:emit([ml],f"[{ts()}] REUSE_V08_DATASET={v08}\n")
            call([sys.executable,ROOT/'scripts'/'12_run_intervention_utility_selector.py','--run-dir',run,'--stage',a.stage,*ck,'--models',a.models,'--feature-sets',a.feature_sets,'--scopes',a.scopes,'--max-k',str(a.max_k),'--protocol',a.protocol,'--cv-blocks',str(a.cv_blocks),'--patch-caps',a.patch_caps,'--transport',a.transport,'--eval-batch-size',str(a.eval_batch_size)],ml,logs/'14_v08_selector.log')
            call([sys.executable,ROOT/'scripts'/'13_summarize_v08.py',run],ml,logs/'15_v08_summary.log')
            call([sys.executable,ROOT/'scripts'/'14_fit_final_intervention_selector.py',run,'--primary-scope','12','--max-k',str(a.max_k),'--patch-caps',a.patch_caps],ml,logs/'16_v08_final_fit.log')
            emit([ml],f"\n[{ts()}] V0.8_COMPLETE\nRUN={run}\nLOGS={logs}\n")
        except Exception as e:
            emit([ml],f"\n[{ts()}] V0.8_FAILED {type(e).__name__}: {e}\nLOGS={logs}\n");raise
if __name__=='__main__':main()
