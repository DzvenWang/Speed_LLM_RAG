#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from srd.config import load_config

def ts():return time.strftime('%Y-%m-%d %H:%M:%S')
def emit(handles,text):
    print(text,end='',flush=True)
    for h in handles:h.write(text);h.flush()
def call(cmd,master,stage_log):
    cmd=[str(x) for x in cmd]; stage_log=Path(stage_log); stage_log.parent.mkdir(parents=True,exist_ok=True); st=time.time(); banner=f"\n[{ts()}] >> {' '.join(cmd)}\n"; emit([master],banner)
    with stage_log.open('w',encoding='utf-8',errors='replace') as sl:
        sl.write(banner);sl.flush();p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        for line in p.stdout:emit([master,sl],line)
        rc=p.wait();trail=f"[{ts()}] STAGE_RETURN_CODE={rc} ELAPSED_SEC={time.time()-st:.1f}\n";emit([master,sl],trail)
        if rc:raise subprocess.CalledProcessError(rc,cmd)
def main():
    ap=argparse.ArgumentParser(description='Run v0.7 Predictive-Utility Selector from completed v0.6 run')
    ap.add_argument('--stage',default='development',choices=['development','formal']);ap.add_argument('--run-tag',required=True);ap.add_argument('--checkpoint',default=None)
    ap.add_argument('--models',default='similarity_rule,logistic_gate,ridge_utility,histgb_utility,two_stage_histgb');ap.add_argument('--scopes',default='1,3,6,12');ap.add_argument('--max-k',type=int,default=50)
    ap.add_argument('--protocol',default='expanding',choices=['expanding','blocked']);ap.add_argument('--cv-blocks',type=int,default=6);ap.add_argument('--patch-caps',default='1,2,4,6,8,12,16');ap.add_argument('--transport',default='affine_alignment');ap.add_argument('--eval-batch-size',type=int,default=64);ap.add_argument('--rebuild-dataset',action='store_true')
    a=ap.parse_args();cfg=load_config(ROOT);dataset=cfg['experiment']['subgraph'];run=Path(cfg['project']['exp_root'])/dataset/a.stage/a.run_tag;logs=Path(cfg['project']['log_root'])/dataset/a.stage/a.run_tag;logs.mkdir(parents=True,exist_ok=True)
    needed=[run/'forecast_utility_oracle'/'single_patch_candidate_actions.csv.gz',run/'patch_retrieval'/'patch_retrieval_topk.csv.gz',run/'forecast_utility_oracle'/'forecast_oracle_by_k.csv']
    miss=[str(p) for p in needed if not p.exists()]
    if miss:raise RuntimeError(f'v0.6 run incomplete; missing {miss}')
    master_path=logs/'v07_upgrade.log';data=run/'predictive_utility_selector'/'utility_action_dataset.csv.gz';ck=[] if not a.checkpoint else ['--checkpoint',a.checkpoint]
    with master_path.open('w',encoding='utf-8',errors='replace') as ml:
        emit([ml],f"[{ts()}] Predictive-Utility Gate + Reranker v0.7.1\nRUN={run}\nLOGS={logs}\nPROTOCOL={a.protocol}\nMODELS={a.models}\nSCOPES={a.scopes}\nMAX_K={a.max_k}\n")
        try:
            if a.rebuild_dataset or not data.exists():
                call([sys.executable,ROOT/'scripts'/'07_build_predictive_utility_dataset.py','--run-dir',run,'--stage',a.stage,*ck],ml,logs/'08_v07_build_dataset.log')
            else:emit([ml],f"[{ts()}] REUSE_DATASET={data}\n")
            call([sys.executable,ROOT/'scripts'/'08_run_predictive_utility_selector.py','--run-dir',run,'--stage',a.stage,*ck,'--models',a.models,'--scopes',a.scopes,'--max-k',str(a.max_k),'--protocol',a.protocol,'--cv-blocks',str(a.cv_blocks),'--patch-caps',a.patch_caps,'--transport',a.transport,'--eval-batch-size',str(a.eval_batch_size)],ml,logs/'09_v07_selector.log')
            call([sys.executable,ROOT/'scripts'/'09_summarize_v07.py',run],ml,logs/'10_v07_summary.log')
            call([sys.executable,ROOT/'scripts'/'10_fit_final_selector.py',run,'--primary-scope','12','--max-k',str(a.max_k),'--patch-caps',a.patch_caps],ml,logs/'11_v07_final_fit.log')
            emit([ml],f"\n[{ts()}] V0.7_COMPLETE\nRUN={run}\nLOGS={logs}\n")
        except Exception as e:
            emit([ml],f"\n[{ts()}] V0.7_FAILED {type(e).__name__}: {e}\nLOGS={logs}\n");raise
if __name__=='__main__':main()
