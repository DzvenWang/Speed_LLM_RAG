#!/usr/bin/env python3
from pathlib import Path
import argparse,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from online_presure.config import load_config
VARIANTS=['frozen','slow_only','rag_similarity_global','rag_similarity_compositional','rag_utility_compositional']

def tee(cmd,stage_log,master,mode='a'):
    stage_log.parent.mkdir(parents=True,exist_ok=True)
    with stage_log.open(mode,encoding='utf-8') as sf, master.open(mode,encoding='utf-8') as mf:
        h='$ '+' '.join(map(str,cmd))+'\n'; sf.write(h); mf.write(h); sf.flush(); mf.flush()
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line,end=''); sf.write(line); mf.write(line); sf.flush(); mf.flush()
        rc=p.wait()
        if rc: raise subprocess.CalledProcessError(rc,cmd)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--split',default=None)
    ap.add_argument('--variant',default='rag_utility_compositional',choices=VARIANTS); ap.add_argument('--project-root',default=None); ap.add_argument('--checkpoint',default=None)
    ap.add_argument('--max-queries',type=int,default=None); ap.add_argument('--replay-memory-size',type=int,default=None); ap.add_argument('--evidence-memory-size',type=int,default=None)
    ap.add_argument('--top-k',type=int,default=None); ap.add_argument('--prefilter-k',type=int,default=None); ap.add_argument('--pool-k',type=int,default=None); ap.add_argument('--node-top-k',type=int,default=None); ap.add_argument('--patch-top-k',type=int,default=None)
    ap.add_argument('--utility-probes',type=int,default=None); ap.add_argument('--utility-min-examples',type=int,default=None); ap.add_argument('--utility-helpful-prob-min',type=float,default=None); ap.add_argument('--utility-harmful-prob-max',type=float,default=None)
    ap.add_argument('--slow-batch-size',type=int,default=None); ap.add_argument('--slow-steps',type=int,default=None); ap.add_argument('--slow-lr',type=float,default=None)
    ap.add_argument('--recency-tau',type=float,default=None); ap.add_argument('--completion-max-gate',type=float,default=None); ap.add_argument('--max-external-weight',type=float,default=None)
    ap.add_argument('--preflight-only',action='store_true'); a=ap.parse_args(); cfg=load_config(ROOT,a.project_root)
    tag=f"online_presure_v330.{a.variant}.{time.strftime('%Y%m%d_%H%M%S')}"; ld=Path(cfg['project']['log_root'])/cfg['experiment']['subgraph']/a.stage/tag; ld.mkdir(parents=True,exist_ok=True)
    master=ld/'run.log'; (ld/'run_meta.json').write_text(json.dumps({'tag':tag,'stage':a.stage,'variant':a.variant,'argv':sys.argv,'created':time.strftime('%F %T')},indent=2),encoding='utf-8')
    common=[]
    if a.split: common += ['--split',a.split]
    if a.project_root: common += ['--project-root',a.project_root]
    if a.checkpoint: common += ['--checkpoint',a.checkpoint]
    tee([sys.executable,str(ROOT/'scripts'/'preflight.py'),'--stage',a.stage]+common,ld/'00_preflight.log',master,'w')
    if a.preflight_only: print(f'LOG_DIR={ld}'); return
    cmd=[sys.executable,str(ROOT/'scripts'/'run_online.py'),'--stage',a.stage,'--variant',a.variant,'--run-tag',tag]+common
    pairs=[('--max-queries',a.max_queries),('--replay-memory-size',a.replay_memory_size),('--evidence-memory-size',a.evidence_memory_size),
           ('--top-k',a.top_k),('--prefilter-k',a.prefilter_k),('--pool-k',a.pool_k),('--node-top-k',a.node_top_k),('--patch-top-k',a.patch_top_k),
           ('--utility-probes',a.utility_probes),('--utility-min-examples',a.utility_min_examples),('--utility-helpful-prob-min',a.utility_helpful_prob_min),('--utility-harmful-prob-max',a.utility_harmful_prob_max),
           ('--slow-batch-size',a.slow_batch_size),('--slow-steps',a.slow_steps),('--slow-lr',a.slow_lr),('--recency-tau',a.recency_tau),
           ('--completion-max-gate',a.completion_max_gate),('--max-external-weight',a.max_external_weight)]
    for f,v in pairs:
        if v is not None: cmd += [f,str(v)]
    tee(cmd,ld/'01_online_stream.log',master,'a'); print(f'LOG_DIR={ld}')
if __name__=='__main__': main()
