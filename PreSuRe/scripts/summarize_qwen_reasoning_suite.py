#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, json, sys
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml, write_json, stamp
from presure.need_gated import load_latest
from presure.pipeline import run_dir

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development'); ap.add_argument('--seed',type=int,default=10); a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure.yaml'); rows={}
    for size in ['0.5b','1.5b']:
        for mode in ['zero_shot','lora']:
            exp=f'QwenSupportValueProbe_{size}_{mode}'
            try:
                _,rd=load_latest(cfg['project']['exp_root'],exp,cfg['experiment']['subgraph'],a.stage)
                s=json.loads((rd/'summary.json').read_text())
                rows[f'{size}_{mode}']={'run_dir':str(rd),'recommendation':s['recommendation'],'masked':s['metrics']['qwen']['masked_triggered'],'decision':s['decision_diagnostics']}
            except Exception as e:
                rows[f'{size}_{mode}']={'missing':True,'error':repr(e)}
    rd=run_dir(cfg,'QwenSupportValueReasoningSuite',a.stage,stamp(),a.seed); rd.mkdir(parents=True,exist_ok=True)
    out={'variant':'QwenSupportValueReasoningSuite','stage':a.stage,'seed':a.seed,'models':rows,'test_evaluated':False}
    write_json(rd/'summary.json',out); print(json.dumps(out,indent=2,sort_keys=True)); print('QwenSupportValueReasoningSuite PASS run_dir='+str(rd))
if __name__=='__main__': main()
