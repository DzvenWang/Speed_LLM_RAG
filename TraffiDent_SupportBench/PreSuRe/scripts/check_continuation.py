#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe');ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml
ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development');ap.add_argument('--force',action='store_true');a=ap.parse_args();cfg=load_yaml(ROOT/'configs/presure.yaml');exp=Path(cfg['project']['exp_root']);sg=cfg['experiment']['subgraph']
def latest(v):
 p=exp/v/sg/a.stage;fs=sorted(p.glob('*/summary.json'),key=lambda x:x.stat().st_mtime) if p.exists() else []
 if not fs: raise RuntimeError(f'No {v} summary found for {a.stage}')
 return json.loads(fs[-1].read_text()),fs[-1]
E,ep=latest('OracleE');M3,m3p=latest('M3');ri=E.get('relative_improvement',{});node_headroom=float(ri.get('masked',0));
print('OracleE summary:',ep);print('M3 summary:',m3p);print(f'OracleE node-local masked headroom={node_headroom:.4%}')
if node_headroom>=0.05 or a.force:
 print('NODE-LOCAL CONTINUATION GATE PASS' if node_headroom>=0.05 else 'NODE-LOCAL CONTINUATION GATE FORCED');sys.exit(0)
print('NODE-LOCAL CONTINUATION GATE BLOCKED: Oracle-E node-local headroom is too small.');sys.exit(2)
