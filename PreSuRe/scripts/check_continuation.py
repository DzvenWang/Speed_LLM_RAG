#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe');ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml
ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development');ap.add_argument('--force',action='store_true');a=ap.parse_args();cfg=load_yaml(ROOT/'configs/presure.yaml');exp=Path(cfg['project']['exp_root']);sg=cfg['experiment']['subgraph']
def latest_summary(v):
 p=exp/v/sg/a.stage;fs=sorted(p.glob('*/summary.json'),key=lambda x:x.stat().st_mtime) if p.exists() else []
 if not fs: raise RuntimeError(f'No {v} summary found for {a.stage}')
 return json.loads(fs[-1].read_text()),fs[-1]
F,fp=latest_summary('OracleF');ri=F.get('relative_improvement',{});g=cfg['oracle']['continuation_gate'];mt=float(g.get('region_masked_min_improvement',0.03));st=float(g.get('region_support_min_improvement',0.04));ok=float(ri.get('masked',0))>=mt or float(ri.get('support',0))>=st
print('OracleF summary:',fp);print(f"OracleF masked={float(ri.get('masked',0)):.4%} support={float(ri.get('support',0)):.4%}");print(f"Region-routing gate masked>={mt:.1%} OR support>={st:.1%}")
if ok: print('LOCAL ROUTING CONTINUATION GATE PASS');sys.exit(0)
if a.force: print('LOCAL ROUTING CONTINUATION GATE FORCED');sys.exit(0)
print('LOCAL ROUTING CONTINUATION GATE BLOCKED: graph-region×horizon headroom is too small; inspect Oracle-E/F and region construction.');sys.exit(2)
