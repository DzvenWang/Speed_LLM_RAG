#!/usr/bin/env python3
"""Evaluate a phase checkpoint independently, including optional held-out severity tests."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--phase',choices=['retrieval','recovery','injection','partial','full','candidate','controller','best'],required=True)
p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--severities',type=int,nargs='+',default=[20,40,60,90]);p.add_argument('--mode',default=None)
a=p.parse_args();ck=Path(a.run_dir)/('best.pt' if a.phase=='best' else 'candidate_best.pt' if a.phase=='candidate' else 'controller_best.pt' if a.phase=='controller' else a.phase+'_best.pt')
if not ck.is_file():raise FileNotFoundError(ck)
cmd=[sys.executable,str(Path(__file__).with_name('evaluate.py')),'--checkpoint',str(ck),'--split',a.split,'--output',str(Path(a.run_dir)/'stages'/a.phase/'independent_evaluation')]
if a.mode:cmd+=['--mode',a.mode]
if a.split=='test':cmd+=['--severities',*map(str,a.severities)]
subprocess.check_call(cmd)
