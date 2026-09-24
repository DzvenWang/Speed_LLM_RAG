#!/usr/bin/env python3
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development'); ap.add_argument('--max-queries',type=int,default=None); ap.add_argument('--project-root',default=None); a=ap.parse_args()
    variants=['slow_only','rag_similarity_global','rag_similarity_compositional','rag_utility_compositional']
    for v in variants:
        cmd=[sys.executable,str(ROOT/'scripts'/'run_pipeline.py'),'--stage',a.stage,'--variant',v]
        if a.max_queries is not None: cmd+=['--max-queries',str(a.max_queries)]
        if a.project_root: cmd+=['--project-root',a.project_root]
        print('$ '+' '.join(cmd),flush=True); subprocess.run(cmd,check=True)
if __name__=='__main__': main()
