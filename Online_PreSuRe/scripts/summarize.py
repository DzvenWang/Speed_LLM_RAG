#!/usr/bin/env python3
from pathlib import Path
import argparse,json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('summary'); a=ap.parse_args(); j=json.loads(Path(a.summary).read_text())
    print('variant',j['variant']); print('slow gain %',j['slow_gains_pct']['masked_mae']); print('final gain %',j['final_gains_pct']['masked_mae']); print('RAG vs slow %',j['rag_increment_vs_slow_pct']['masked_mae'])
    print('perfect completion oracle vs slow %',j['perfect_completion_oracle_vs_slow_pct']['masked_mae'])
    print('retrieved case Top-8 oracle vs slow %',j['retrieved_case_oracle_vs_slow_pct']['masked_mae'])
    for k,v in sorted(j['retrieved_entry_oracle_by_k_vs_slow_pct'].items(),key=lambda z:int(z[0])):
        print(f'entry oracle @{k} vs slow %',v['masked_mae'])
    print('completion',json.dumps(j['missing_completion'],indent=2))
if __name__=='__main__': main()
