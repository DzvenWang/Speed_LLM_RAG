#!/usr/bin/env python3
from pathlib import Path
import argparse,json,pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root'); a=ap.parse_args(); rows=[]
    for p in Path(a.root).glob('*/summary.json'):
        j=json.loads(p.read_text(encoding='utf-8')); r=j['rag_config']
        rows.append({'run':p.parent.name,'queries':j['queries'],'k':r['top_k'],'memory':r['memory_size'],'alpha':r['alpha'],
                     'base_masked_mae':j['base']['masked_mae'],'rag_masked_mae':j['rag']['masked_mae'],'gain_pct':j['gains_pct']['masked_mae'],
                     'oracle_gain_pct':j['oracle_topk_diagnostic']['gain_pct']})
    df=pd.DataFrame(rows).sort_values('gain_pct',ascending=False) if rows else pd.DataFrame()
    print(df.to_string(index=False) if len(df) else 'No summary.json found')
if __name__=='__main__': main()
