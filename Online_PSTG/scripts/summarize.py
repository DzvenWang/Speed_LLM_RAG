#!/usr/bin/env python3
from pathlib import Path
import argparse,json,pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root'); a=ap.parse_args(); rows=[]
    for p in Path(a.root).glob('*/summary.json'):
        j=json.loads(p.read_text(encoding='utf-8'))
        rows.append({'run':p.parent.name,'variant':j.get('variant'),'queries':j.get('queries'),'updates':j.get('updates'),
                     'static_masked_mae':j['static']['masked_mae'],'online_masked_mae':j['online']['masked_mae'],'gain_pct':j['gains_pct']['masked_mae'],
                     'overall_gain_pct':j['gains_pct']['overall_mae'],'support_gain_pct':j['gains_pct']['support_mae']})
    df=pd.DataFrame(rows).sort_values('gain_pct',ascending=False) if rows else pd.DataFrame()
    print(df.to_string(index=False) if len(df) else 'No summary.json found')
if __name__=='__main__': main()
