#!/usr/bin/env python3
from pathlib import Path
import argparse,json,pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root'); a=ap.parse_args(); rows=[]
    for p in Path(a.root).glob('*/summary.json'):
        j=json.loads(p.read_text()); ors=j.get('retrieved_entry_oracle_by_k_vs_slow_pct',{})
        rows.append({'run':p.parent.name,'variant':j['variant'],'masked_mae':j['online_presure_v32']['masked_mae'],
                     'gain_vs_frozen_pct':j['final_gains_pct']['masked_mae'],'rag_vs_slow_pct':j['rag_increment_vs_slow_pct']['masked_mae'],
                     'perfect_oracle_pct':j['perfect_completion_oracle_vs_slow_pct']['masked_mae'],
                     'entry@8_pct':ors.get('8',{}).get('masked_mae'),'entry@16_pct':ors.get('16',{}).get('masked_mae'),
                     'entry@32_pct':ors.get('32',{}).get('masked_mae'),'entry@64_pct':ors.get('64',{}).get('masked_mae'),
                     'fill_coverage':j['missing_completion']['fill_coverage']})
    if rows: print(pd.DataFrame(rows).sort_values(['variant','masked_mae']).to_string(index=False))
if __name__=='__main__': main()
