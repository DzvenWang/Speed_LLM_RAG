#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from shared_protocol.sampler import balanced_unique_group_pick
fams=['temporal_outage','regional_outage','directional_outage','predictive_regional_outage','predictive_directional_outage']
rows=[]
for i in range(200):
    for f in fams:
        if not (i%11==0 and f=='predictive_directional_outage'):
            rows.append({'view_id':f'v{i}_{f}','view_group_id':f'g{i}','window_start':i,'family':f})
df=pd.DataFrame(rows); out=balanced_unique_group_pick(df,fams,100,10)
assert len(out)==100 and out.view_group_id.nunique()==100
assert set(out.family.value_counts())=={20}
print('SHARED PROTOCOL SAMPLER SMOKE PASS')
