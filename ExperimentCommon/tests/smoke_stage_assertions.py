import pandas as pd
from experiment_common.cache import _assert_split_contract

families = [
    'entry_random','sensor_outage','temporal_outage',
    'regional_outage','directional_outage',
]
rows=[]
for f in families:
    for i in range(3):
        rows.append({'view_group_id':f'{f}_{i}','family':f,'severity':0.4})
df=pd.DataFrame(rows)
cfg={
    'experiment':{'severity':0.4},
    'protocol':{
        'name':'protocol_s40',
        'stage_assertions':{
            'formal':{
                'train_rows':15,
                'train_unique_groups':15,
                'family_rows':{'train':3},
            }
        }
    }
}
_assert_split_contract(cfg,'formal',{'train':df})

bad=df.iloc[:-1].copy()
try:
    _assert_split_contract(cfg,'formal',{'train':bad})
except RuntimeError:
    pass
else:
    raise AssertionError('Unbalanced formal split was not rejected')
print('SMOKE STAGE ASSERTIONS PASS')
