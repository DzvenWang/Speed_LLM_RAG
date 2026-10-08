#!/usr/bin/env python3
"""Review learning-to-retrieve and forecast trends in a FARE run's history.jsonl."""
import argparse,csv,json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--run-dir',required=True)
p.add_argument('--output')
a=p.parse_args()
path=Path(a.run_dir)/'history.jsonl'
if not path.is_file():raise FileNotFoundError(path)
records=[]
for line in path.read_text(encoding='utf-8').splitlines():
    if not line.strip():continue
    data=json.loads(line)
    row={'phase':data['phase'],'epoch':data['epoch'],'seconds':data.get('seconds'),
         'score':data['score']}
    for split in ('train','val'):
        for col in ('ranking','ranking_eligible_fraction','ranking_selected_error',
                    'ranking_oracle_error','ranking_uniform_error','forecast',
                    'retrieval','state','innovation','retrieval_valid_groups',
                    'candidate_delta','final_gate_mean'):
            row[split+'_'+col]=data.get(split,{}).get(col)
    records.append(row)
    print('FARE_RANKING_EPOCH '+json.dumps(row),flush=True)
if not records:raise RuntimeError('history.jsonl has no epochs')
out=Path(a.output) if a.output else Path(a.run_dir)/'ranking_diagnostics.csv'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
print('FARE_RANKING_REPORT_PASS output='+str(out),flush=True)
