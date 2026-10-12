#!/usr/bin/env python3
"""Fixed-checkpoint inference interventions with comparable JSON/CSV reports.

For causal module attribution use additional MATCHED RETRAINED ablations; this
script changes inference behavior on one already-trained checkpoint.
"""
import argparse,csv,json,subprocess,sys
from datetime import datetime
from pathlib import Path
import _bootstrap

MODES=('full','retrieval_off','no_memory','similarity_memory','utility_off',
       'random_memory','cross_query_shuffle','weight_shuffle','null_evidence')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90])
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--output');p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    root=Path(a.output) if a.output else (Path(a.checkpoint).resolve().parent/'ablations'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    root.mkdir(parents=True,exist_ok=True)
    records=[]
    for mode in a.modes:
        dest=root/mode
        cmd=[sys.executable,str(Path(__file__).with_name('evaluate.py')),
             '--checkpoint',str(a.checkpoint),'--split',a.split,'--mode',mode,
             '--output',str(dest)]
        if a.smoke:cmd.append('--smoke')
        if a.split=='test':cmd+=['--severities',*map(str,a.severities)]
        print('FARE_ABLATION_START '+mode,flush=True)
        subprocess.check_call(cmd)
        severities=[None] if a.split=='val' else a.severities
        for severity in severities:
            tag='val_'+mode if severity is None else f's{severity:02d}_{mode}'
            path=(dest if severity is None else dest/f's{severity:02d}')/f'{tag}_summary.json'
            j=json.loads(path.read_text(encoding='utf-8'))
            row={'mode':mode,'split':a.split,'severity':('' if severity is None else severity),
                 **j['final'],'base_MAE':j['base']['MAE'],
                 'candidate_MAE':j['candidate']['MAE'],
                 'gate_mean':j.get('gate_mean'),
                 'valid_retrieval':j.get('retrieval_diagnostics',{}).get('valid_group_fraction'),
                 'innovation_abs':j.get('retrieval_diagnostics',{}).get('innovation_abs'),
                 'retrieved_token_abs':j.get('retrieval_diagnostics',{}).get('retrieved_token_abs')}
            records.append(row)
            print('FARE_ABLATION_RESULT '+json.dumps(row),flush=True)
    (root/'ablation_summary.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    keys=list(dict.fromkeys(k for row in records for k in row))
    with (root/'ablation_summary.csv').open('w',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=keys);w.writeheader();w.writerows(records)
    print(f'FARE_ABLATION_PASS output={root} rows={len(records)}',flush=True)
if __name__=='__main__':main()
