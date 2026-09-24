#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from presure.p1_config import load_config
from presure.p1_paths import run_root
from presure.p1_data import VARIANTS
from presure.utils import write_json,log

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development');ap.add_argument('--run-tag',required=True);a=ap.parse_args();cfg=load_config(ROOT);rr=run_root(cfg,a.stage,a.run_tag); rows=[]
    for v in VARIANTS:
        p=rr/v/'summary.json'
        if p.exists():
            j=json.loads(p.read_text()); rows.append({'variant':v,'masked_mae':j['model_dev']['masked_mae'],'gain_pct':j['dev_gain_pct'],'params':j['trainable_params'],'corr_abs_norm':j['correction_abs_mean_norm']})
    if not rows: raise RuntimeError(f'No P1 summaries under {rr}')
    base=json.loads((rr/rows[0]['variant']/'summary.json').read_text())['baseline_dev']
    out={'baseline_patchstg':base,'variants':rows,'retrieval_value_tests':{}}
    by={r['variant']:r for r in rows}
    if 'adapter_only' in by and 'history_future' in by: out['retrieval_value_tests']['history_future_vs_adapter_only_masked_mae_delta']=by['adapter_only']['masked_mae']-by['history_future']['masked_mae']
    if 'shuffled_support' in by and 'history_future' in by: out['retrieval_value_tests']['history_future_vs_shuffled_masked_mae_delta']=by['shuffled_support']['masked_mae']-by['history_future']['masked_mae']
    if 'history_only' in by and 'history_future' in by: out['retrieval_value_tests']['future_evolution_increment_masked_mae_delta']=by['history_only']['masked_mae']-by['history_future']['masked_mae']
    go=(('adapter_only' in by and 'history_future' in by and by['history_future']['masked_mae'] < by['adapter_only']['masked_mae']) and ('shuffled_support' in by and by['history_future']['masked_mae'] < by['shuffled_support']['masked_mae']))
    out['go_no_go']={'retrieval_evidence_beats_adapter_only_and_shuffled':bool(go),'decision':'PROCEED_TO_P2' if go else 'STOP_AND_DIAGNOSE_P1'}
    write_json(rr/'P1_SUMMARY.json',out)
    log('P1 SUMMARY baseline_masked_MAE={:.6f}'.format(base['masked_mae']))
    for r in rows: log('{:<18} masked={:.6f} gain={:+.3f}% corr={:.6f} params={}'.format(r['variant'],r['masked_mae'],r['gain_pct'],r['corr_abs_norm'],r['params']))
    log('P1 DECISION '+out['go_no_go']['decision'])
    log(f'P1_SUMMARY={rr/"P1_SUMMARY.json"}')
if __name__=='__main__':main()
