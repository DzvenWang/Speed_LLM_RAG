#!/usr/bin/env python3
"""Fail-closed paired protocol check. Runs on CPU after both projects finish."""
import argparse,json,sys
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--dataset',default='TraffiDent_Net_06');p.add_argument('--seed',type=int,default=10)
    a=p.parse_args();root=Path(a.root);out={};error=False
    for target in ('speed','flow'):
        for sev in (20,40,60,90):
            left=root/'TCRR-STLA_exps'/'diagnostics_v3'/a.dataset/f'seed{a.seed}'/target/f's{sev}'/'manifest.json'
            right=root/'PCFT-STLA_exps'/'diagnostics_v3'/a.dataset/f'seed{a.seed}'/target/f's{sev}'/'manifest.json'
            if not(left.is_file() and right.is_file()):
                out[f'{target}_s{sev}']={'status':'MISSING','tcrr':str(left),'pcft':str(right)};error=True;continue
            t=json.loads(left.read_text());p=json.loads(right.read_text())
            diff={k:{'TCRR':t['protocol']['field_sha256'].get(k),'PCFT':p['protocol']['field_sha256'].get(k)} for k in t['protocol']['field_sha256'] if t['protocol']['field_sha256'].get(k)!=p['protocol']['field_sha256'].get(k)}
            bmatch=t['base_raw_sha256']==p['base_raw_sha256']
            # Locate actual Query-level mismatches for actionable repair.
            qa=left.parent/'protocol_queries.csv';qb=right.parent/'protocol_queries.csv'
            samples={}
            if qa.is_file() and qb.is_file():
                A=pd.read_csv(qa).set_index('query_idx');B=pd.read_csv(qb).set_index('query_idx')
                for col in ('window_start','x_sha256','mask_sha256','tod_sha256','dow_sha256',
                            'y_raw_sha256','future_valid_sha256','direct_mask_sha256','base_prediction_sha256'):
                    if col not in A or col not in B:continue
                    bad=A[col].reindex(A.index).astype(str).ne(B[col].reindex(A.index).astype(str))
                    samples[col]={'mismatch_count':int(bad.sum()),'first_query_indices':[int(x) for x in bad[bad].index[:8]]}
            if diff or not bmatch:error=True
            out[f'{target}_s{sev}']={'status':'PASS' if not diff and bmatch else 'MISMATCH',
                'input_mismatch':diff,'query_mismatch_samples':samples,'base_forecast_identical':bmatch,
                'tcrr_backbone_sha256':t['backbone_sha256'],'pcft_backbone_sha256':p['backbone_sha256'],
                'backbone_same':t['backbone_sha256']==p['backbone_sha256']}
            if not out[f'{target}_s{sev}']['backbone_same']:
                out[f'{target}_s{sev}']['status']='MISMATCH';error=True
    path=root/'TCRR_PCFT_v3_protocol_comparison.json'
    path.write_text(json.dumps({'schema':'paired-diagnostic-protocol-v1','results':out,'status':'FAIL' if error else 'PASS'},indent=2))
    print(json.dumps(out,indent=2));print('PROTOCOL_COMPARISON_'+('FAIL' if error else 'PASS'),path)
    sys.exit(1 if error else 0)
if __name__=='__main__':main()
