#!/usr/bin/env python3
"""Compare separately RETRAINED v9.2 variants, only with identical Normal and cohort.

Usage: python scripts/compare_ablation_v92.py /path/run1 /path/run2 ...
Does not train any variant itself. Refuses to infer a retrained effect from toggles.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch

KEYS=('normal_core.query.','normal_core.match.','normal_core.normal_fusion.',
      'normal_core.normal_scale.','normal_core.none_normal')

def normal_signature(run,include_graph=False):
    p=run/'best_normal.pt'
    ck=torch.load(p,map_location='cpu',weights_only=False)
    h=hashlib.sha256()
    for name,value in sorted(ck['model'].items()):
        if name.startswith(KEYS) or (include_graph and name.startswith('normal_core.graph.')):
            h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def compare(runs):
    summary=[]
    for folder in map(Path,runs):
        info=json.loads((folder/'run_identity.json').read_text())
        diag=json.loads((folder/'diagnostics/summary.json').read_text())
        selected=diag['selected_stage']
        value=diag['stages'][selected]['results']['full']
        summary.append(dict(run=str(folder.resolve()),target=info['target'],dataset=info['dataset'],
            seed=info['seed'],backbone_sha=info['backbone_sha256'],
            evidence_root=info['evidence_root'],cohort_sha=diag['cohort_sha256'],
            normal_sha=normal_signature(folder),graph_sha=normal_signature(folder,include_graph=True),
            variant=info.get('configured_variant'),
            cf_gain=value['gain_vs_normal_norm'],mae=value['mae_norm'],
            cf_gain_ci=value['gain_ci95_norm']))
    for key in ('target','dataset','seed','backbone_sha','evidence_root','cohort_sha','normal_sha','graph_sha'):
        values={str(r[key]) for r in summary}
        if len(values)!=1:raise RuntimeError(f'Incomparable ablation runs: {key} differs: {values}')
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('run_dirs',nargs='+')
    p.add_argument('--output',default='v92_retrained_ablation_comparison.json')
    a=p.parse_args();rows=compare(a.run_dirs)
    out=Path(a.output);out.write_text(json.dumps(rows,indent=2))
    with out.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print('V92_RETRAINED_ABLATION_COMPARISON_PASS',out,len(rows))
if __name__=='__main__':main()
