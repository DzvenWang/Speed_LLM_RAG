#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts._common import setup, run_dir
from srd.utils import write_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development'); ap.add_argument('--run-tag',required=True)
    a=ap.parse_args(); cfg=setup(a.stage); rd=run_dir(cfg,a.stage,a.run_tag)
    d=rd/'fragment_completion_diagnostic'; sm=pd.read_csv(d/'summary_table.csv')
    scope=int(cfg['fragment_diagnostic'].get('primary_scope',12)); g=sm[sm.recent_steps==scope].copy()
    oracle=g[g['mode']=='oracle_completion'].sort_values('forecast_gain_pct',ascending=False)
    realiz=g[g['mode'].isin(['local_observed','local_softmax','global_top1'])].sort_values('forecast_gain_pct',ascending=False)
    whole=oracle[oracle.scheme=='oracle_completion:whole_patch']
    best_o=oracle.iloc[0].to_dict() if len(oracle) else {}
    best_r=realiz.iloc[0].to_dict() if len(realiz) else {}
    whole_gain=float(whole.forecast_gain_pct.iloc[0]) if len(whole) else float('nan')
    whole_comp=float(whole.completion_improvement_pct.iloc[0]) if len(whole) else float('nan')
    best_o_gain=float(best_o.get('forecast_gain_pct',float('nan')))
    best_o_comp=float(best_o.get('completion_improvement_pct',float('nan')))
    payload={
        'version':'0.9.0','primary_scope':scope,
        'whole_patch_completion_oracle_forecast_gain_pct':whole_gain,
        'whole_patch_completion_oracle_completion_improvement_pct':whole_comp,
        'best_fragment_oracle_scheme':best_o.get('scheme'),
        'best_fragment_oracle_forecast_gain_pct':best_o_gain,
        'best_fragment_oracle_completion_improvement_pct':best_o_comp,
        'fragment_oracle_gain_over_whole_patch_pp':best_o_gain-whole_gain if pd.notna(whole_gain) else float('nan'),
        'best_realizable_scheme':best_r.get('scheme'),
        'best_realizable_forecast_gain_pct':float(best_r.get('forecast_gain_pct',float('nan'))),
        'best_realizable_completion_improvement_pct':float(best_r.get('completion_improvement_pct',float('nan'))),
        'fragment_oracle_reaches_2pct':bool(best_o_gain>=2.0) if pd.notna(best_o_gain) else False,
        'fragment_oracle_reaches_4pct':bool(best_o_gain>=4.0) if pd.notna(best_o_gain) else False,
    }
    write_json(rd/'fragment_diagnostic_summary.json',payload)
    lines=[
      '# PreSuRe State-Retrieval Fragment Completion Diagnostic v0.9.0','',
      '## Primary question','',
      'Can incomplete traffic history be improved by retrieving **valuable local fragments** and composing them before PatchSTG forecasting?','',
      '## Core numbers','',
      f"- Whole-patch completion Oracle forecast gain: **{whole_gain:.3f}%**.",
      f"- Best fragment-composition Oracle: **{best_o.get('scheme','n/a')}**, forecast gain **{best_o_gain:.3f}%**.",
      f"- Best fragment-composition Oracle completion improvement: **{best_o_comp:.3f}%**.",
      f"- Fragment composition gain over whole-patch Oracle: **{payload['fragment_oracle_gain_over_whole_patch_pp']:.3f} pp**.",
      f"- Best inference-available fragment composition: **{best_r.get('scheme','n/a')}**, forecast gain **{payload['best_realizable_forecast_gain_pct']:.3f}%**.",'',
      '## Interpretation','',
      '- Oracle fragment selection uses only hidden **history** truth to measure the ceiling of missing-state completion; it never uses the future forecasting target.',
      '- `local_observed` and `local_softmax` use only observed query context and train-split historical candidates.',
      '- If fragment Oracle substantially exceeds whole-patch Oracle, the next model should focus on fragment discovery and composition rather than whole-window candidate selection.',
      '- If fragment Oracle improves completion but not forecasting, the next model must learn a PatchSTG-aware composer rather than use hard value replacement.',''
    ]
    (rd/'FRAGMENT_DIAGNOSTIC_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines),flush=True)
    print(f"SUMMARY={rd/'fragment_diagnostic_summary.json'}",flush=True)

if __name__=='__main__': main()
