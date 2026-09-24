#!/usr/bin/env python3
from pathlib import Path
import argparse, json, math, pandas as pd

def safe(v):
    try:
        f=float(v); return f if math.isfinite(f) else None
    except Exception:return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('run_dir'); a=ap.parse_args(); rd=Path(a.run_dir).resolve(); d=rd/'predictive_utility_selector'
    sm=pd.read_csv(d/'model_comparison.csv'); summ=json.loads((d/'summary.json').read_text(encoding='utf-8'))
    primary=int(summ.get('primary_scope',12)); p=sm[sm.recent_steps==primary].sort_values('realized_gain_pct',ascending=False)
    best=p.iloc[0] if len(p) else sm.sort_values('realized_gain_pct',ascending=False).iloc[0]
    lines=['# Predictive-Utility Gate + Reranker v0.7.1','', '## Primary result','',
           f"- Primary scope: **recent{int(best.recent_steps)}**.",
           f"- Best model: **{best.model}**.",
           f"- OOF baseline MAE: **{best.baseline_forecast_mae:.6f}**.",
           f"- OOF realized MAE: **{best.realized_forecast_mae:.6f}**.",
           f"- Realized gain: **{best.realized_gain_pct:+.3f}%**.",
           f"- Forecast-Utility Oracle reference: **{best.oracle_reference_pct:+.3f}%**.",
           f"- Oracle conversion: **{best.oracle_conversion_ratio:.3f}**.",
           f"- Intervention query rate: **{best.intervention_query_rate:.3f}**.",
           f"- Selected patch precision: **{best.selected_individual_precision:.3f}**.",
           f"- Harmful selected-patch rate: **{best.harmful_selected_patch_rate:.3f}**.", '',
           '## Leakage contract','',
           '- Primary evaluation uses expanding temporal CV: fit labels are older than calibration labels, and calibration labels are older than every test block.',
           '- Hidden history and future targets are never model features. They are used only as v0.6/v0.7 training labels or diagnostic outcomes.', '',
           '## Decision checks','',
           f"- Reaches +3% target: **{bool(best.realized_gain_pct>=3.0)}**.",
           f"- Reaches +5% strong target: **{bool(best.realized_gain_pct>=5.0)}**.",
           f"- Positive realized gain: **{bool(best.realized_gain_pct>0.0)}**."]
    (rd/'V07_PREDICTIVE_UTILITY_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    print(f"SUMMARY={rd/'V07_PREDICTIVE_UTILITY_SUMMARY.md'}")
if __name__=='__main__':main()
