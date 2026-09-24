#!/usr/bin/env python3
from pathlib import Path
import argparse, json, math
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_dir');a=ap.parse_args();rd=Path(a.run_dir).resolve();d=rd/'intervention_utility_selector'
    sm=pd.read_csv(d/'model_comparison.csv');summ=json.loads((d/'summary.json').read_text(encoding='utf-8'));primary=int(summ.get('primary_scope',12))
    p=sm[sm.recent_steps==primary].sort_values('realized_gain_pct',ascending=False);best=p.iloc[0] if len(p) else sm.sort_values('realized_gain_pct',ascending=False).iloc[0]
    lines=['# Intervention-Aware Predictive Utility v0.8.0','', '## Primary result','',
           f'- Primary scope: **recent{int(best.recent_steps)}**.',f'- Best model: **{best.model}**.',f'- Feature set: **{best.feature_set}**.',
           f'- OOF baseline MAE: **{best.baseline_forecast_mae:.6f}**.',f'- OOF realized MAE: **{best.realized_forecast_mae:.6f}**.',f'- Realized gain: **{best.realized_gain_pct:+.3f}%**.',
           f'- Forecast-Utility Oracle@{int(summ.get("max_k",20))}: **{best.oracle_reference_pct:+.3f}%**.',f'- Oracle conversion: **{best.oracle_conversion_ratio:.3f}**.',
           f'- Intervention query rate: **{best.intervention_query_rate:.3f}**.',f'- Selected patch precision: **{best.selected_individual_precision:.3f}**.',f'- Harmful selected-patch rate: **{best.harmful_selected_patch_rate:.3f}**.', '',
           '## What changed from v0.7','',
           '- Each candidate is materialized as an actual completion proposal before utility scoring.',
           '- The selector receives transport magnitude, patch/neighbor coherence, frozen-PatchSTG prediction shift and prehead latent shift.',
           '- Risk-aware models explicitly penalize harmful interventions rather than optimizing similarity or reconstruction quality.', '',
           '## Leakage contract','',
           '- Primary evaluation uses expanding temporal CV: fit labels precede calibration labels, which precede every test block.',
           '- Proposal/latent/prediction-shift features use only current observed input/mask, retrieved train-memory episodes and frozen PatchSTG outputs.',
           '- Hidden history and future targets are never v0.8 model features; they are historical labels or evaluation outcomes only.', '',
           '## Decision checks','',
           f'- Reaches +2% target: **{bool(best.realized_gain_pct>=2.0)}**.',f'- Reaches +3% strong target: **{bool(best.realized_gain_pct>=3.0)}**.',f'- Positive realized gain: **{bool(best.realized_gain_pct>0.0)}**.']
    path=rd/'V08_INTERVENTION_UTILITY_SUMMARY.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines));print(f'SUMMARY={path}')
if __name__=='__main__':main()
