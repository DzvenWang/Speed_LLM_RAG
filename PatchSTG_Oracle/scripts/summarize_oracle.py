#!/usr/bin/env python3
from pathlib import Path
import argparse,json
ap=argparse.ArgumentParser(); ap.add_argument('summary'); a=ap.parse_args(); j=json.loads(Path(a.summary).read_text())
print('Decision:',j['decision']); print('Scope:',j['scope']); print('Final complete Masked MAE gain:',f"{j['final_complete_masked_mae_gain_pct']:+.3f}%")
for v,p in j['metrics'].items():
    m=p['final']; g=0 if v=='incomplete' else j['gains_pct'][v]['final']['masked_mae']; print(f'{v:20s} MAE={m["masked_mae"]:.6f} RMSE={m["masked_rmse"]:.6f} gain={g:+.3f}%')
print('Missing groups:')
for k,v in j['missing_group_gains'].items(): print(k,v)
print('Latent:',j['latent_gap_and_correlation']['final'])
