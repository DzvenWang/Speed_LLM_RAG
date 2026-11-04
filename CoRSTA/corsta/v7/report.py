"""Readable four-severity report for the final two-module model."""

def render_markdown(report):
    modes=('backbone','normal','c1','c12','selected')
    sev=report['severities'];means=report['mean_four_severities']
    lines=['# TCRR-STLA v7 — Counterfactual Support and Spatial Transfer',
           '', 'All TEST predictions use a model fixed before TEST evaluation.',
           '', '|Stage|s20 MAE|s40 MAE|s60 MAE|s90 MAE|Macro MAE|Macro RMSE|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for key in modes:
        row=[key]
        row.extend(f'{sev["s"+str(i)][key]["mae"]:.6f}' if 's'+str(i) in sev else 'n/a'
                   for i in (20,40,60,90))
        row.extend((f'{means[key]["mae"]:.6f}',f'{means[key]["rmse"]:.6f}'))
        lines.append('|'+ '|'.join(row)+'|')
    lines.extend(['','## Incremental correction improvements',''])
    for a,b in [('backbone','normal'),('normal','c1'),('c1','c12')]:
        baseline=means[a]['mae'];gain=baseline-means[b]['mae']
        lines.append(f'- {a} -> {b}: {gain:+.6f} MAE, {gain/max(baseline,1e-8)*100:+.4f}%.')
    lines.extend(['','Four severity levels are macro-averaged, not pooled.',
                  'Historical futures are TRAIN-only; current query futures never enter retrieval.'])
    return '\n'.join(lines)+'\n'
