#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for p in [ROOT, REPO_ROOT / 'TraffiDent_SupportBench']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared_protocol.utils import load_yaml
from supportbench.dataset import SupportBench

cfg = load_yaml(ROOT / 'configs' / 'protocol_s40.yaml')
b = SupportBench(cfg['experiment']['dataset'], root=cfg['project']['benchmark_root'])
fams = list(cfg['experiment']['families'])
train = b.views_for('train').copy()
train = train[train.family.astype(str).isin(fams)]
print('TRAIN GRID COVERAGE')
print('dataset=', cfg['experiment']['dataset'])
for sev in [0.20, 0.40, 0.60, 0.90]:
    d = train[abs(train.severity.astype(float) - sev) < 1e-9]
    counts = d.family.astype(str).value_counts().to_dict()
    print(f'severity={sev:.2f} rows={len(d):,}')
    for f in fams:
        print(f'  {f}: {int(counts.get(f,0)):,}')
