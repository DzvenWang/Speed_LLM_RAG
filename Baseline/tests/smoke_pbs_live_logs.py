#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pbs_files = sorted((ROOT / 'pbs').rglob('*.pbs'))
assert pbs_files, 'no PBS files found'

errors = []
for p in pbs_files:
    text = p.read_text()
    if 'method_gpu_template.pbs' in str(p):
        # The template also carries the logger so instantiated jobs inherit it.
        pass
    required = [
        'scripts/pbs_live_log.sh',
        'baseline_live_log_start',
        'baseline_install_exit_trap',
    ]
    missing = [x for x in required if x not in text]
    if missing:
        errors.append(f'{p.relative_to(ROOT)} missing {missing}')

helper = (ROOT / 'scripts' / 'pbs_live_log.sh').read_text()
for token in ['%Y%m%d_%H%M%S', '.live.log', 'timestamp_stream.py', 'tee -a']:
    if token not in helper:
        errors.append(f'pbs_live_log.sh missing {token!r}')

if errors:
    raise SystemExit('\n'.join(errors))
print(f'PBS LIVE LOG CONTRACT PASS files={len(pbs_files)}')
