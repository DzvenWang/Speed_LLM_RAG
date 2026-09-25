#!/usr/bin/env python3
"""Cross-platform frozen Backbone severity evaluation runner."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--severities', nargs='+', default=['20', '40', '60', '90'])
    ap.add_argument('--prepare-cache', action='store_true', help='Prepare/validate evaluation cache before inference.')
    args = ap.parse_args()
    for sev in args.severities:
        cfg = ROOT / 'configs' / f'backbone_eval_s{sev}.yaml'
        if args.prepare_cache:
            cmd = [sys.executable, str(ROOT / 'scripts' / 'prepare_data.py'), '--stage', 'evaluation', '--config', str(cfg)]
            print('RUN', ' '.join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=ROOT)
        cmd = [sys.executable, str(ROOT / 'scripts' / 'evaluate_frozen_test.py'), '--config', str(cfg), '--eval-tag', f's{sev}']
        print('RUN', ' '.join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=ROOT)
    print('SEVERITY EVALUATION PASS')


if __name__ == '__main__':
    main()
