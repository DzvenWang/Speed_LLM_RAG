#!/usr/bin/env python3
"""Run the configured PatchSTG formal seeds sequentially on any platform."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backbone.utils import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'backbone_selection.yaml'))
    ap.add_argument('--model', default='PatchSTG')
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    seeds = [int(x) for x in cfg['experiment']['formal']['seeds']]
    for seed in seeds:
        cmd = [sys.executable, str(ROOT / 'scripts' / 'run_experiment.py'), '--model', a.model, '--stage', 'formal', '--seed', str(seed), '--config', str(Path(a.config).resolve())]
        print('RUN', ' '.join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=ROOT)
    cmd = [sys.executable, str(ROOT / 'scripts' / 'summarize_formal.py'), '--config', str(Path(a.config).resolve())]
    print('RUN', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


if __name__ == '__main__':
    main()
