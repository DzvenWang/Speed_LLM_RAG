#!/usr/bin/env python3
from pathlib import Path
import argparse, os, subprocess, sys
ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'TraffiDent_SupportBench'))
from backbone.runner import train_model
from backbone.utils import load_yaml, log
from supportbench.dataset import SupportBench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['STAEformer', 'HimNet', 'BigST', 'STDN', 'PatchSTG', 'DST-SGNN'])
    ap.add_argument('--stage', required=True, choices=['screening', 'formal'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--config', default=str(ROOT / 'configs/backbone_selection.yaml'))
    ap.add_argument('--run-stamp', default=None)
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    subprocess.check_call(['/scratch/mh20/zw1165/envs/TFP/bin/python', str(ROOT/'scripts/source_lock.py'), 'verify', '--root', str(ROOT), '--sources', str(ROOT/'configs/backbone_selection.yaml')])
    sg = cfg['experiment']['subgraph']
    if a.stage == 'screening':
        expected = int(cfg['experiment']['screening']['seed'])
        if a.seed != expected:
            raise ValueError(f'Screening seed is fixed to {expected}; got {a.seed}')
    elif a.seed not in [int(x) for x in cfg['experiment']['formal']['seeds']]:
        raise ValueError(f'Formal seed {a.seed} is not in configured formal seeds.')
    cache = Path(cfg['project']['exp_root']) / '_cache' / sg / a.stage
    # Self-validate cache/Protocol identity even when invoked outside PBS.
    subprocess.check_call(['/scratch/mh20/zw1165/envs/TFP/bin/python', str(ROOT/'scripts/validate_cache.py'), '--stage', a.stage])
    if not (cache / 'READY.json').exists():
        raise FileNotFoundError(f'Prepare cache first: {cache}/READY.json')
    b = SupportBench(sg, root=cfg['project']['benchmark_root'])
    st = b.stats['channels'][b.manifest['primary_channel']]
    stamp = a.run_stamp or os.environ.get('RUN_STAMP') or __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    run = Path(cfg['project']['exp_root']) / a.model / sg / a.stage / f'{stamp}_seed{a.seed}'
    log(f'run_dir={run}')
    train_model(
        a.model, cfg, cache, run, ROOT / 'vendor', b.adj, b.nodes,
        float(st['mean']), float(st['std']), a.seed, stage=a.stage
    )


if __name__ == '__main__':
    main()
