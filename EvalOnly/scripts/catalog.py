#!/usr/bin/env python3
"""Discover only checkpoint-backed eval-only jobs. Python stdlib; usable at Gadi login."""
from __future__ import annotations
import os
import re
from pathlib import Path

BB_MODELS = ('STAEformer', 'HimNet', 'BigST', 'STDN', 'PatchSTG', 'STLAformer')
BL_METHODS = ('GinAR', 'ImputeFormer', 'LRTC-TMCP', 'RAFT', 'RAST', 'SARAF', 'ReMFT',
              'RAFT-PatchSTG', 'RAFT-STLAformer', 'RAST-PatchSTG', 'RAST-STLAformer')
NATIVE = frozenset(('GinAR', 'RAFT', 'RAST', 'SARAF', 'ReMFT'))
REFERENCES = {
    'ImputeFormer': ('STLAformer', 'PatchSTG'),
    'LRTC-TMCP': ('STLAformer', 'PatchSTG'),
    'RAFT-PatchSTG': ('PatchSTG',), 'RAST-PatchSTG': ('PatchSTG',),
    'RAFT-STLAformer': ('STLAformer',), 'RAST-STLAformer': ('STLAformer',),
}
STATUS_RE = re.compile(r'"status"\s*:\s*"COMPLETE"')


def env_words(key, default):
    return os.environ.get(key, ' '.join(default)).split()


def latest_backbone(parent: Path, seed: int):
    if not parent.is_dir(): return None
    matches = sorted(parent.glob(f'*_seed{seed}/best.pt'), reverse=True)
    return next((str(p.resolve()) for p in matches if p.is_file()), None)


def latest_formal(parent: Path, seed: int):
    if not parent.is_dir(): return None
    for run in sorted(parent.glob(f'*_seed{seed}'), reverse=True):
        if not run.is_dir() or not (run / 'best.pt').is_file(): continue
        summary = run / 'summary.json'
        if summary.is_file() and STATUS_RE.search(summary.read_text(errors='replace')):
            return str(run.resolve())
    return None


def discover(root: Path, kind: str, seed: int = 10):
    root = root.resolve()
    targets = env_words('TARGETS', ('speed', 'flow'))
    for t in targets:
        if t not in ('speed', 'flow'): raise ValueError(f'Invalid TARGETS member: {t}')
    jobs, skips = [], []
    if kind == 'backbone':
        models = env_words('BACKBONE_MODELS', BB_MODELS)
        modes = env_words('BACKBONE_MODES', ('incomplete', 'complete'))
        for m in models:
            if m not in BB_MODELS: raise ValueError(f'Unsupported Backbone: {m}')
        for mode in modes:
            if mode not in ('incomplete', 'complete'): raise ValueError(f'Invalid mode: {mode}')
        for t in targets:
            for m in models:
                for mode in modes:
                    stage = 'formal_complete' if mode == 'complete' else 'formal'
                    base = root / 'Backbone_exps' / t / m / 'TraffiDent_Net_06' / stage
                    ckpt = latest_backbone(base, seed)
                    label = f'BB {t}/{m}/{mode}'
                    if ckpt:
                        jobs.append(dict(kind=kind, target=t, model=m, mode=mode, seed=seed, checkpoint=ckpt, label=label))
                    else: skips.append(label)
    elif kind == 'baseline':
        methods = env_words('BASELINE_METHODS', BL_METHODS)
        modes = env_words('REFERENCE_MODES', ('incomplete', 'complete'))
        for m in methods:
            if m not in BL_METHODS: raise ValueError(f'Unsupported Baseline: {m}')
        for mode in modes:
            if mode not in ('incomplete', 'complete'): raise ValueError(f'Invalid mode: {mode}')
        for t in targets:
            for m in methods:
                refs = ('native',) if m in NATIVE else REFERENCES[m]
                for ref in refs:
                    for mode in (('incomplete',) if ref == 'native' else modes):
                        if ref == 'native':
                            parent = root / 'Baseline_exps' / t / m / 'TraffiDent_Net_06' / 'formal'
                        else:
                            parent = root / 'Baseline_exps' / t / m / ref
                            if mode == 'complete': parent /= 'complete'
                            parent = parent / 'TraffiDent_Net_06' / 'formal'
                        run_dir = latest_formal(parent, seed)
                        label = f'BL {t}/{m}/{ref}/{mode}'
                        if run_dir:
                            jobs.append(dict(kind=kind, target=t, method=m, backbone=ref if ref != 'native' else '',
                                             mode=mode, seed=seed, formal_run_dir=run_dir, label=label))
                        else: skips.append(label)
    else: raise ValueError(kind)
    return jobs, skips
