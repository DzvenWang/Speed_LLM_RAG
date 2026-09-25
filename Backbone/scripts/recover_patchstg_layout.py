#!/usr/bin/env python3
"""Recover the spatial-layout arrays for a historical PatchSTG checkpoint.

PatchSTG's official implementation keeps ori_parts_idx/reo_parts_idx/reo_all_idx as
plain Python/numpy attributes rather than state_dict entries. Historical Backbone
checkpoints therefore retained only a fingerprint. This tool searches historical
SupportBench roots, recomputes the layout, and writes spatial_layout.npz beside the
checkpoint only when the fingerprint exactly matches the historical checkpoint.
"""
from pathlib import Path
import argparse
import datetime
import json
import os
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get('SPEED_LLM_RAG_ROOT', ROOT.parent)).expanduser().resolve()
for p in (ROOT, REPO_ROOT / 'TraffiDent_SupportBench'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backbone.patchstg import compute_patchstg_layout, save_layout_snapshot
from backbone.utils import load_yaml
from supportbench.dataset import SupportBench


def resolve_checkpoint(raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    candidates = [(Path.cwd()/p).resolve(), (REPO_ROOT/p).resolve(), (ROOT/p).resolve()]
    return next((x for x in candidates if x.exists()), candidates[0])


def candidate_benchmark_roots(subgraph: str, explicit: list[str], search_roots: list[str]):
    seen = set()
    for x in explicit:
        p = Path(x).expanduser().resolve()
        if p not in seen:
            seen.add(p); yield p
    defaults = [
        REPO_ROOT / 'data' / 'processed' / 'traffident',
        REPO_ROOT / 'data',
        REPO_ROOT / 'artifacts',
        REPO_ROOT / 'TraffiDent_SupportBench',
    ]
    roots = [Path(x).expanduser().resolve() for x in search_roots] + defaults
    for sr in roots:
        if not sr.exists():
            continue
        direct = sr / subgraph
        if direct.is_dir() and sr not in seen:
            seen.add(sr); yield sr
        try:
            for d in sr.rglob(subgraph):
                if not d.is_dir():
                    continue
                parent = d.parent.resolve()
                if parent not in seen:
                    seen.add(parent); yield parent
        except (PermissionError, OSError):
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', default=str(ROOT/'configs'/'backbone.yaml'))
    ap.add_argument('--benchmark-root', action='append', default=[], help='Known historical SupportBench root; repeatable')
    ap.add_argument('--search-root', action='append', default=[], help='Directory to recursively search for TraffiDent_Net_06; repeatable')
    ap.add_argument('--output', default=None, help='Default: spatial_layout.npz beside best.pt')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    checkpoint = resolve_checkpoint(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict):
        raise RuntimeError('Checkpoint payload is not a dict')
    target = None
    if isinstance(ck.get('adapter_metadata'), dict):
        target = ck['adapter_metadata'].get('spatial_layout_fingerprint')
    if not target:
        raise RuntimeError('Checkpoint has no spatial_layout_fingerprint to recover against')
    subgraph = str(ck.get('run_identity', {}).get('dataset') or cfg['experiment']['subgraph'])
    # run_identity.dataset can be a dataset label rather than folder; keep configured subgraph unless exact folder-like value.
    if not subgraph.startswith('TraffiDent_Net_'):
        subgraph = str(cfg['experiment']['subgraph'])
    num_nodes = int(ck.get('num_nodes', 0))
    if num_nodes <= 0:
        raise RuntimeError('Checkpoint num_nodes missing/invalid')
    model_cfg = ck.get('model_config')
    if not isinstance(model_cfg, dict):
        raise RuntimeError('Checkpoint model_config missing')

    vendor = ROOT/'vendor'/'PatchSTG'
    print(f'TARGET_FINGERPRINT={target}', flush=True)
    print(f'CHECKPOINT={checkpoint}', flush=True)
    print(f'SUBGRAPH={subgraph} N={num_nodes}', flush=True)

    tried = []
    for broot in candidate_benchmark_roots(subgraph, args.benchmark_root, args.search_root):
        try:
            bench = SupportBench(subgraph, root=str(broot))
            if int(bench.N) != num_nodes:
                print(f'SKIP root={broot} N={bench.N} expected={num_nodes}', flush=True)
                continue
            ori, reo, all_idx, fp = compute_patchstg_layout(vendor, bench.nodes, bench.adj, bench.N, model_cfg)
            tried.append({'benchmark_root': str(broot), 'fingerprint': fp})
            print(f'CANDIDATE root={broot} fingerprint={fp}', flush=True)
            if str(fp) == str(target):
                out = Path(args.output).expanduser().resolve() if args.output else checkpoint.parent/'spatial_layout.npz'
                save_layout_snapshot(out, {
                    'ori_parts_idx': ori,
                    'reo_parts_idx': reo,
                    'reo_all_idx': all_idx,
                    'spatial_layout_fingerprint': fp,
                })
                meta = {
                    'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
                    'checkpoint': str(checkpoint),
                    'checkpoint_fingerprint': str(target),
                    'matched_benchmark_root': str(broot),
                    'subgraph': subgraph,
                    'num_nodes': num_nodes,
                    'output': str(out),
                }
                (out.parent/'spatial_layout.recovery.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
                print(f'MATCH FOUND root={broot}', flush=True)
                print(f'WROTE {out}', flush=True)
                return
        except Exception as e:
            print(f'SKIP root={broot} error={type(e).__name__}: {e}', flush=True)

    report = checkpoint.parent/'spatial_layout.recovery.failed.json'
    report.write_text(json.dumps({
        'checkpoint': str(checkpoint),
        'target_fingerprint': str(target),
        'subgraph': subgraph,
        'tried': tried,
    }, indent=2), encoding='utf-8')
    raise RuntimeError(
        'No historical SupportBench root reproduced the checkpoint spatial layout. '
        f'Searched {len(tried)} loadable candidate(s). Report={report}. '
        'If you still have the old benchmark copy elsewhere, rerun with --benchmark-root <that_supportbench_root> '
        'or --search-root <archive_parent>.'
    )


if __name__ == '__main__':
    main()
