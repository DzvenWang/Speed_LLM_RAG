#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'TraffiDent_SupportBench'))

from backbone.adapters import build_adapter
from backbone.data import CachedSplit
from backbone.identity import load_cache_identity, require_identity
from backbone.metrics import METRICS_REVISION
from backbone.runner import evaluate
from backbone.selection import MODELS
from backbone.utils import load_yaml, write_json
from supportbench.dataset import SupportBench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strict-all', action='store_true', help='Fail if any configured model lacks a completed screening checkpoint.')
    ap.add_argument('--models', nargs='*', default=None)
    args = ap.parse_args()
    cfg = load_yaml(ROOT / 'configs/backbone_selection.yaml')
    exp = Path(cfg['project']['exp_root'])
    sg = cfg['experiment']['subgraph']
    cache = exp / '_cache' / sg / 'screening'
    identity = load_cache_identity(cache)
    bench = SupportBench(sg, root=cfg['project']['benchmark_root'])
    if not torch.cuda.is_available():
        raise SystemExit(
            'ERROR: refresh_screening_metrics requires a GPU. '
            'Submit pbs/refresh_screening_metrics.pbs instead of running on a login node.'
        )
    device = torch.device('cuda:0')
    val = CachedSplit(cache, 'val')

    print(
        f'REFRESH DEVICE={device} val_queries={len(val)} models={",".join(args.models or MODELS)}',
        flush=True,
    )

    for model_name in (args.models or MODELS):
        print(f'REFRESH START model={model_name}', flush=True)
        root = exp / model_name / sg / 'screening'
        candidates = []
        for f in sorted(root.glob('*/summary.json')) if root.exists() else []:
            try:
                s = json.loads(f.read_text())
                require_identity(s, identity, str(f))
                if (f.parent / 'best.pt').exists():
                    candidates.append((f, s))
            except Exception:
                continue
        if not candidates:
            msg=f'{model_name}: no current-identity completed screening run with best.pt'
            if args.strict_all or args.models:
                raise SystemExit(msg)
            print('REFRESH SKIP '+msg, flush=True)
            continue
        f, s = max(candidates, key=lambda x: x[0].stat().st_mtime)
        run_dir = f.parent
        ck = torch.load(run_dir / 'best.pt', map_location=device, weights_only=False)
        adapter = build_adapter(
            model_name, ROOT / 'vendor', bench.N, device,
            float(ck['mean']), float(ck['std']), ck['model_config'], bench.adj, bench.nodes
        )
        adapter.model.load_state_dict(ck['state_dict'])
        loader = DataLoader(
            val,
            batch_size=int(ck.get('batch_size', ck['model_config']['batch_size'])),
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        pred, df, summary = evaluate(adapter, loader, val.meta, device)
        np.savez_compressed(
            run_dir / 'val_predictions.npz',
            view_id=np.asarray(val.meta.view_id.astype(str)),
            pred=pred.astype(np.float32),
        )
        df.to_csv(run_dir / 'val_metrics.csv', index=False)
        s['validation'] = summary
        s['metrics_revision'] = METRICS_REVISION
        write_json(run_dir / 'summary.json', s)
        print(
            f'REFRESH DONE model={model_name} MAE={summary["MAE"]:.6f} '
            f'RMSE={summary["RMSE"]:.6f} '
            f'masked_RMSE={summary["directly_masked_RMSE"]:.6f} '
            f'support_RMSE={summary["support_dependent_RMSE"]:.6f}'
        , flush=True)

    print('SCREENING METRICS REFRESH PASS', flush=True)


if __name__ == '__main__':
    main()
