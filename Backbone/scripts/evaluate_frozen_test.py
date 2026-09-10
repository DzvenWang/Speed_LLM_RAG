#!/usr/bin/env python3
from pathlib import Path
import json, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'TraffiDent_SupportBench'))
from backbone.adapters import build_adapter
from backbone.data import CachedSplit
from backbone.runner import evaluate
from backbone.utils import load_yaml, write_json, log
from backbone.identity import load_cache_identity
from supportbench.dataset import SupportBench


def main():
    cfg = load_yaml(ROOT / 'configs/backbone_selection.yaml')
    exp = Path(cfg['project']['exp_root'])
    frozen_path = exp / 'FROZEN_BACKBONE.json'
    if not frozen_path.exists():
        raise FileNotFoundError(frozen_path)
    frozen = json.load(open(frozen_path))
    model_name = frozen['selected_model']
    checkpoint = Path(frozen['reference_checkpoint'])
    cache = exp / '_cache' / cfg['experiment']['subgraph'] / 'evaluation'
    if not (cache / 'READY.json').exists():
        raise RuntimeError('Build/freeze Protocol/evaluation, then run pbs/prepare_test_after_freeze.pbs first.')
    eval_identity = load_cache_identity(cache)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    b = SupportBench(cfg['experiment']['subgraph'], root=cfg['project']['benchmark_root'])
    adapter = build_adapter(
        model_name, ROOT / 'vendor', b.N, device,
        float(ck['mean']), float(ck['std']), ck['model_config'], b.adj, b.nodes
    )
    adapter.model.load_state_dict(ck['state_dict'])
    test = CachedSplit(cache, 'test')
    loader = DataLoader(
        test, batch_size=int(ck.get('batch_size', ck['model_config']['batch_size'])),
        shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    pred, df, summary = evaluate(adapter, loader, test.meta, device)
    out = exp / 'frozen_test'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'test_predictions.npz', view_id=np.asarray(test.meta.view_id.astype(str)), pred=pred.astype(np.float32))
    df.to_csv(out / 'test_metrics.csv', index=False)
    obj = {
        'project_version': cfg.get('version'),
        'selected_model': model_name,
        'reference_checkpoint': str(checkpoint),
        'reference_seed': frozen['reference_seed'],
        'selection_was_frozen_before_test': True,
        'formal_run_identity': frozen.get('run_identity'),
        'evaluation_run_identity': eval_identity,
        'test': summary,
    }
    write_json(out / 'test_summary.json', obj)
    log(f'FROZEN TEST COMPLETE model={model_name} MAE={summary["MAE"]:.6f} support_MAE={summary["support_dependent_MAE"]:.6f}')


if __name__ == '__main__':
    main()
