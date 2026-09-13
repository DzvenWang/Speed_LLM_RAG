#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

GADI_ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
for p in [ROOT, ROOT.parent / 'TraffiDent_SupportBench']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backbone.adapters import build_adapter
from backbone.data import CachedSplit
from backbone.runner import evaluate
from backbone.utils import load_yaml, write_json, log, sha256_file
from backbone.shared_cache import cache_path
from backbone.identity import (
    load_cache_identity,
    require_cache_version_compatibility,
    require_checkpoint_identity,
)
from supportbench.dataset import SupportBench


def _validate_frozen_manifest(cfg: dict, frozen: dict, checkpoint: Path) -> None:
    if int(frozen.get('schema_version', 0)) < 3:
        raise RuntimeError('FROZEN_BACKBONE.json is old/incompatible; rerun current formal experiments and summarize_formal.py.')
    if str(frozen.get('project_version')) != str(cfg.get('version')):
        raise RuntimeError('Frozen backbone project version does not match current Backbone version.')
    expected_sha = str(frozen.get('reference_checkpoint_sha256', ''))
    if not expected_sha:
        raise RuntimeError('Frozen manifest does not contain reference_checkpoint_sha256.')
    got_sha = sha256_file(checkpoint)
    if got_sha != expected_sha:
        raise RuntimeError(f'Frozen checkpoint SHA256 mismatch: expected={expected_sha} got={got_sha}')


def main() -> None:
    cfg = load_yaml(ROOT / 'configs/backbone_selection.yaml')
    exp = Path(cfg['project']['exp_root'])
    frozen_path = exp / 'FROZEN_BACKBONE.json'
    if not frozen_path.exists():
        raise FileNotFoundError(frozen_path)
    frozen = json.load(open(frozen_path))
    model_name = frozen['selected_model']
    checkpoint = Path(frozen['reference_checkpoint'])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    _validate_frozen_manifest(cfg, frozen, checkpoint)

    cache = cache_path(cfg, 'evaluation')
    if not (cache / 'READY.json').exists():
        raise RuntimeError('Build/freeze Protocol/evaluation, then run pbs/prepare_test_after_freeze.pbs first.')
    eval_identity = load_cache_identity(cache)
    require_cache_version_compatibility(cfg, eval_identity, 'evaluation')

    formal_identity = frozen.get('run_identity')
    if not isinstance(formal_identity, dict):
        raise RuntimeError('Frozen manifest is missing formal run_identity.')
    for key in ['dataset','shared_contract_version','data_contract_revision','time_feature_revision','metrics_revision','prediction_contract_revision','time_feature_fingerprint','time_feature_source']:
        if formal_identity.get(key) != eval_identity.get(key):
            raise RuntimeError(
                f'Frozen formal/evaluation data-contract mismatch in {key}: '
                f'{formal_identity.get(key)} != {eval_identity.get(key)}'
            )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    require_checkpoint_identity(ck, formal_identity, label=str(checkpoint))
    if str(ck.get('model')) != str(model_name):
        raise RuntimeError(f'Checkpoint model={ck.get("model")} conflicts with frozen model={model_name}')
    if int(ck.get('seed', -1)) != int(frozen['reference_seed']):
        raise RuntimeError('Checkpoint seed conflicts with frozen reference_seed.')

    b = SupportBench(cfg['experiment']['subgraph'], root=cfg['project']['benchmark_root'])
    adapter = build_adapter(
        model_name, ROOT / 'vendor', b.N, device,
        float(ck['mean']), float(ck['std']), ck['model_config'], b.adj, b.nodes,
    )
    adapter.model.load_state_dict(ck['state_dict'])
    test = CachedSplit(cache, 'test')
    loader = DataLoader(
        test,
        batch_size=int(ck.get('batch_size', ck['model_config']['batch_size'])),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    pred, df, summary = evaluate(adapter, loader, test.meta, device)
    out = exp / 'frozen_test'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / 'test_predictions.npz',
        view_id=np.asarray(test.meta.view_id.astype(str)),
        pred=pred.astype(np.float32),
    )
    df.to_csv(out / 'test_metrics.csv', index=False)
    obj = {
        'project_version': cfg.get('version'),
        'selected_model': model_name,
        'reference_checkpoint': str(checkpoint),
        'reference_checkpoint_sha256': frozen['reference_checkpoint_sha256'],
        'reference_seed': frozen['reference_seed'],
        'selection_was_frozen_before_test': True,
        'formal_run_identity': formal_identity,
        'evaluation_run_identity': eval_identity,
        'test': summary,
    }
    write_json(out / 'test_summary.json', obj)
    log(
        f'FROZEN TEST COMPLETE model={model_name} MAE={summary["MAE"]:.6f} '
        f'support_MAE={summary["support_dependent_MAE"]:.6f}'
    )


if __name__ == '__main__':
    main()
