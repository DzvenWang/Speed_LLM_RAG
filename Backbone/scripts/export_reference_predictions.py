#!/usr/bin/env python3
"""Export frozen-backbone predictions for support/RAG stages.

By default only the validation-selected frozen checkpoint is accepted. The
checkpoint checksum, formal run identity, and current time-feature fingerprint
must all match before inference starts.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

GADI_ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
COMMON = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists(): COMMON = ROOT.parent / 'ExperimentCommon'
for p in [ROOT, COMMON, ROOT.parent / 'TraffiDent_SupportBench']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backbone.adapters import build_adapter
from backbone.identity import require_checkpoint_identity
from experiment_common.time_features import assert_resolved_time_features, resolve_time_features
from experiment_common.predictions import save_prediction_npz
from experiment_common.utils import load_yaml as load_shared_yaml
from backbone.utils import load_yaml, ensure_dir, log, sha256_file, write_json
from supportbench.dataset import SupportBench


def batches(seq, bs):
    for i in range(0, len(seq), bs):
        yield seq[i:i + bs]


def make_batch_views(b, vids, tod, dow, mean, std, fill_value):
    xs, yr, th, dh, tf, df = [], [], [], [], [], []
    H, P = int(b.H), int(b.P)
    for vid in vids:
        s = b.sample(str(vid))
        mv = b.model_view(s, speed_only=True, normalized=True, fill_value=fill_value)
        st = int(s.window_start)
        xs.append(mv['history'][..., 0])
        yr.append(s.future_speed)
        th.append(tod[st:st + H])
        dh.append(dow[st:st + H])
        tf.append(tod[st + H:st + H + P])
        df.append(dow[st + H:st + H + P])
    return {
        'x': torch.tensor(np.asarray(xs)),
        'y_raw': torch.tensor(np.asarray(yr)),
        'y_norm': torch.tensor((np.asarray(yr) - mean) / std),
        'tod_hist': torch.tensor(np.asarray(th)),
        'dow_hist': torch.tensor(np.asarray(dh)),
        'tod_future': torch.tensor(np.asarray(tf)),
        'dow_future': torch.tensor(np.asarray(df)),
    }


def make_batch_memory(b, eids, tod, dow, mean, std):
    xs, yr, th, dh, tf, df = [], [], [], [], [], []
    H, P = int(b.H), int(b.P)
    for eid in eids:
        st = int(b.memory_starts[int(eid)])
        ep = b.memory_episode_from_start(st)
        x = (ep['history_speed'] - mean) / std
        xs.append(x)
        yr.append(ep['future_speed'])
        th.append(tod[st:st + H])
        dh.append(dow[st:st + H])
        tf.append(tod[st + H:st + H + P])
        df.append(dow[st + H:st + H + P])
    return {
        'x': torch.tensor(np.asarray(xs)),
        'y_raw': torch.tensor(np.asarray(yr)),
        'y_norm': torch.tensor((np.asarray(yr) - mean) / std),
        'tod_hist': torch.tensor(np.asarray(th)),
        'dow_hist': torch.tensor(np.asarray(dh)),
        'tod_future': torch.tensor(np.asarray(tf)),
        'dow_future': torch.tensor(np.asarray(df)),
    }


def infer(adapter, items, maker, bs):
    out = []
    adapter.model.eval()
    with torch.no_grad():
        for j, chunk in enumerate(batches(list(items), bs), 1):
            batch = maker(chunk)
            out.append(adapter.raw(adapter.eval_forward(batch)).cpu().numpy().astype(np.float32))
            if j % 20 == 0:
                log(f'inference batches={j}')
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs/backbone_selection.yaml'))
    ap.add_argument('--model', default=None)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--allow-unfrozen', action='store_true')
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    exp = Path(cfg['project']['exp_root'])
    frozen_path = exp / 'FROZEN_BACKBONE.json'
    frozen = None

    if not a.allow_unfrozen:
        if not frozen_path.exists():
            raise RuntimeError('Backbone is not frozen: missing FROZEN_BACKBONE.json')
        frozen = json.load(open(frozen_path))
        if int(frozen.get('schema_version', 0)) < 3:
            raise RuntimeError('Frozen manifest is old/incompatible; rerun formal reference experiments and summarize_formal.py.')
        model = frozen['selected_model']
        compat = cfg.get('compatibility', {}).get('formal_reference_project_versions', {})
        allowed_versions = [str(x) for x in compat.get(model, [cfg.get('version')])]
        frozen_version = str(frozen.get('project_version'))
        if frozen_version not in allowed_versions:
            raise RuntimeError(
                f'Frozen manifest project_version={frozen_version} is incompatible with current Backbone '
                f'v{cfg.get("version")} for model={model}; allowed={allowed_versions}.'
            )
        checkpoint = frozen['reference_checkpoint']
        if a.model is not None and a.model != model:
            raise RuntimeError(f'CLI model={a.model} conflicts with frozen model={model}')
        if a.checkpoint is not None and str(Path(a.checkpoint)) != str(Path(checkpoint)):
            raise RuntimeError('CLI checkpoint conflicts with frozen reference checkpoint.')
        got_sha = sha256_file(checkpoint)
        if got_sha != str(frozen.get('reference_checkpoint_sha256', '')):
            raise RuntimeError('Frozen reference checkpoint SHA256 mismatch.')
    else:
        if not a.model or not a.checkpoint:
            raise ValueError('--allow-unfrozen requires --model and --checkpoint')
        model, checkpoint = a.model, a.checkpoint

    b = SupportBench(cfg['experiment']['subgraph'], root=cfg['project']['benchmark_root'])
    cp = Path(cfg['shared_contract']['config'])
    if not cp.exists(): cp = COMMON / 'configs/experiment_contract.yaml'
    shared_cfg = load_shared_yaml(cp)
    tf_cfg = shared_cfg['time_features']
    tf_result = resolve_time_features(
        b,
        allow_index_fallback=bool(tf_cfg.get('allow_index_modulo_fallback', False)),
        index_origin_dow=int(tf_cfg.get('index_origin_dow', 0)),
    )
    tf_audit = assert_resolved_time_features(tf_result)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    if str(ck.get('model')) != str(model):
        raise RuntimeError(f'Checkpoint model={ck.get("model")} conflicts with requested model={model}')
    if frozen is not None:
        formal_identity = frozen['run_identity']
        require_checkpoint_identity(ck, formal_identity, label=str(checkpoint))
        if tf_audit['fingerprint'] != formal_identity.get('time_feature_fingerprint'):
            raise RuntimeError(
                'Current SupportBench time-feature fingerprint differs from the frozen formal checkpoint. '
                'Do not export predictions across data contracts.'
            )

    mean = float(ck['mean'])
    std = float(ck['std'])
    adapter = build_adapter(model, ROOT / 'vendor', b.N, device, mean, std, ck['model_config'], b.adj, b.nodes)
    adapter.model.load_state_dict(ck['state_dict'])
    tod, dow = tf_result.tod, tf_result.dow
    bs = int(ck.get('batch_size', ck['model_config'].get('batch_size', 16)))
    fill_value = float(cfg['experiment']['normalized_fill_value'])
    art = Path(cfg['project']['support_artifact_root']) / b.sg_dir.name
    out = ensure_dir(exp / 'reference_predictions' / model)

    counts = {}
    for split in ['val', 'test']:
        z = np.load(art / f'{split}_challenge_pool.npz', allow_pickle=False)
        vids = np.asarray(z['view_id']).astype(str)
        pred = infer(adapter, vids, lambda c: make_batch_views(b, c, tod, dow, mean, std, fill_value), bs)
        save_prediction_npz(out / f'{split}_query_predictions.npz', vids, pred, manifest_path=out / f'{split}_query_predictions.manifest.json', metadata={'producer':'Backbone','model':model,'split':split,'checkpoint_sha256':sha256_file(checkpoint)})
        counts[split] = int(len(vids))
        log(f'wrote {split}: {len(vids)}')

    eids = b.memory_episode_ids_for_budget('development')
    pred = infer(adapter, eids, lambda c: make_batch_memory(b, c, tod, dow, mean, std), bs)
    np.savez_compressed(out / 'train_memory_predictions.npz', episode_id=np.asarray(eids, np.int64), pred=pred)
    log(f'wrote memory: {len(eids)}')

    meta = {
        'project_version': cfg.get('version'),
        'frozen': not a.allow_unfrozen,
        'frozen_model': model,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint),
        'checkpoint_run_identity': ck.get('run_identity'),
        'time_feature_revision': cfg['shared_contract']['time_feature_revision'],
        'time_feature_source': tf_result.source,
        'time_feature_fingerprint': tf_audit['fingerprint'],
        'batch_size': bs,
        'val_queries': counts['val'],
        'test_queries': counts['test'],
        'memory_episodes': int(len(eids)),
    }
    write_json(out / 'export_metadata.json', meta)
    print('OUTPUT_DIR=' + str(out), flush=True)


if __name__ == '__main__':
    main()
