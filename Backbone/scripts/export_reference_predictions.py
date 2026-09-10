#!/usr/bin/env python3
"""Export formal 25k Val/Test challenge + Train-memory predictions.

By default this script only uses the frozen backbone written by
scripts/summarize_formal.py. This prevents post-hoc backbone switching after
Test or support-headroom results are visible.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import torch
ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'TraffiDent_SupportBench'))
from backbone.adapters import build_adapter
from backbone.data import resolve_time_features
from backbone.utils import load_yaml, ensure_dir, log
from supportbench.dataset import SupportBench


def batches(seq, bs):
    for i in range(0, len(seq), bs):
        yield seq[i:i + bs]


def make_batch_views(b, vids, tod, dow, mean, std, fill_value):
    xs, yr, th, dh, tf, df = [], [], [], [], [], []
    for vid in vids:
        s = b.sample(str(vid))
        mv = b.model_view(s, speed_only=True, normalized=True, fill_value=fill_value)
        st = s.window_start
        xs.append(mv['history'][..., 0]); yr.append(s.future_speed)
        th.append(tod[st:st + 12]); dh.append(dow[st:st + 12])
        tf.append(tod[st + 12:st + 24]); df.append(dow[st + 12:st + 24])
    return {
        'x': torch.tensor(np.asarray(xs)),
        'y_raw': torch.tensor(np.asarray(yr)),
        'y_norm': torch.tensor((np.asarray(yr) - mean) / std),
        'tod_hist': torch.tensor(np.asarray(th)), 'dow_hist': torch.tensor(np.asarray(dh)),
        'tod_future': torch.tensor(np.asarray(tf)), 'dow_future': torch.tensor(np.asarray(df)),
    }


def make_batch_memory(b, eids, tod, dow, mean, std):
    xs, yr, th, dh, tf, df = [], [], [], [], [], []
    for eid in eids:
        st = int(b.memory_starts[int(eid)])
        ep = b.memory_episode_from_start(st)
        x = (ep['history_speed'] - mean) / std
        xs.append(x); yr.append(ep['future_speed'])
        th.append(tod[st:st + 12]); dh.append(dow[st:st + 12])
        tf.append(tod[st + 12:st + 24]); df.append(dow[st + 12:st + 24])
    return {
        'x': torch.tensor(np.asarray(xs)),
        'y_raw': torch.tensor(np.asarray(yr)),
        'y_norm': torch.tensor((np.asarray(yr) - mean) / std),
        'tod_hist': torch.tensor(np.asarray(th)), 'dow_hist': torch.tensor(np.asarray(dh)),
        'tod_future': torch.tensor(np.asarray(tf)), 'dow_future': torch.tensor(np.asarray(df)),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs/backbone_selection.yaml'))
    ap.add_argument('--model', default=None)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--allow-unfrozen', action='store_true')
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    exp = Path(cfg['project']['exp_root'])
    frozen_path = exp / 'FROZEN_BACKBONE.json'
    if not a.allow_unfrozen:
        if not frozen_path.exists():
            raise RuntimeError('Backbone is not frozen: missing FROZEN_BACKBONE.json')
        frozen = json.load(open(frozen_path))
        model = frozen['selected_model']
        checkpoint = frozen['reference_checkpoint']
        if a.model is not None and a.model != model:
            raise RuntimeError(f'CLI model={a.model} conflicts with frozen model={model}')
        if a.checkpoint is not None and str(Path(a.checkpoint)) != str(Path(checkpoint)):
            raise RuntimeError('CLI checkpoint conflicts with frozen reference checkpoint.')
    else:
        if not a.model or not a.checkpoint:
            raise ValueError('--allow-unfrozen requires --model and --checkpoint')
        model, checkpoint = a.model, a.checkpoint

    b = SupportBench(cfg['experiment']['subgraph'], root=cfg['project']['benchmark_root'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    mean = float(ck['mean']); std = float(ck['std'])
    adapter = build_adapter(model, ROOT / 'vendor', b.N, device, mean, std, ck['model_config'], b.adj, b.nodes)
    adapter.model.load_state_dict(ck['state_dict'])
    tod, dow, _ = resolve_time_features(b)
    bs = int(ck.get('batch_size', ck['model_config'].get('batch_size', 16)))
    fill_value = float(cfg['experiment']['normalized_fill_value'])
    art = Path(cfg['project']['support_artifact_root']) / b.sg_dir.name
    out = ensure_dir(exp / 'reference_predictions' / model)
    for split in ['val', 'test']:
        z = np.load(art / f'{split}_challenge_pool.npz', allow_pickle=False)
        vids = np.asarray(z['view_id']).astype(str)
        pred = infer(adapter, vids, lambda c: make_batch_views(b, c, tod, dow, mean, std, fill_value), bs)
        np.savez_compressed(out / f'{split}_query_predictions.npz', view_id=vids, pred=pred)
        log(f'wrote {split}: {len(vids)}')
    eids = b.memory_episode_ids_for_budget('development')
    pred = infer(adapter, eids, lambda c: make_batch_memory(b, c, tod, dow, mean, std), bs)
    np.savez_compressed(out / 'train_memory_predictions.npz', episode_id=np.asarray(eids, np.int64), pred=pred)
    log(f'wrote memory: {len(eids)}')
    meta = {
        'project_version': cfg.get('version'),
        'frozen_model': model,
        'checkpoint': str(checkpoint),
        'batch_size': bs,
        'val_queries': int(len(np.load(art / 'val_challenge_pool.npz', allow_pickle=False)['view_id'])),
        'test_queries': int(len(np.load(art / 'test_challenge_pool.npz', allow_pickle=False)['view_id'])),
        'memory_episodes': int(len(eids)),
    }
    import json as _json
    with open(out / 'export_metadata.json', 'w') as f:
        _json.dump(meta, f, indent=2)
    print('OUTPUT_DIR=' + str(out), flush=True)


if __name__ == '__main__':
    main()
