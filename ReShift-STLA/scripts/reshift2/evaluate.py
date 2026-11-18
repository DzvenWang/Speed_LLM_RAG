#!/usr/bin/env python3
"""Evaluate a ReShift-STLA v0.2 checkpoint on the TEST views of every severity.

--audit additionally runs the in-checkpoint interventions
(wrong_evidence, drop_evolution, drop_replay, drop_cf, base) on the same queries.
"""
from pathlib import Path
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from reshift2.config import cache_path, relocate_saved_config, file_sha256
from reshift2.data import SupportSplit, FullTrainRepository
from reshift2.trainer import build_model, setup_graph, make_batches, predict, needs_pfull, query_bank
from reshift2.teacher import prepare_pfull_bank
from reshift2.memory import LatentMemoryIndex
from reshift2.evaluation import summarize_by_family, save_summary
from reshift2.model import EVAL_MODES

AUDIT = ('wrong_evidence', 'drop_evolution', 'drop_replay', 'drop_cf', 'base')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True)
    p.add_argument('--severities', nargs='+', type=int, default=[20, 40, 60, 90])
    p.add_argument('--backbone-checkpoint'); p.add_argument('--modes', nargs='+', choices=EVAL_MODES)
    p.add_argument('--audit', action='store_true'); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = relocate_saved_config(ck['config']); os.environ['TRAFFIDENT_TARGET'] = cfg['experiment']['target']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bpath = Path(a.backbone_checkpoint or os.environ.get('BACKBONE_CHECKPOINT') or ck['backbone_checkpoint'])
    if ck.get('backbone_sha256') and file_sha256(bpath) != ck['backbone_sha256']:
        raise RuntimeError('STLA checkpoint SHA256 differs from training')
    model, _ = build_model(cfg, bpath, dev); setup_graph(model, cfg, cache_path(cfg, 'formal'))
    model.load_state_dict(ck['state_dict'], strict=True); model.eval()
    repo = FullTrainRepository(cache_path(cfg, 'formal'), 256 if a.smoke else None)
    if not a.smoke and cfg['runtime'].get('preload_repository_gpu', True): repo.preload(dev)
    index = None
    if model.uses_retrieval:
        index = LatentMemoryIndex.load(Path(a.checkpoint).resolve().parent / 'best_memory_index.npz').to(dev)
        if not np.array_equal(repo.starts, index.starts_np): raise RuntimeError('Index / repository mismatch')
    store = Path(cfg['project']['store_root']) / cfg['experiment']['dataset']
    pfull = prepare_pfull_bank(model.teacher, repo, bpath, store, dev, smoke=a.smoke) if needs_pfull(model) else None
    modes = list(a.modes or ['full'])
    if a.audit: modes += [m for m in AUDIT if m not in modes]
    mean = float(ck['mean']); std = float(ck['std']); rows = []; run = Path(a.checkpoint).resolve().parent
    amp = bool(cfg['runtime'].get('amp', False))
    for sev in a.severities:
        ds = SupportSplit(cache_path(cfg, 'evaluation', sev), 'test', 24 if a.smoke else None)
        qrows = np.arange(len(ds))
        bank = query_bank(model, cfg, f'test_s{sev}', ds, qrows, repo, pfull, bpath, dev, a.smoke, with_cf=False)
        dl = make_batches(ds, qrows, 3 if a.smoke else int(cfg['runtime'].get('eval_batch_size', 128)), False, cfg, dev, a.smoke)
        for mode in modes:
            pr = predict(model, dl, index, repo, pfull, bank, dev, mean, std, mode, amp=amp)
            meta = ds.meta.iloc[:len(pr['prediction'])]
            fm = summarize_by_family(pr['prediction'], pr['true'], pr['valid'], pr['direct'], meta)
            bm = summarize_by_family(pr['base'], pr['true'], pr['valid'], pr['direct'], meta)
            out = run / 'evaluation' / f's{sev}' / mode; out.mkdir(parents=True, exist_ok=True)
            if mode == 'full':
                np.savez_compressed(out / 'test_predictions.npz', prediction=pr['prediction'].astype(np.float32),
                                    base_prediction=pr['base'].astype(np.float32))
            save_summary(out / 'summary.json', {'severity': sev, 'mode': mode, 'checkpoint_sha256': file_sha256(a.checkpoint),
                                                'metrics': fm, 'base': bm, 'allocation': pr['allocation']})
            rows.append({'severity': sev, 'mode': mode, **fm['ALL'], **{'base_' + k: v for k, v in bm['ALL'].items()}})
            print('RESHIFT_EVAL ' + json.dumps({k: rows[-1][k] for k in ('severity', 'mode', 'MAE', 'RMSE', 'base_MAE')}), flush=True)
    df = pd.DataFrame(rows); (run / 'evaluation').mkdir(exist_ok=True)
    df.to_csv(run / 'evaluation' / 'severity_summary.csv', index=False)
    full = df[df['mode'] == 'full']
    sfile = run / 'summary.json'; data = json.loads(sfile.read_text()) if sfile.is_file() else {}
    data.update(test_evaluated=True, test_macro_MAE=float(full['MAE'].mean()), test_macro_RMSE=float(full['RMSE'].mean()),
                evaluation_summary=full.to_dict('records'))
    save_summary(sfile, data); print('RESHIFT_EVALUATION_PASS', run / 'evaluation' / 'severity_summary.csv', flush=True)


if __name__ == '__main__': main()
