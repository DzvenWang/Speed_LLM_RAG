#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT = ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presure.utils import load_yaml, write_json, stamp, seed_all
from presure.contract import cache_dir
from presure.data import CacheSplit, cache_stats
from presure.need_gated import instantiate_utility_a, load_latest
from presure.probes import SupportUtilityProbeNet
from presure.proposal_meta import SupportProposalMetaRanker
from presure.deployment_routing import predicted_need_mask
from presure.pipeline import load_artifacts, run_dir
from presure.qwen_reasoner import (
    ACTION_USE, ACTION_REJECT, ACTION_ABSTAIN,
    decision_label, proposal_prompt_features, format_reasoning_user,
    reasoning_system_prompt, target_response, write_jsonl,
)
from scripts.run_need_conditioned_utility_specialist_probe import (
    batches, t, load_state_q90, predict_need_q90, load_top4, load_state_top4,
    view_np,
)
from scripts.run_deployment_aligned_utility_router_probe import load_v074_hard
from scripts.run_regret_aware_support_proposal_meta_probe import (
    EXPERT_ORDER, load_v075, build_bundle,
)


class Reservoir:
    def __init__(self, cap, rng):
        self.cap = int(cap)
        self.rng = rng
        self.rows = []
        self.seen = 0

    def add(self, row):
        self.seen += 1
        if len(self.rows) < self.cap:
            self.rows.append(row)
            return
        j = int(self.rng.integers(0, self.seen))
        if j < self.cap:
            self.rows[j] = row


def load_v076_meta(v076_rd: Path, dev):
    p = v076_rd / 'proposal_meta_ranker_best.pt'
    ck = torch.load(p, map_location=dev, weights_only=False)
    if ck.get('model_type') != 'regret_aware_support_proposal_meta_ranker_v076':
        raise RuntimeError(f'v0.7.6 meta-ranker contract mismatch: {p}')
    m = SupportProposalMetaRanker(**ck['model_args']).to(dev)
    m.load_state_dict(ck['state_dict']); m.eval()
    return m, ck


def _proposal_record(feature, target_utility, valid, pid, digits):
    d = proposal_prompt_features(feature, EXPERT_ORDER)
    out = {
        'id': f'P{pid}',
        'valid': bool(valid),
        'target_utility': round(float(target_utility), digits),
        'proposed_by': d['proposed_by'],
        'agreement_ratio': round(float(d['agreement_ratio']), digits),
        'm3_confidence': round(float(d['m3_confidence']), digits),
        'm3_margin': round(float(d['m3_margin']), digits),
        'local_residual': round(float(d['local_residual']), digits),
        'residual_mean': round(float(d['residual_mean']), digits),
        'residual_std': round(float(d['residual_std']), digits),
        'residual_maxabs': round(float(d['residual_maxabs']), digits),
        'context_cosine': None if d['context_cosine'] is None else round(float(d['context_cosine']), digits),
        'context_absdiff_mean': None if d['context_absdiff_mean'] is None else round(float(d['context_absdiff_mean']), digits),
        'expert_probabilities': {k: round(float(v), digits) for k, v in d['expert_probabilities'].items()},
    }
    return out


def make_record(bundle, meta_score, ids, local_pos, cell, ds, split_name, digits, label_cfg):
    b, n, h = [int(x) for x in cell]
    pm = bundle['mask'][b, n, h].detach().cpu().numpy().astype(bool)
    ut = bundle['target'][b, n, h].detach().cpu().numpy().astype(np.float64)
    feat = bundle['features'][b, n, h].detach().cpu().numpy()
    unique = int(pm.sum())
    label = decision_label(
        ut, pm,
        beneficial_threshold=float(label_cfg['beneficial_threshold']),
        reject_threshold=float(label_cfg['reject_threshold']),
    )
    props = [_proposal_record(feat[s], ut[s], pm[s], s, digits) for s in range(len(pm)) if pm[s]]
    # Proposal slots are stable P0..P3; invalid slots are omitted from prompt records.
    need_prob = float(feat[np.flatnonzero(pm)[0], 4 * len(EXPERT_ORDER)]) if np.any(pm) else 0.0
    meta = np.asarray(meta_score[b, n, h].detach().cpu(), np.float64)
    meta = np.where(pm, meta, -np.inf)
    meta_choice = int(np.argmax(meta)) if np.any(pm) else -1
    qid = int(ids[int(local_pos[b])])
    view_masked = bool(np.asarray(ds.direct[qid], bool)[n])
    view_support = bool(np.asarray(ds.support[qid], bool)[n])
    rec = {
        'record_id': f'{split_name}.q{qid}.n{n}.h{h}',
        'source_split': split_name,
        'query_id': qid,
        'node_index': n,
        'horizon_index': h,
        'need_probability': round(need_prob, digits),
        'unique_proposals': unique,
        'expert_disagreement': round((unique - 1) / max(len(EXPERT_ORDER) - 1, 1), digits),
        'proposals': props,
        'label_action': label['action'],
        'label_proposal_index': int(label['proposal_index']),
        'oracle_best_utility': round(float(label['best_utility']), digits),
        'oracle_decision_value': round(float(label['oracle_value']), digits),
        'baseline_generic_proposal': 'P0' if pm[0] else 'NONE',
        'baseline_v076_meta_proposal': f'P{meta_choice}' if meta_choice >= 0 else 'NONE',
        'view_masked': view_masked,
        'view_support': view_support,
    }
    rec['system_prompt'] = reasoning_system_prompt()
    rec['user_prompt'] = format_reasoning_user(rec)
    rec['target_response'] = target_response(rec)
    return rec


def collect_records(
    models, meta, cfg, art, ds, ids, top4, need_prob, thresholds,
    local_positions, split_name, max_records, dev, rng, balance_actions=False,
):
    qc = cfg['qwen_support_value_probe']
    label_cfg = qc['labels']
    dcfg = qc['dataset']
    digits = int(dcfg['context_round_digits'])
    _, residual_std = cache_stats(cache_dir(cfg, 'development'))
    bs = int(cfg['proposal_meta_ranking_probe']['evaluation']['eval_batch_size'])
    min_unique = int(qc['trigger']['min_unique_proposals'])
    pred20_thr = np.asarray(thresholds['pred20'], np.float32)

    if balance_actions:
        each = max(1, int(max_records) // 3)
        reservoirs = {a: Reservoir(each, rng) for a in [ACTION_USE, ACTION_REJECT, ACTION_ABSTAIN]}
    else:
        reservoir = Reservoir(max_records, rng)

    for pos in batches(np.asarray(local_positions, np.int64), bs, shuffle=False):
        _, _, _, valid, _, _, bundle = build_bundle(
            models, cfg, art, ds, ids, top4, pos, need_prob, dev, residual_std
        )
        with torch.no_grad():
            meta_score = meta(bundle['features'])
        dep = predicted_need_mask(
            np.asarray(need_prob[pos], np.float32), pred20_thr,
            np.asarray(valid.detach().cpu(), bool),
        )  # [B,P,N]
        dep_t = np.transpose(dep, (0, 2, 1))
        unique = np.asarray(bundle['unique_count'].detach().cpu(), np.float32)
        eligible = dep_t & (unique >= min_unique)
        coords = np.argwhere(eligible)
        if len(coords) == 0:
            continue
        # Bound Python formatting work per numerical batch; random before labels.
        cap_batch = min(len(coords), 768)
        if len(coords) > cap_batch:
            coords = coords[rng.choice(len(coords), size=cap_batch, replace=False)]
        for cell in coords:
            rec = make_record(bundle, meta_score, ids, pos, cell, ds, split_name, digits, label_cfg)
            if balance_actions:
                reservoirs[rec['label_action']].add(rec)
            else:
                reservoir.add(rec)

    if balance_actions:
        rows = []
        for a in [ACTION_USE, ACTION_REJECT, ACTION_ABSTAIN]:
            rows.extend(reservoirs[a].rows)
        rng.shuffle(rows)
        rows = rows[:max_records]
        seen = {a: reservoirs[a].seen for a in reservoirs}
    else:
        rows = reservoir.rows
        rng.shuffle(rows)
        seen = {'eligible_seen': reservoir.seen}
    return rows, seen


def label_counts(rows):
    out = defaultdict(int)
    for r in rows:
        out[r['label_action']] += 1
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='development', choices=['development'])
    ap.add_argument('--seed', type=int, default=10)
    a = ap.parse_args()
    cfg = load_yaml(ROOT / 'configs/presure.yaml'); seed_all(a.seed)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    art = load_artifacts(cfg, a.stage); cache = cache_dir(cfg, a.stage)
    tr = CacheSplit(cache, 'train'); va = CacheSplit(cache, 'val')

    _, exact_rd = load_latest(cfg['project']['exp_root'], 'PredictiveNeedProbe', cfg['experiment']['subgraph'], a.stage)
    _, state_rd = load_latest(cfg['project']['exp_root'], 'PredictiveNeedStateProbe', cfg['experiment']['subgraph'], a.stage)
    _, v074_rd = load_latest(cfg['project']['exp_root'], 'NeedConditionedUtilitySpecialistProbe', cfg['experiment']['subgraph'], a.stage)
    _, v075_rd = load_latest(cfg['project']['exp_root'], 'DeploymentAlignedUtilityRouterProbe', cfg['experiment']['subgraph'], a.stage)
    _, v076_rd = load_latest(cfg['project']['exp_root'], 'RegretAwareSupportProposalMetaRankingProbe', cfg['experiment']['subgraph'], a.stage)

    generic, _ = instantiate_utility_a(exact_rd / 'utility_A_best.pt', dev)
    hard, _ = load_v074_hard(v074_rd, dev)
    v075_models, _, _ = load_v075(v075_rd, dev)
    models = {'generic': generic, 'hard': hard, **v075_models}
    q90_model, _ = load_state_q90(state_rd, dev)
    meta, meta_ck = load_v076_meta(v076_rd, dev)

    v075_summary = json.loads((v075_rd / 'summary.json').read_text())
    thresholds = {k: np.asarray(v, np.float32) for k, v in v075_summary['train_only_predicted_need_thresholds'].items()}

    ids_dev = np.load(exact_rd / 'utility_dev_ids.npy').astype(np.int64)
    top_dev = load_top4(exact_rd, 'utility_dev')
    nft = np.load(art / 'train_node_features.npy', mmap_mode='r')
    need_dev = predict_need_q90(
        q90_model, nft, ids_dev,
        int(cfg['proposal_meta_ranking_probe']['evaluation']['need_eval_batch_size']), dev,
    )

    dc = cfg['qwen_support_value_probe']['dataset']
    rng = np.random.default_rng(a.seed + int(dc['sampling_seed_offset']))
    qorder = rng.permutation(len(ids_dev))
    ntr = max(1, int(round(len(qorder) * float(dc['train_fraction']))))
    qtrain, qdev = qorder[:ntr], qorder[ntr:]
    if len(qdev) == 0:
        qdev = qorder[-max(1, len(qorder)//5):]
        qtrain = qorder[:max(1, len(qorder)-len(qdev))]

    train_rows, train_seen = collect_records(
        models, meta, cfg, art, tr, ids_dev, top_dev, need_dev, thresholds,
        qtrain, 'reason_train', int(dc['train_max_records']), dev, rng, balance_actions=bool(dc['balance_train_actions'])
    )
    dev_rows, dev_seen = collect_records(
        models, meta, cfg, art, tr, ids_dev, top_dev, need_dev, thresholds,
        qdev, 'reason_dev', int(dc['dev_max_records']), dev, rng, balance_actions=False
    )

    state_split = json.loads((state_rd / 'split_manifest.json').read_text())
    val_ids = np.asarray(state_split['final_val'], np.int64)
    need_val = np.asarray(np.load(state_rd / 'prob_need_q90_final_val.npy', mmap_mode='r'), np.float32)[..., 1]
    top_val = load_state_top4(state_rd)
    val_local = np.arange(len(val_ids), dtype=np.int64)
    val_rows, val_seen = collect_records(
        models, meta, cfg, art, va, val_ids, top_val, need_val, thresholds,
        val_local, 'final_val', int(dc['final_val_max_records']), dev, rng, balance_actions=False
    )

    rd = run_dir(cfg, 'QwenSupportValueReasoningDataset', a.stage, stamp(), a.seed)
    rd.mkdir(parents=True, exist_ok=True)
    write_jsonl(rd / 'reason_train.jsonl', train_rows)
    write_jsonl(rd / 'reason_dev.jsonl', dev_rows)
    write_jsonl(rd / 'reason_final_val.jsonl', val_rows)
    split = {
        'utility_dev_query_count': int(len(ids_dev)),
        'reason_train_query_local_ids': [int(x) for x in qtrain],
        'reason_dev_query_local_ids': [int(x) for x in qdev],
        'final_val_query_count': int(len(val_ids)),
    }
    write_json(rd / 'split_manifest.json', split)
    summary = {
        'variant': 'QwenSupportValueReasoningDataset',
        'stage': a.stage,
        'seed': a.seed,
        'implementation_revision': cfg.get('implementation_revision'),
        'contract': {
            'trigger': 'Train-frozen predicted Need top20 AND >=2 unique expert proposals',
            'future_usage': 'Train future builds labels only; Validation future is reporting only',
            'inference_features_exclude': ['current-query future', 'true residual', 'true q90', 'direct_mask', 'support_mask'],
            'direct_support_masks': 'stored only as evaluation views; never included in prompts',
            'experts_frozen': EXPERT_ORDER,
            'v076_meta_frozen': True,
        },
        'sources': {
            'v071': str(exact_rd), 'v072': str(state_rd), 'v074': str(v074_rd),
            'v075': str(v075_rd), 'v076': str(v076_rd),
            'v076_meta_epoch': int(meta_ck['epoch']),
        },
        'records': {
            'train': len(train_rows), 'dev': len(dev_rows), 'final_val': len(val_rows),
            'train_labels': label_counts(train_rows),
            'dev_labels': label_counts(dev_rows),
            'final_val_labels': label_counts(val_rows),
            'train_seen': train_seen, 'dev_seen': dev_seen, 'final_val_seen': val_seen,
        },
    }
    write_json(rd / 'summary.json', summary)
    latest = Path(cfg['project']['exp_root']) / 'QwenSupportValueReasoningDataset' / cfg['experiment']['subgraph'] / a.stage
    latest.mkdir(parents=True, exist_ok=True)
    write_json(latest / 'LATEST.json', {'run_dir': str(rd), 'summary': str(rd / 'summary.json'), 'implementation_revision': cfg.get('implementation_revision')})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f'QwenSupportValueReasoningDataset PASS run_dir={rd}', flush=True)


if __name__ == '__main__':
    main()
