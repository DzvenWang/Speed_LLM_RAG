#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import torch

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT = ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presure.utils import load_yaml, write_json, stamp, seed_all
from presure.need_gated import load_latest
from presure.pipeline import run_dir
from presure.qwen_reasoner import (
    ACTION_USE, ACTION_REJECT, ACTION_ABSTAIN,
    QwenDecisionMetrics, resolve_hf_snapshot, load_jsonl, write_jsonl,
    parse_reasoner_output,
)
from scripts.train_qwen_support_value_reasoner import model_key


def baseline_prediction(record, key):
    p = str(record.get(key, 'NONE')).upper()
    if p == 'NONE':
        return ACTION_ABSTAIN, 'NONE'
    return ACTION_USE, p


def oracle_prediction(record):
    a = str(record['label_action']).upper()
    p = 'NONE' if a != ACTION_USE else f"P{int(record['label_proposal_index'])}"
    return a, p


def metric_views(label_cfg):
    return {
        'all_triggered': QwenDecisionMetrics(float(label_cfg['beneficial_threshold']), float(label_cfg['reject_threshold'])),
        'masked_triggered': QwenDecisionMetrics(float(label_cfg['beneficial_threshold']), float(label_cfg['reject_threshold'])),
        'support_triggered': QwenDecisionMetrics(float(label_cfg['beneficial_threshold']), float(label_cfg['reject_threshold'])),
    }


def update_views(acc, record, action, proposal, ok=True):
    acc['all_triggered'].update(record, action, proposal, ok)
    if bool(record.get('view_masked', False)):
        acc['masked_triggered'].update(record, action, proposal, ok)
    if bool(record.get('view_support', False)):
        acc['support_triggered'].update(record, action, proposal, ok)


def summarize_views(acc):
    return {k: v.summary() for k, v in acc.items()}


def load_reasoner(cfg, size, mode, device):
    qc = cfg['qwen_support_value_probe']; key = model_key(size)
    mc = qc['models'][key]; common = qc['models']['common']
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError('transformers is required for Qwen v0.8.0 probe') from e
    base = resolve_hf_snapshot(mc['selector_cache'])
    tok = AutoTokenizer.from_pretrained(
        str(base), local_files_only=bool(common['local_files_only']),
        trust_remote_code=bool(common['trust_remote_code']), use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    dtype = torch.float16 if bool(common['fp16']) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(base), local_files_only=bool(common['local_files_only']),
        trust_remote_code=bool(common['trust_remote_code']), torch_dtype=dtype,
        attn_implementation=str(common['attn_implementation']),
    )
    adapter = None
    if mode == 'lora':
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError('peft is required for --mode lora') from e
        _, trd = load_latest(cfg['project']['exp_root'], f'QwenSupportValueReasoner_{size}', cfg['experiment']['subgraph'], 'development')
        adapter = trd / 'adapter_best'
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.to(device); model.eval(); model.config.use_cache = True
    return tok, model, base, adapter, mc, common


def generate_batch(tok, model, records, common, device):
    texts = []
    for r in records:
        messages = [
            {'role': 'system', 'content': r['system_prompt']},
            {'role': 'user', 'content': r['user_prompt']},
        ]
        texts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    batch = tok(
        texts, return_tensors='pt', padding=True, truncation=True,
        max_length=int(common['max_length']), add_special_tokens=False,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.inference_mode():
        out = model.generate(
            **batch, max_new_tokens=int(common['max_new_tokens']), do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    new = out[:, batch['input_ids'].shape[1]:]
    return tok.batch_decode(new, skip_special_tokens=True)


def decision(qwen_masked, generic_masked, cfg):
    d = cfg['qwen_support_value_probe']['decision']
    g = max(float(generic_masked['mean_selection_regret']), 1e-12)
    q = float(qwen_masked['mean_selection_regret'])
    rr = (g - q) / g
    benef = float(qwen_masked['selected_beneficial_rate'])
    harm = float(qwen_masked['selected_harmful_rate'])
    usep = qwen_masked['use_precision']
    usep = 0.0 if usep is None else float(usep)
    hreg = float(qwen_masked['harmful_regret_rate'])
    go = (
        benef >= float(d['selected_beneficial_go']) and
        harm <= float(d['selected_harmful_go']) and
        rr >= float(d['mean_regret_reduction_vs_generic_go']) and
        usep >= float(d['use_precision_go']) and
        hreg <= float(d['harmful_regret_go'])
    )
    cond = (
        harm <= float(d['selected_harmful_conditional']) and
        rr >= float(d['mean_regret_reduction_vs_generic_conditional'])
    )
    status = 'GO_V081_QWEN_REASONING' if go else ('CONDITIONAL_V081_QWEN_REASONING' if cond else 'NO_GO_V081_QWEN_REASONING')
    return status, {
        'selected_beneficial_rate': benef,
        'selected_harmful_rate': harm,
        'use_precision': usep,
        'harmful_regret_rate': hreg,
        'generic_mean_regret': g,
        'qwen_mean_regret': q,
        'mean_regret_reduction_vs_generic': rr,
        'thresholds': d,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='development', choices=['development'])
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--model-size', default='0.5b', choices=['0.5b','1.5b'])
    ap.add_argument('--mode', default='lora', choices=['zero_shot','lora'])
    ap.add_argument('--dataset-run', default='')
    ap.add_argument('--max-records', type=int, default=0)
    a = ap.parse_args()
    cfg = load_yaml(ROOT / 'configs/presure.yaml'); seed_all(a.seed)
    qc = cfg['qwen_support_value_probe']; label_cfg = qc['labels']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if a.dataset_run:
        ds_rd = Path(a.dataset_run)
    else:
        _, ds_rd = load_latest(cfg['project']['exp_root'], 'QwenSupportValueReasoningDataset', cfg['experiment']['subgraph'], a.stage)
    rows = load_jsonl(ds_rd / 'reason_final_val.jsonl')
    if a.max_records > 0:
        rows = rows[:a.max_records]
    if not rows:
        raise RuntimeError('final validation reasoning dataset is empty')

    tok, model, base, adapter, mc, common = load_reasoner(cfg, a.model_size, a.mode, device)
    bs = int(mc['infer_batch_size'])
    variants = {
        'generic_always_use': metric_views(label_cfg),
        'v076_meta_always_use': metric_views(label_cfg),
        'qwen': metric_views(label_cfg),
        'oracle_action': metric_views(label_cfg),
    }
    preds = []
    for st in range(0, len(rows), bs):
        chunk = rows[st:st+bs]
        texts = generate_batch(tok, model, chunk, common, device)
        for r, txt in zip(chunk, texts):
            valid = [p['id'] for p in r.get('proposals', []) if p.get('valid', True)]
            parsed = parse_reasoner_output(txt, valid)
            update_views(variants['qwen'], r, parsed['action'], parsed['proposal'], parsed['parse_ok'])
            ga, gp = baseline_prediction(r, 'baseline_generic_proposal')
            ma, mp = baseline_prediction(r, 'baseline_v076_meta_proposal')
            oa, op = oracle_prediction(r)
            update_views(variants['generic_always_use'], r, ga, gp, True)
            update_views(variants['v076_meta_always_use'], r, ma, mp, True)
            update_views(variants['oracle_action'], r, oa, op, True)
            preds.append({
                'record_id': r['record_id'], 'view_masked': r.get('view_masked'),
                'label_action': r['label_action'], 'label_proposal_index': r['label_proposal_index'],
                'raw_output': txt, 'parsed_action': parsed['action'], 'parsed_proposal': parsed['proposal'],
                'parse_ok': parsed['parse_ok'],
            })
        print(f'QWEN-PROBE {a.model_size} {a.mode} processed={min(st+bs,len(rows))}/{len(rows)}', flush=True)

    summaries = {k: summarize_views(v) for k, v in variants.items()}
    qmasked = summaries['qwen']['masked_triggered']
    gmasked = summaries['generic_always_use']['masked_triggered']
    status, diag = decision(qmasked, gmasked, cfg)

    rd = run_dir(cfg, f'QwenSupportValueProbe_{a.model_size}_{a.mode}', a.stage, stamp(), a.seed)
    rd.mkdir(parents=True, exist_ok=True)
    write_jsonl(rd / 'predictions.jsonl', preds)
    summary = {
        'variant': 'QwenSupportValueProbe', 'model_size': a.model_size, 'mode': a.mode,
        'model_name': mc['name'], 'base_cache': mc['selector_cache'], 'resolved_snapshot': str(base),
        'adapter': None if adapter is None else str(adapter), 'dataset_run': str(ds_rd),
        'record_count': len(rows), 'metrics': summaries,
        'recommendation': status, 'decision_diagnostics': diag,
        'contract': {
            'Qwen_role': 'support-value reasoner only; never a traffic-speed or residual predictor',
            'actions': ['USE(proposal)', 'REJECT', 'ABSTAIN'],
            'validation_future': 'labels/reporting only; never in prompt',
            'test_evaluated': False,
        },
    }
    write_json(rd / 'summary.json', summary)
    (rd / 'DECISION.txt').write_text(status + '\n' + json.dumps(diag, indent=2, sort_keys=True) + '\n')
    latest = Path(cfg['project']['exp_root']) / f'QwenSupportValueProbe_{a.model_size}_{a.mode}' / cfg['experiment']['subgraph'] / a.stage
    latest.mkdir(parents=True, exist_ok=True)
    write_json(latest / 'LATEST.json', {'run_dir': str(rd), 'summary': str(rd/'summary.json'), 'recommendation': status})
    print('QWEN SUPPORT-VALUE PROBE FINAL', status, flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f'QwenSupportValueProbe PASS run_dir={rd}', flush=True)


if __name__ == '__main__':
    main()
