#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys

import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT = ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presure.utils import load_yaml, write_json, stamp, seed_all
from presure.need_gated import load_latest
from presure.qwen_reasoner import resolve_hf_snapshot, load_jsonl
from presure.pipeline import run_dir


def model_key(size):
    s = str(size).lower().replace('qwen_', '').replace('b', '')
    if s in {'0.5', '0_5', '05'}:
        return 'qwen_0_5b'
    if s in {'1.5', '1_5', '15'}:
        return 'qwen_1_5b'
    raise ValueError(f'unsupported Qwen size: {size}')


class ReasoningSFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        messages = [
            {'role': 'system', 'content': r['system_prompt']},
            {'role': 'user', 'content': r['user_prompt']},
        ]
        prompt_ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        target_ids = self.tok(r['target_response'] + (self.tok.eos_token or ''), add_special_tokens=False)['input_ids']
        # Preserve the whole assistant target; trim only the prompt tail if necessary.
        max_prompt = max(16, self.max_length - len(target_ids))
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[:max_prompt]
        ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.ones(len(ids), dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def collate(batch, pad_id):
    mx = max(x['input_ids'].numel() for x in batch)
    ids = torch.full((len(batch), mx), int(pad_id), dtype=torch.long)
    att = torch.zeros((len(batch), mx), dtype=torch.long)
    lab = torch.full((len(batch), mx), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        n = x['input_ids'].numel()
        ids[i, :n] = x['input_ids']
        att[i, :n] = x['attention_mask']
        lab[i, :n] = x['labels']
    return {'input_ids': ids, 'attention_mask': att, 'labels': lab}


def eval_loss(model, loader, device):
    model.eval(); total = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            bs = batch['input_ids'].shape[0]
            total += float(out.loss.detach().cpu()) * bs
            n += bs
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='development', choices=['development'])
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--model-size', default='0.5b', choices=['0.5b','1.5b'])
    ap.add_argument('--dataset-run', default='')
    a = ap.parse_args()
    cfg = load_yaml(ROOT / 'configs/presure.yaml'); seed_all(a.seed)
    qc = cfg['qwen_support_value_probe']; key = model_key(a.model_size)
    mc = qc['models'][key]; common = qc['models']['common']

    if a.dataset_run:
        ds_rd = Path(a.dataset_run)
    else:
        _, ds_rd = load_latest(cfg['project']['exp_root'], 'QwenSupportValueReasoningDataset', cfg['experiment']['subgraph'], a.stage)
    train_rows = load_jsonl(ds_rd / 'reason_train.jsonl')
    dev_rows = load_jsonl(ds_rd / 'reason_dev.jsonl')
    if not train_rows or not dev_rows:
        raise RuntimeError('reasoning dataset is empty')

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as e:
        raise RuntimeError('v0.8.0 LoRA SFT requires transformers and peft in the TFP environment') from e

    model_path = resolve_hf_snapshot(mc['selector_cache'])
    tok = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=bool(common['local_files_only']),
        trust_remote_code=bool(common['trust_remote_code']), use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'

    dtype = torch.float16 if bool(common['fp16']) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=bool(common['local_files_only']),
        trust_remote_code=bool(common['trust_remote_code']), torch_dtype=dtype,
        attn_implementation=str(common['attn_implementation']),
    )
    model.config.use_cache = False
    if bool(common['gradient_checkpointing']):
        model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
    lora = LoraConfig(
        r=int(mc['lora_r']), lora_alpha=int(mc['lora_alpha']),
        lora_dropout=float(mc['lora_dropout']), bias='none', task_type=TaskType.CAUSAL_LM,
        target_modules=list(common['target_modules']),
    )
    model = get_peft_model(model, lora)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_ds = ReasoningSFTDataset(train_rows, tok, common['max_length'])
    dev_ds = ReasoningSFTDataset(dev_rows, tok, common['max_length'])
    cfun = lambda x: collate(x, tok.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=int(mc['train_batch_size']), shuffle=True, num_workers=0, collate_fn=cfun, pin_memory=torch.cuda.is_available())
    dev_loader = DataLoader(dev_ds, batch_size=int(mc['eval_batch_size']), shuffle=False, num_workers=0, collate_fn=cfun, pin_memory=torch.cuda.is_available())

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(mc['learning_rate']), weight_decay=0.01)
    accum = int(mc['grad_accumulation']); epochs = int(mc['epochs'])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and dtype == torch.float16))

    rd = run_dir(cfg, f'QwenSupportValueReasoner_{a.model_size}', a.stage, stamp(), a.seed)
    rd.mkdir(parents=True, exist_ok=True)
    best = math.inf; best_epoch = 0; global_step = 0
    history = []

    print(f'QWEN-SFT model={mc["name"]} local_snapshot={model_path}', flush=True)
    print(f'QWEN-SFT train={len(train_ds)} dev={len(dev_ds)} trainable_params={sum(p.numel() for p in params)}', flush=True)
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True); total = 0.0; seen = 0
        for bi, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and dtype == torch.float16)):
                out = model(**batch)
                loss = out.loss / accum
            scaler.scale(loss).backward()
            if bi % accum == 0 or bi == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                global_step += 1
            bs = batch['input_ids'].shape[0]
            total += float(out.loss.detach().cpu()) * bs; seen += bs
            if bi % max(20, len(train_loader)//10) == 0:
                print(f'QWEN-SFT epoch={ep:02d} batch={bi}/{len(train_loader)} loss={total/max(seen,1):.5f}', flush=True)
        tl = total / max(seen, 1)
        dl = eval_loss(model, dev_loader, device)
        history.append({'epoch': ep, 'train_loss': tl, 'dev_loss': dl})
        print(f'QWEN-SFT epoch={ep:02d} train_loss={tl:.5f} dev_loss={dl:.5f}', flush=True)
        if dl < best:
            best, best_epoch = dl, ep
            outdir = rd / 'adapter_best'
            model.save_pretrained(outdir)
            tok.save_pretrained(outdir)
            write_json(rd / 'best.json', {'epoch': ep, 'dev_loss': dl})

    meta = {
        'variant': 'QwenSupportValueReasonerLoRA', 'model_size': a.model_size,
        'model_name': mc['name'], 'base_cache': mc['selector_cache'], 'resolved_snapshot': str(model_path),
        'dataset_run': str(ds_rd), 'train_records': len(train_rows), 'dev_records': len(dev_rows),
        'best_epoch': best_epoch, 'best_dev_loss': best, 'history': history,
        'training_contract': 'LoRA SFT on Train-only USE/REJECT/ABSTAIN labels; no Validation future in training',
        'test_evaluated': False,
    }
    write_json(rd / 'summary.json', meta)
    latest = Path(cfg['project']['exp_root']) / f'QwenSupportValueReasoner_{a.model_size}' / cfg['experiment']['subgraph'] / a.stage
    latest.mkdir(parents=True, exist_ok=True)
    write_json(latest / 'LATEST.json', {'run_dir': str(rd), 'summary': str(rd/'summary.json'), 'adapter': str(rd/'adapter_best')})
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
    print(f'QwenSupportValueReasonerLoRA PASS run_dir={rd}', flush=True)


if __name__ == '__main__':
    main()
