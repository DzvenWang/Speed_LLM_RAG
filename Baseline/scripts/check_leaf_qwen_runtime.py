#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline'); sys.path.insert(0,str(ROOT))
from baseline.runners.common import config
from baseline.llm.leaf_qwen import resolve_snapshot


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu-load',action='store_true'); a=ap.parse_args()
    cfg=config(); q=cfg['runners']['LEAF']['qwen_selector']; snap=resolve_snapshot(q['model_cache_root'],q.get('snapshot'))
    required=['config.json','tokenizer_config.json']
    for name in required:
        if not (snap/name).exists(): raise SystemExit(f'LEAF_QWEN_RUNTIME_FAIL missing {snap/name}')
    import torch, transformers
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(snap,local_files_only=True)
    out={'python':sys.executable,'torch':torch.__version__,'transformers':transformers.__version__,'snapshot':str(snap),'tokenizer':tok.__class__.__name__,'gpu_load':bool(a.gpu_load)}
    if a.gpu_load:
        if not torch.cuda.is_available(): raise SystemExit('LEAF_QWEN_RUNTIME_FAIL --gpu-load requires GPU compute node')
        from transformers import AutoModelForCausalLM
        model=AutoModelForCausalLM.from_pretrained(snap,local_files_only=True,torch_dtype=torch.float16).cuda().eval()
        text=tok.apply_chat_template([{'role':'system','content':'Return only the requested choice number.'},{'role':'user','content':'Return only 2.'}],tokenize=False,add_generation_prompt=True)
        x=tok(text,return_tensors='pt').to('cuda')
        with torch.inference_mode(): y=model.generate(**x,max_new_tokens=4,do_sample=False,pad_token_id=tok.pad_token_id or tok.eos_token_id)
        ans=tok.decode(y[0,x['input_ids'].shape[1]:],skip_special_tokens=True).strip(); out['smoke_answer']=ans
        if '2' not in ans: raise SystemExit(f'LEAF_QWEN_RUNTIME_FAIL unexpected generation {ans!r}')
    print(json.dumps(out,indent=2)); print('LEAF QWEN1.5B RUNTIME PASS')
if __name__=='__main__': main()
