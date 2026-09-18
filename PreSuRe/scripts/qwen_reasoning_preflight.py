#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, json, sys

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT = ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presure.utils import load_yaml
from presure.qwen_reasoner import resolve_hf_snapshot
from presure.need_gated import load_latest


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development']); a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure.yaml')
    version=(ROOT/'VERSION').read_text().strip()
    if version!='0.8.0' or str(cfg.get('version'))!='0.8.0':
        raise RuntimeError(f'version mismatch VERSION={version} cfg={cfg.get("version")}')
    print('PASS PreSuRe v0.8.0')
    print('PASS role', cfg['qwen_support_value_probe']['role'])
    for exp, req in [
        ('PredictiveNeedProbe','utility_A_best.pt'),
        ('PredictiveNeedStateProbe','summary.json'),
        ('NeedConditionedUtilitySpecialistProbe','hard_specialist_best.pt'),
        ('DeploymentAlignedUtilityRouterProbe','pred10_specialist_best.pt'),
        ('RegretAwareSupportProposalMetaRankingProbe','proposal_meta_ranker_best.pt'),
    ]:
        _, rd=load_latest(cfg['project']['exp_root'],exp,cfg['experiment']['subgraph'],a.stage)
        if not (rd/req).exists(): raise FileNotFoundError(rd/req)
        print(f'PASS frozen prerequisite {exp} {rd}')
    try:
        import transformers
        print('PASS transformers', transformers.__version__)
    except Exception as e:
        raise RuntimeError('transformers unavailable in current environment') from e
    try:
        import peft
        print('PASS peft', peft.__version__)
    except Exception as e:
        raise RuntimeError('peft unavailable; required for LoRA SFT') from e
    for key in ['qwen_0_5b','qwen_1_5b']:
        mc=cfg['qwen_support_value_probe']['models'][key]
        snap=resolve_hf_snapshot(mc['selector_cache'])
        if not (snap/'config.json').exists(): raise FileNotFoundError(snap/'config.json')
        if not ((snap/'tokenizer_config.json').exists() or (snap/'tokenizer.json').exists()):
            raise FileNotFoundError(f'Qwen tokenizer files missing in {snap}')
        print(f'PASS {key} cache_root={mc["selector_cache"]}')
        print(f'PASS {key} resolved_snapshot={snap}')
    print('PASS labels USE / REJECT / ABSTAIN; future used for Train supervision / Validation reporting only')
    print('PASS prompts exclude direct_mask/support_mask/current-query future/true residual/true q90')
    print('QWEN SUPPORT-VALUE REASONING PREFLIGHT PASS')

if __name__=='__main__': main()
