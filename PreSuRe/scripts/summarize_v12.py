#!/usr/bin/env python3
from pathlib import Path
import json
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_exps/_v12_artifacts/TraffiDent_Net_06/development/LATEST_RUN.txt')
if not ROOT.exists(): raise SystemExit('No LATEST_RUN.txt')
r=Path(ROOT.read_text().strip()); j=json.loads((r/'summary.json').read_text()); print('RUN',r); print('BEST_EPOCH',j['best_epoch']); print('DEV_MASKED_GAIN',f"{j['dev_masked_improvement_pct']:+.3f}%"); print('FINAL_MASKED_GAIN',f"{j['final_masked_improvement_pct']:+.3f}%"); print('RETRIEVER_DEV',j['retriever_dev']); print('DEV_AUX',j['dev_aux']); print('FINAL_AUX',j['final_aux']); print('DECISION',j['decision'])
