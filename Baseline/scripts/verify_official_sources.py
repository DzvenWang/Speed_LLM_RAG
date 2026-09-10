#!/usr/bin/env python3
from pathlib import Path
import subprocess
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
subprocess.check_call(['/scratch/mh20/zw1165/envs/TFP/bin/python',str(ROOT/'scripts/source_lock.py'),'verify','--root',str(ROOT),'--sources',str(ROOT/'configs/official_sources.yaml')])
