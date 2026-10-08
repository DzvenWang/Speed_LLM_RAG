#!/usr/bin/env python3
"""Summarize Base/Candidate/Final validation metrics for every FARE phase.

Works with v1.0.3-v1.2.0 stage evaluation JSONs. Does not load models.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

PHASES = ('retrieval','recovery','injection','partial','full')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    root=args.run_dir
    print('FARE_STAGE_REPORT',str(root.resolve()))
    print(f'{"Phase":<12} {"BaseMAE":>12} {"CandidateMAE":>14} {"FinalMAE":>12} {"Gate":>10} {"Valid":>10} {"CandGain":>12} {"FinalGain":>12}')
    for phase in PHASES:
        path=root/'stages'/phase/'val_summary.json'
        if not path.exists():
            print(f'{phase:<12} MISSING {path}')
            continue
        data=json.loads(path.read_text(encoding='utf-8'))
        diag=data.get('retrieval_diagnostics',{})
        gain=data.get('corrections',{}).get('candidate_MAE_gain',float('nan'))
        final_gain=data.get('corrections',{}).get('final_MAE_gain',float('nan'))
        print(f'{phase:<12} {data["base"]["MAE"]:12.6f} '
              f'{data["candidate"]["MAE"]:14.6f} '
              f'{data["final"]["MAE"]:12.6f} '
              f'{data.get("gate_mean",float("nan")):10.6f} '
              f'{diag.get("valid_group_fraction",float("nan")):10.4f} '
              f'{gain:12.6f} {final_gain:12.6f}')


if __name__=='__main__':main()