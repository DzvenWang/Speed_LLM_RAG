from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from fare.context import FAREContext
from fare.training import train_context,build_retrieval_index,build_retrieval_utility_labels,train_retrieval,train_backbone_aware,calibrate_structural
from fare.evaluation import evaluate_experts
from fare.utils import log


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--target',required=True,choices=['speed','flow'])
    ap.add_argument('--stage',choices=['all','train','eval','context','index','retrieval_utility','retrieval','backbone_aware','structural_calibration'],default='all')
    ap.add_argument('--smoke',action='store_true'); ap.add_argument('--force',action='store_true')
    ap.add_argument('--severity',type=int,choices=[20,40,60,90],default=None); ap.add_argument('--max-rows',type=int,default=None)
    a=ap.parse_args(); mode='smoke' if a.smoke else 'formal'; ctx=FAREContext(a.target,run_mode=mode)
    subprocess.run([sys.executable,str(ROOT/'scripts'/'preflight.py'),'--target',a.target,'--run-mode',mode],check=True)
    def ensure(path,fn,label):
        if Path(path).exists() and not a.force: log(f'REUSE_{label} {path}')
        else: fn()
    if a.stage in ('all','train','context'):
        ensure(ctx.artifact_root/'context'/'best.pt',lambda:train_context(ctx,a.smoke),'CONTEXT')
    if a.stage in ('all','train','index'):
        if not (ctx.artifact_root/'context'/'best.pt').exists(): raise FileNotFoundError('Context checkpoint required before retrieval index')
        ensure(ctx.artifact_root/'retrieval_index'/'index_meta.json',lambda:build_retrieval_index(ctx,a.smoke),'RETRIEVAL_INDEX')
    if a.stage in ('all','train','retrieval_utility'):
        if not (ctx.artifact_root/'retrieval_index'/'index_meta.json').exists(): raise FileNotFoundError('Retrieval index required before utility labels')
        ensure(ctx.artifact_root/'retrieval_utility'/'utility_meta.json',lambda:build_retrieval_utility_labels(ctx,a.smoke),'RETRIEVAL_UTILITY')
    if a.stage in ('all','train','retrieval'):
        if not (ctx.artifact_root/'retrieval_utility'/'utility_meta.json').exists(): raise FileNotFoundError('Retrieval utility labels required before Retrieval training')
        ensure(ctx.artifact_root/'retrieval'/'best.pt',lambda:train_retrieval(ctx,a.smoke),'RETRIEVAL')
    if a.stage in ('all','train','backbone_aware'):
        ensure(ctx.artifact_root/'backbone_aware'/'best.pt',lambda:train_backbone_aware(ctx,a.smoke),'BACKBONE_AWARE')
    if a.stage in ('all','train','structural_calibration'):
        ensure(ctx.artifact_root/'structural'/'calibration.json',lambda:calibrate_structural(ctx,a.smoke),'STRUCTURAL_CALIBRATION')
    if a.stage in ('all','eval'):
        required=[ctx.artifact_root/'context'/'best.pt',ctx.artifact_root/'retrieval_index'/'index_meta.json',ctx.artifact_root/'retrieval_utility'/'utility_meta.json',ctx.artifact_root/'retrieval'/'best.pt',ctx.artifact_root/'backbone_aware'/'best.pt',ctx.artifact_root/'structural'/'calibration.json']
        miss=[str(p) for p in required if not p.exists()]
        if miss: raise FileNotFoundError('Missing trained artifacts: '+str(miss))
        sevs=[a.severity] if a.severity else ([40] if a.smoke else list(ctx.cfg['experiment']['evaluation_severities']))
        rows=a.max_rows if a.max_rows is not None else (ctx.cfg['runtime']['smoke_rows'] if a.smoke else ctx.cfg['runtime']['diagnostic_rows'])
        for sev in sevs: evaluate_experts(ctx,int(sev),int(rows) if rows else None,a.smoke)
    log('FARE_V023_PIPELINE_PASS')
if __name__=='__main__': main()
