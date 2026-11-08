"""Full-observation teacher residual store, indexed by train-only memory row.

Separates ordinary STLA error from the change induced by applying a query mask.
No test outcome ever enters this cache; full historical futures are train only.
"""
from pathlib import Path
import os,json
import numpy as np
import torch
from .config import file_sha256
from .evidence import counterfactual_residuals

class NormalTeacherBank:
    def __init__(self,path,device,on_gpu=False):
        self.path=Path(path);self.array=np.load(path,mmap_mode='r',allow_pickle=False);self.gpu=None
        if on_gpu and torch.device(device).type=='cuda':
            try:self.gpu=torch.as_tensor(np.array(self.array),device=device)
            except torch.cuda.OutOfMemoryError:self.gpu=None;torch.cuda.empty_cache()
    def fetch(self,indices,device):
        if self.gpu is not None:return self.gpu[indices.long().to(self.gpu.device)].float()
        idx=indices.detach().cpu().numpy().astype(np.int64)
        return torch.from_numpy(np.array(self.array[idx],copy=True)).to(device).float()

def prepare_normal_teacher_bank(cfg,memory,teacher,backbone_checkpoint,device,smoke=False):
    if teacher is None:raise RuntimeError('Normal residual decomposition requires a frozen teacher')
    n=len(memory);P=int(cfg['experiment']['horizon']);N=int(memory.arrays['x'].shape[-1])
    folder=Path(cfg['project']['retrieval_root'])/cfg['experiment']['dataset']/('v5' if str(cfg.get('version','')).startswith('5.') else ('v41' if str(cfg.get('version','')).startswith('4.1.') else ('v4' if str(cfg.get('version','')).startswith('4.') else 'v3')))/('formal_smoke' if smoke else 'formal')/('normal_teacher' if str(cfg.get('version','')).startswith('5.') else 'normal_teacher_v2')
    folder.mkdir(parents=True,exist_ok=True)
    mask_aware=str(cfg.get('version','')).startswith(('4.','5.'))
    dtype='float32' if mask_aware else 'float16'
    path=folder/f'normal_residual_{dtype}.npy';meta=path.with_suffix('.json')
    source=dict(schema=26,train_rows=n,target=cfg['experiment']['target'],nodes=N,horizon=P,
                backbone_sha=file_sha256(backbone_checkpoint),
                train_history_path=str(Path(memory.arrays['x'].filename).resolve()),
                train_history_mtime=Path(memory.arrays['x'].filename).stat().st_mtime_ns,
                train_future_path=str(Path(memory.arrays['y'].filename).resolve()),
                train_future_mtime=Path(memory.arrays['y'].filename).stat().st_mtime_ns,
                teacher_fingerprint=cfg.get('runtime',{}).get('teacher_fingerprint','zero_initialization'),
                source_observed_sha=file_sha256(Path(memory.arrays['x'].filename).parent/'train_history_observed.npy')
                    if str(cfg.get('version','')).startswith(('4.','5.')) else None,
                source_mask_sha=file_sha256(Path(memory.arrays['x'].filename).parent/'train_artificial_missing.npy')
                    if str(cfg.get('version','')).startswith(('4.','5.')) else None)
    if path.is_file() and meta.is_file():
        try:
            arr=np.load(path,mmap_mode='r',allow_pickle=False)
            if json.loads(meta.read_text())==source and arr.shape==(n,P,N):
                print('NORMAL_TEACHER_BANK_HIT',path,flush=True)
                return NormalTeacherBank(path,device,not smoke and cfg['runtime'].get('normal_teacher_on_gpu',True))
        except (ValueError,IOError):pass
    tmp=path.with_name(path.name+'.partial.npy')
    arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=dtype,shape=(n,P,N))
    bsz=int(cfg['retrieval'].get('replay_batch_size',64))
    teacher.eval()
    with torch.inference_mode():
        for s in range(0,n,bsz):
            ids=torch.arange(s,min(s+bsz,n),device=device,dtype=torch.long)[:,None]
            cand=memory.fetch(ids,device)
            query_mask=cand['source_valid'][:,0].float() if str(cfg.get('version','')).startswith(('4.','5.')) else torch.ones_like(cand['x'][:,0])
            res=counterfactual_residuals(teacher,cand,query_mask,chunk_size=bsz,replay='full',strict_source_mask=str(cfg.get('version','')).startswith(('4.','5.')))[:,0]
            arr[s:s+len(res)]=res.cpu().numpy().astype(dtype)
    arr.flush();del arr;os.replace(tmp,path)
    mt=meta.with_name(meta.name+'.partial');mt.write_text(json.dumps(source,indent=2));os.replace(mt,meta)
    print('NORMAL_TEACHER_BANK_PASS',path,flush=True)
    return NormalTeacherBank(path,device,not smoke and cfg['runtime'].get('normal_teacher_on_gpu',True))
