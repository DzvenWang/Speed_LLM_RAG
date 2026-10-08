from __future__ import annotations
import copy,json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Subset
from .memory import TrainMemory
from .model import FARE
from .data import loader,to_device
from .trainer import FARETrainer
from .config import json_write

def time_folds(ds,k):
    times=ds.meta.iloc[:len(ds)].window_start.to_numpy(np.int64)
    unique=np.unique(times); blocks=np.array_split(unique,k)
    return [(np.where(np.isin(times,block))[0], (int(block.min()),int(block.max()))) for block in blocks if len(block)]

def build_oof(model,memory,ds,cfg,meta,device,out):
    """True time-blocked cross-fitting; no fold labels in its candidate model or retrieval store."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    k=int(cfg['training']['oof_folds']); gap=int(cfg['training']['oof_time_gap'])
    feats=[];util=[];risk=[];opt=[];fold_meta=[]
    starts=ds.meta.iloc[:len(ds)].window_start.to_numpy(np.int64)
    for j,(held,(low,high)) in enumerate(time_folds(ds,k)):
        training=np.where((starts < low-gap)|(starts > high+gap))[0]
        if not len(training):raise RuntimeError(f'OOF fold {j}: no eligible training rows after purging')
        mask=(memory.starts.cpu().numpy() < low-gap)|(memory.starts.cpu().numpy() > high+gap)
        if not mask.any():raise RuntimeError(f'OOF fold {j}: no eligible memory rows')
        ix=torch.as_tensor(np.where(mask)[0],device=memory.x.device,dtype=torch.long)
        mem=TrainMemory(memory.x[ix].detach().cpu(),memory.valid[ix].detach().cpu(),memory.starts[ix].detach().cpu(),memory.latent[ix].detach().cpu() if memory.latent is not None else None,device=device,tod=memory.tod[ix].detach().cpu() if memory.tod is not None else None,dow=memory.dow[ix].detach().cpu() if memory.dow is not None else None)
        # Critical: initialize fold models from the PRETRAINED teacher, NOT the all-data candidate.
        torch.manual_seed(int(cfg['experiment']['seed'])+j)
        fold_model=FARE(copy.deepcopy(model.teacher),model.stla.graph_matrix.detach().clone(),cfg).to(device)
        fold_ds=Subset(ds,training.tolist())
        fold_trainer=FARETrainer(fold_model,mem,fold_ds,fold_ds,cfg,meta,device,out/f'fold{j:02d}',None)
        epochs=int(cfg['training']['oof_candidate_epochs'])
        for phase in ['retrieval','recovery','injection','partial','full']:
            fold_model.set_phase(phase);optimizer=fold_trainer._optimizer(phase)
            for ep in range(epochs):
                fold_model.refresh(mem)
                record=fold_trainer._epoch(phase,optimizer)
                print('OOF_TRAIN '+json.dumps({'fold':j,'phase':phase,'epoch':ep+1,**record}),flush=True)
        fold_model.eval();fold_model.refresh(mem)
        sampled=0
        with torch.no_grad():
            for batch in loader(Subset(ds,held.tolist()),cfg['training']['batch_size']):
                bt=to_device(batch,device)
                pred=fold_model(bt,mem)
                # Uniform node subset independent of target, preventing supervised selection bias.
                n=bt['x_obs'].shape[-1];pick=torch.linspace(0,n-1,min(n,int(cfg['training']['oof_nodes_per_window'])),device=device).long()
                f=pred['control_raw_features'][:,:,pick]
                y=bt['y_raw'][:,:,pick]
                b=pred['base_prediction'][:,:,pick]*meta['std']+meta['mean']
                c=pred['candidate_prediction'][:,:,pick]*meta['std']+meta['mean']
                v=bt['future_valid'][:,:,pick].bool()
                gain=((b-y).abs()-(c-y).abs())/max(meta['std'],1e-6)
                harm=((c-y).square()-(b-y).square()).clamp_min(0)/max(meta['std']**2,1e-6)
                alpha=((y-b)/(c-b).where((c-b).abs()>1e-5,torch.ones_like(c-b))).clamp(0,1)
                alpha=torch.where((c-b).abs()>1e-5,alpha,torch.zeros_like(alpha))
                feats.append(f[v].cpu().numpy());util.append(gain[v].cpu().numpy());risk.append(harm[v].cpu().numpy());opt.append(alpha[v].cpu().numpy())
                sampled+=int(v.sum())
        fold_meta.append({'fold':j,'time_range':[low,high],'heldout_rows':len(held),'train_rows':len(training),'memory_rows':len(mem.x),'sampled_positions':sampled})
        print('OOF_FOLD_COMPLETE '+json.dumps(fold_meta[-1]),flush=True)
        del fold_model,fold_trainer,mem
        if device.type=='cuda':torch.cuda.empty_cache()
    path=out/'oof_labels.npz'
    np.savez_compressed(path,features=np.concatenate(feats).astype(np.float32),utility=np.concatenate(util).astype(np.float32),risk=np.concatenate(risk).astype(np.float32),optimal_gate=np.concatenate(opt).astype(np.float32))
    json_write(out/'oof_manifest.json',{'folds':fold_meta,'source':'time-blocked, purged, independently trained fold-specific FARE candidates','file':str(path)})
    return path

def train_controller(controller,oof_path,cfg,device):
    z=np.load(oof_path,allow_pickle=False)
    ft=torch.from_numpy(z['features']);uy=torch.from_numpy(z['utility']);ry=torch.from_numpy(z['risk']);gy=torch.from_numpy(z['optimal_gate'])
    n=len(ft);gen=torch.Generator().manual_seed(int(cfg['experiment']['seed']))
    perm=torch.randperm(n,generator=gen);cut=int(n*.9);a,b=perm[:cut],perm[cut:]
    for p in controller.parameters():p.requires_grad=True
    controller.to(device);optim=torch.optim.AdamW(controller.parameters(),lr=cfg['training']['new_lr'])
    batch_size=4096;best=float('inf');best_state=None
    for ep in range(int(cfg['training']['oof_controller_epochs'])):
        controller.train();order=a[torch.randperm(len(a),generator=gen)]
        for lo in range(0,len(order),batch_size):
            ix=order[lo:lo+batch_size];f=ft[ix].to(device);u=uy[ix].to(device);r=ry[ix].to(device);g=gy[ix].to(device)
            U,R,G=controller.from_raw(f)
            loss=torch.nn.functional.smooth_l1_loss(U,u)+.1*torch.nn.functional.smooth_l1_loss(R,r)+.2*torch.nn.functional.mse_loss(G,g)
            optim.zero_grad(set_to_none=True);loss.backward();optim.step()
        controller.eval()
        with torch.no_grad():
            # bounded held-out chunk to avoid GPU memory spikes
            val=[]
            for ix in b.split(batch_size):
                U,R,G=controller.from_raw(ft[ix].to(device))
                val.append(float((torch.nn.functional.smooth_l1_loss(U,uy[ix].to(device))+.1*torch.nn.functional.smooth_l1_loss(R,ry[ix].to(device))+.2*torch.nn.functional.mse_loss(G,gy[ix].to(device))).cpu()))
            score=float(np.mean(val))
        print(f'OOF_CONTROLLER epoch={ep+1} val_loss={score:.6f}',flush=True)
        if score<best:best=score;best_state={k:v.cpu().clone() for k,v in controller.state_dict().items()}
    controller.load_state_dict(best_state);return best
