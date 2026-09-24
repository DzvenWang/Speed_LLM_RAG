#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,json,math
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml,write_json,seed_all,log
from presure.contract import cache_dir
from presure.data import CacheSplit,ForecastDataset,deterministic_split
from presure.paths import artifact_dir,run_dir
from presure.model import DualQueryConditionalForecaster
from presure.metrics import metric_dict_np,weighted_forecast_loss,improvement_pct,horizon_masked_mae


def to_device(b,device): return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in b.items()}

@torch.no_grad()
def evaluate(model,dl,device,mean,std):
    model.eval(); preds=[]; rets=[]; stats={'alpha_sum':0.0,'alpha_sq':0.0,'alpha_n':0,'attn_max_sum':0.0,'attn_ent_sum':0.0,'attn_n':0,'dual_gate_sum':0.0,'dual_gate_n':0,'delta_abs_sum':0.0,'delta_n':0}
    for b in dl:
        b=to_device(b,device); o=model(b['query_state'],b['query_evidence'],b['query_base'],b['support_state'],b['support_evidence'],b['support_future'],b['support_base'],b['support_future_valid'],b['candidate_scores'],mean,std); preds.append(o['pred'].cpu().numpy()); rets.append(o['retrieval_pred'].cpu().numpy())
        a=o['alpha']; stats['alpha_sum']+=float(a.sum()); stats['alpha_sq']+=float((a*a).sum()); stats['alpha_n']+=a.numel(); at=o['attention']; stats['attn_max_sum']+=float(at.max(-1).values.sum()); stats['attn_ent_sum']+=float(o['attention_entropy'].sum()); stats['attn_n']+=o['attention_entropy'].numel(); g=o['dual_gate_query']; stats['dual_gate_sum']+=float(g.sum()); stats['dual_gate_n']+=g.numel(); d=torch.abs(o['delta']); stats['delta_abs_sum']+=float(d.sum()); stats['delta_n']+=d.numel()
    am=stats['alpha_sum']/max(stats['alpha_n'],1); av=max(stats['alpha_sq']/max(stats['alpha_n'],1)-am*am,0.0)
    aux={'alpha_mean':am,'alpha_std':math.sqrt(av),'attention_max_mean':stats['attn_max_sum']/max(stats['attn_n'],1),'attention_entropy_mean':stats['attn_ent_sum']/max(stats['attn_n'],1),'dual_gate_mean':stats['dual_gate_sum']/max(stats['dual_gate_n'],1),'delta_abs_mean':stats['delta_abs_sum']/max(stats['delta_n'],1)}
    return np.concatenate(preds),np.concatenate(rets),aux

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); a=ap.parse_args(); cfg=load_yaml(ROOT/'configs/presure_v12.yaml'); seed_all(cfg['experiment']['seed']); out=artifact_dir(cfg,a.stage); ready=json.loads((out/'READY.json').read_text()); ret_summary=json.loads((out/'retriever_summary.json').read_text())
    cache=cache_dir(cfg,a.stage); tr=CacheSplit(cache,'train'); va=CacheSplit(cache,'val'); tb=np.load(out/'train_base_pred.npy',mmap_mode='r'); vb=np.load(out/'val_base_pred.npy',mmap_mode='r'); ts=np.load(out/'train_node_state.npy',mmap_mode='r'); vs=np.load(out/'val_node_state.npy',mmap_mode='r'); te=np.load(out/'train_node_evidence.npy',mmap_mode='r'); ve=np.load(out/'val_node_evidence.npy',mmap_mode='r'); tk=np.load(out/'train_topk8.npz'); vk=np.load(out/'val_topk8.npz'); ti,tis=tk['indices'],tk['scores']; vi,vis=vk['indices'],vk['scores']
    fc=cfg['forecaster']; seed=int(cfg['experiment']['seed']); rng=np.random.default_rng(seed); train_ids=rng.permutation(len(tr))[:min(int(fc['train_queries']),len(tr))]; dev_ids,final_ids=deterministic_split(len(va),int(fc['dev_queries']),int(fc['final_val_queries']),seed)
    rdir=run_dir(cfg,'DualQueryRetrievalConditionedForecasting',a.stage); rdir.mkdir(parents=True,exist_ok=True); np.savez(rdir/'split_ids.npz',train=train_ids,dev=dev_ids,final=final_ids); (out/'LATEST_RUN.txt').write_text(str(rdir)+'\n')
    train_ds=ForecastDataset(tr,tr,tb,tb,ts,ts,te,te,ti,tis,train_ids); dev_ds=ForecastDataset(va,tr,vb,tb,vs,ts,ve,te,vi,vis,dev_ids); final_ds=ForecastDataset(va,tr,vb,tb,vs,ts,ve,te,vi,vis,final_ids)
    rt=cfg['runtime']; nw=int(rt['num_workers']); common={'num_workers':nw,'pin_memory':bool(rt['pin_memory']) and torch.cuda.is_available()};
    if nw>0: common.update(prefetch_factor=int(rt['prefetch_factor']),persistent_workers=bool(rt['persistent_workers']))
    train_dl=DataLoader(train_ds,batch_size=int(fc['batch_size']),shuffle=True,drop_last=False,**common); dev_dl=DataLoader(dev_ds,batch_size=int(fc['eval_batch_size']),shuffle=False,drop_last=False,**common); final_dl=DataLoader(final_ds,batch_size=int(fc['eval_batch_size']),shuffle=False,drop_last=False,**common)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); mean=float(ready['speed_mean']); std=float(ready['speed_std']); model=DualQueryConditionalForecaster(int(ready['state_node_dim']),int(ready['evidence_node_dim']),int(cfg['experiment']['horizon']),int(fc['d_model']),int(fc['state_hidden']),int(fc['evidence_hidden']),int(fc['support_hidden']),int(fc['decoder_hidden']),int(fc['attention_heads']),float(fc['dropout']),float(fc['selector_prior_weight']),float(fc['cross_attention_temperature']),float(fc['alpha_init']),float(fc['max_delta_sigma']),int(fc.get('attention_batch_chunk',32768))).to(device); opt=torch.optim.AdamW(model.parameters(),lr=float(fc['lr']),weight_decay=float(fc['weight_decay']))
    if device.type=='cuda': p=torch.cuda.get_device_properties(device); log(f'GPU {p.name} total_memory={p.total_memory/1024**3:.2f}GB train_batch={fc["batch_size"]} eval_batch={fc["eval_batch_size"]} TopK={cfg["candidate_pool"]["top_k"]} support_attn_chunk={fc.get("attention_batch_chunk",32768)}')
    base_dev=metric_dict_np(np.asarray(vb[dev_ids],np.float32),np.asarray(va.y[dev_ids],np.float32),np.asarray(va.valid[dev_ids],bool),np.asarray(va.direct[dev_ids],bool),np.asarray(va.support[dev_ids],bool)); log(f'BASE DEV masked_MAE={base_dev["masked_mae"]:.6f}; Final Validation remains locked')
    best=None; bad=0; loss_cfg=cfg['loss']; K=int(cfg['candidate_pool']['top_k'])
    for ep in range(1,int(fc['epochs'])+1):
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        model.train(); total=[]; fused_ls=[]; ret_ls=[]; alpha=[]; ent=[]; amax=[]
        for b in train_dl:
            b=to_device(b,device); o=model(b['query_state'],b['query_evidence'],b['query_base'],b['support_state'],b['support_evidence'],b['support_future'],b['support_base'],b['support_future_valid'],b['candidate_scores'],mean,std)
            lf,_=weighted_forecast_loss(o['pred'],b['target'],b['future_valid'],b['direct_mask'],b['support_mask'],loss_cfg); lr,_=weighted_forecast_loss(o['retrieval_pred'],b['target'],b['future_valid'],b['direct_mask'],b['support_mask'],loss_cfg)
            floor=float(loss_cfg['attention_entropy_floor']); epen=torch.relu(torch.tensor(floor,device=device)-o['attention_entropy']).pow(2).mean(); asmooth=torch.abs(o['alpha'][:,1:]-o['alpha'][:,:-1]).mean() if o['alpha'].shape[1]>1 else o['alpha'].sum()*0.0; dpen=torch.abs(o['delta']).mean()
            loss=lf+float(loss_cfg['retrieval_branch_aux_weight'])*lr+float(loss_cfg['attention_entropy_weight'])*epen+float(loss_cfg['alpha_smooth_weight'])*asmooth+float(loss_cfg['delta_l1_weight'])*dpen
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(fc['grad_clip'])); opt.step(); total.append(float(loss.item())); fused_ls.append(float(lf.item())); ret_ls.append(float(lr.item())); alpha.append(float(o['alpha'].mean())); ent.append(float(o['attention_entropy'].mean())); amax.append(float(o['attention'].max(-1).values.mean()))
        dp,dr,daux=evaluate(model,dev_dl,device,mean,std); dm=metric_dict_np(dp,np.asarray(va.y[dev_ids],np.float32),np.asarray(va.valid[dev_ids],bool),np.asarray(va.direct[dev_ids],bool),np.asarray(va.support[dev_ids],bool)); rm=metric_dict_np(dr,np.asarray(va.y[dev_ids],np.float32),np.asarray(va.valid[dev_ids],bool),np.asarray(va.direct[dev_ids],bool),np.asarray(va.support[dev_ids],bool)); imp=improvement_pct(base_dev['masked_mae'],dm['masked_mae']); rimp=improvement_pct(base_dev['masked_mae'],rm['masked_mae']); mem=''
        if device.type=='cuda': mem=f' cuda_peak_alloc={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB cuda_peak_reserved={torch.cuda.max_memory_reserved(device)/1024**3:.2f}GB'
        log(f'V1.2 EPOCH {ep:02d} train={np.mean(total):.5f} dev_masked={dm["masked_mae"]:.6f} fused_gain={imp:+.3f}% retrieval_branch_gain={rimp:+.3f}% alpha={daux["alpha_mean"]:.3f}±{daux["alpha_std"]:.3f} attnmax={daux["attention_max_mean"]:.3f} entropy={daux["attention_entropy_mean"]:.3f} |delta|={daux["delta_abs_mean"]:.3f}{mem}')
        score=dm['masked_mae']
        if best is None or score<best['score']-1e-6:
            best={'score':score,'epoch':ep,'dev':dm,'ret_dev':rm,'aux':daux,'state':{k:v.detach().cpu() for k,v in model.state_dict().items()}}; bad=0; torch.save({'version':cfg['version'],'epoch':ep,'state_dict':best['state'],'state_dim':int(ready['state_node_dim']),'evidence_dim':int(ready['evidence_node_dim']),'speed_mean':mean,'speed_std':std,'config':cfg},rdir/'best.pt')
        else:
            bad+=1
            if bad>=int(fc['patience']): log(f'EARLY STOP epoch={ep} best_epoch={best["epoch"]}'); break
    model.load_state_dict(best['state']); dp,dr,daux=evaluate(model,dev_dl,device,mean,std); dm=metric_dict_np(dp,np.asarray(va.y[dev_ids],np.float32),np.asarray(va.valid[dev_ids],bool),np.asarray(va.direct[dev_ids],bool),np.asarray(va.support[dev_ids],bool)); drm=metric_dict_np(dr,np.asarray(va.y[dev_ids],np.float32),np.asarray(va.valid[dev_ids],bool),np.asarray(va.direct[dev_ids],bool),np.asarray(va.support[dev_ids],bool)); dimp=improvement_pct(base_dev['masked_mae'],dm['masked_mae'])
    log(f'FINAL VALIDATION UNLOCKED after best epoch={best["epoch"]} frozen on Development')
    base_final=metric_dict_np(np.asarray(vb[final_ids],np.float32),np.asarray(va.y[final_ids],np.float32),np.asarray(va.valid[final_ids],bool),np.asarray(va.direct[final_ids],bool),np.asarray(va.support[final_ids],bool)); fp,fr,faux=evaluate(model,final_dl,device,mean,std); fm=metric_dict_np(fp,np.asarray(va.y[final_ids],np.float32),np.asarray(va.valid[final_ids],bool),np.asarray(va.direct[final_ids],bool),np.asarray(va.support[final_ids],bool)); frm=metric_dict_np(fr,np.asarray(va.y[final_ids],np.float32),np.asarray(va.valid[final_ids],bool),np.asarray(va.direct[final_ids],bool),np.asarray(va.support[final_ids],bool)); fimp=improvement_pct(base_final['masked_mae'],fm['masked_mae'])
    np.save(rdir/'dev_pred.npy',dp.astype(np.float16)); np.save(rdir/'final_val_pred.npy',fp.astype(np.float16)); np.save(rdir/'dev_retrieval_pred.npy',dr.astype(np.float16)); np.save(rdir/'final_retrieval_pred.npy',fr.astype(np.float16))
    bh=horizon_masked_mae(np.asarray(vb[final_ids],np.float32),np.asarray(va.y[final_ids],np.float32),np.asarray(va.valid[final_ids],bool),np.asarray(va.direct[final_ids],bool)); mh=horizon_masked_mae(fp,np.asarray(va.y[final_ids],np.float32),np.asarray(va.valid[final_ids],bool),np.asarray(va.direct[final_ids],bool)); write_json(rdir/'horizon_metrics.json',[{'horizon':i+1,'base_masked_mae':bh[i],'model_masked_mae':mh[i],'improvement_pct':improvement_pct(bh[i],mh[i])} for i in range(len(bh))])
    gate=cfg['stage_gate']; decision='GO_V13' if dimp>=float(gate['dev_masked_mae_improvement_min_pct']) and fimp>=float(gate['final_masked_mae_improvement_min_pct']) else ('CONDITIONAL_RETRIEVAL' if dimp>=float(gate['stop_dev_below_pct']) and fimp>=0.0 else 'STOP_RETRIEVAL_PRIMARY')
    summary={'version':cfg['version'],'best_epoch':best['epoch'],'run_dir':str(rdir),'retriever_dev':ret_summary['dev'],'baseline_dev':base_dev,'model_dev':dm,'retrieval_branch_dev':drm,'dev_aux':daux,'dev_masked_improvement_pct':dimp,'baseline_final':base_final,'model_final':fm,'retrieval_branch_final':frm,'final_aux':faux,'final_masked_improvement_pct':fimp,'decision':decision,'stage_gate':gate,'design':'Dual Query + Top8 multi-support episode encoding + support-set self-attention + node-horizon cross-attention + standalone retrieval-conditioned forecast + adaptive forecast mixture','scientific_scope':cfg['scientific_scope']}; write_json(rdir/'summary.json',summary); (rdir/'DECISION.txt').write_text(decision+'\n')
    log('='*80); log(f'BEST epoch={best["epoch"]}'); log(f'DEV masked: {base_dev["masked_mae"]:.6f} -> {dm["masked_mae"]:.6f} ({dimp:+.3f}%) retrieval_branch={drm["masked_mae"]:.6f}'); log(f'FINAL masked: {base_final["masked_mae"]:.6f} -> {fm["masked_mae"]:.6f} ({fimp:+.3f}%) retrieval_branch={frm["masked_mae"]:.6f}'); log(f'FINAL AUX alpha={faux["alpha_mean"]:.3f}±{faux["alpha_std"]:.3f} attnmax={faux["attention_max_mean"]:.3f} entropy={faux["attention_entropy_mean"]:.3f}'); log(f'DECISION={decision}'); log(f'SUMMARY={rdir/"summary.json"}'); log('='*80)
if __name__=='__main__': main()
