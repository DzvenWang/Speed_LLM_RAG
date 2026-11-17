"""M0 / M1 go-no-go diagnostics (no training of any ReShift module).

M0 (needs only ReShift's own S0 + Val banks): how well do failure-replayed
neighbour statistics predict the QUERY's own exact decomposition?
   corr(d_ret, P_full_q - P0_q), corr(r_ret, Y - P_full_q), corr(g_ret, gain_q)
   MAE of P0 + retrieved corrections, AUC of the neighbours' leave-one-out harm
   rate for "applying d_ret harms the query".
M1 (needs ReShift's own anchor_cf): extra per-element oracle headroom of the
retrieval arms over {Normal, Normal + CF}, and whether the LOO harm rate
separates harmful CF candidates.

Both read Validation labels (and, for M0, the Validation complete history that
the benchmark already uses to build masked validation scenarios) on the
SELECTION partition only, purely as an offline analysis.  They never touch TEST
and no checkpoint is selected with them.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from reshift.util import to_device
from reshift.stla.backbone import base_features
from reshift.retrieval.stats import evidence_stats
from reshift.data.dataset import FAMILIES


class _Corr:
    def __init__(self):self.s=np.zeros(6)
    def add(self,a,b,v):
        m=v.bool();a=a[m].double();b=b[m].double()
        self.s+=np.array([float(a.numel()),float(a.sum()),float(b.sum()),float((a*a).sum()),
                          float((b*b).sum()),float((a*b).sum())])
    def value(self):
        n,sa,sb,saa,sbb,sab=self.s
        if n<2:return None
        cov=sab-sa*sb/n;va=saa-sa*sa/n;vb=sbb-sb*sb/n
        return float(cov/np.sqrt(max(va*vb,1e-30)))


class _AUC:
    """Streaming ROC-AUC from a 0..1 score histogram (ties count one half)."""
    def __init__(self,bins=200):self.pos=np.zeros(bins);self.neg=np.zeros(bins);self.bins=bins
    def add(self,score,label,v):
        m=v.bool();s=score[m].float().clamp(0,1);l=label[m].bool()
        idx=(s*(self.bins-1)).round().long().cpu().numpy();lab=l.cpu().numpy()
        self.pos+=np.bincount(idx[lab],minlength=self.bins);self.neg+=np.bincount(idx[~lab],minlength=self.bins)
    def value(self):
        P=self.pos.sum();N=self.neg.sum()
        if P==0 or N==0:return None
        below=np.cumsum(self.neg)-self.neg
        return float(((self.pos*(below+0.5*self.neg)).sum())/(P*N))


def _mae():return [0.,0.]


def _full_history(ds,rows,device):
    xs=[];ms=[]
    for r in rows.tolist():
        src=ds.raw[r];ov=np.asarray(src['original_valid'].numpy(),dtype=bool)
        full=np.nan_to_num(np.asarray(ds.full[r],dtype=np.float32),nan=0.,posinf=0.,neginf=0.)
        xs.append(np.where(ov,full,0.).astype(np.float32));ms.append(ov.astype(np.float32))
    return torch.from_numpy(np.stack(xs)).to(device),torch.from_numpy(np.stack(ms)).to(device)


@torch.no_grad()
def m0(stla,provider,val_ds,indices,device,mean,std,tau,batch_size=32):
    out={}
    for sev,ds in val_ds.items():
        if ds.full is None:raise RuntimeError('M0 needs val_x_full_norm.npy (validation scenario source)')
        corr={k:_Corr() for k in ('d_ret~cf','r_ret~residual','g_ret~gain')}
        auc=_AUC();maes={k:_mae() for k in ('p0','p0+d_ret','p0+r_ret','p0+d_ret+r_ret','oracle_p_full')}
        fam={f:{'p0':_mae(),'p0+d_ret+r_ret':_mae()} for f in FAMILIES}
        for raw in DataLoader(Subset(ds,indices),batch_size=batch_size,shuffle=False):
            b=to_device(raw,device)
            ev=provider.batch(f'val_s{sev}',b['index'],'full')
            _,_,p0=base_features(stla,b['x'],b['mask'],b['tod'],b['dow'])
            xf,mf=_full_history(ds,raw['index'],device)
            _,_,pf=base_features(stla,xf,mf,b['tod'],b['dow'])
            p0=p0.float();pf=pf.float();v=b['future_valid'].float();y=b['y_norm'].float()
            z=torch.zeros_like(p0)
            st=evidence_stats(ev,tau,z,z)
            corr['d_ret~cf'].add(st['d_ret'],pf-p0,v)
            corr['r_ret~residual'].add(st['r_ret'],y-pf,v)
            corr['g_ret~gain'].add(st['g_ret'],(p0-y).abs()-(pf-y).abs(),v)
            harmful=((p0+st['d_ret']-y).abs()>(p0-y).abs()).float()
            auc.add(st['features'][...,5],harmful,v)
            preds={'p0':p0,'p0+d_ret':p0+st['d_ret'],'p0+r_ret':p0+st['r_ret'],
                   'p0+d_ret+r_ret':p0+st['d_ret']+st['r_ret'],'oracle_p_full':pf}
            yr=b['y_raw'].float()
            for k,p in preds.items():
                e=((p*std+mean)-yr).abs()*v;maes[k][0]+=float(e.sum());maes[k][1]+=float(v.sum())
            for fi,f in enumerate(FAMILIES):
                m=(b['family_id']==fi).float().view(-1,1,1)*v
                if float(m.sum())==0:continue
                for k in fam[f]:
                    e=((preds[k]*std+mean)-yr).abs()*m;fam[f][k][0]+=float(e.sum());fam[f][k][1]+=float(m.sum())
        mae={k:s/max(c,1.) for k,(s,c) in maes.items()}
        out[str(sev)]={'correlation':{k:c.value() for k,c in corr.items()},
                       'loo_harm_auc_for_d_ret_harm':auc.value(),
                       'mae_raw':mae,
                       'relative_gain_vs_p0':{k:(mae['p0']-v_)/max(mae['p0'],1e-12) for k,v_ in mae.items()},
                       'per_family_mae_raw':{f:{k:s/max(c,1.) for k,(s,c) in d.items()}
                                             for f,d in fam.items() if d['p0'][1]>0}}
    return out


@torch.no_grad()
def m1(model,provider,val_ds,indices,device,mean,std,tau,batch_size=32):
    """Model must be at its trained anchor_cf state (model.stage == 'anchor_cf')."""
    model.eval();out={}
    names=('normal','normal+cf','normal+ret_res','normal+ret_shift','normal+ret_both')
    for sev,ds in val_ds.items():
        maes={k:_mae() for k in names+('oracle{normal,cf}','oracle{all}')}
        auc_cf=_AUC();auc_both=_AUC()
        for raw in DataLoader(Subset(ds,indices),batch_size=batch_size,shuffle=False):
            b=to_device(raw,device)
            ev=provider.batch(f'val_s{sev}',b['index'],'full')
            o=model({k:b[k] for k in ('x','mask','tod','dow')},enable_cf=True,enable_controller=False)
            pn=o['normal_prediction'].float();dh=o['delta_cf_candidate'].float()
            rh=(pn-o['backbone'].float())
            st=evidence_stats(ev,tau,rh,dh)
            corr={'normal':torch.zeros_like(pn),'normal+cf':dh,
                  'normal+ret_res':st['r_ret']-rh,'normal+ret_shift':st['d_ret'],
                  'normal+ret_both':st['r_ret']-rh+st['d_ret']}
            v=b['future_valid'].float();yr=b['y_raw'].float();y=b['y_norm'].float()
            err={k:((pn+c)*std+mean-yr).abs() for k,c in corr.items()}
            err['oracle{normal,cf}']=torch.minimum(err['normal'],err['normal+cf'])
            err['oracle{all}']=torch.stack([err[k] for k in names]).min(0).values
            for k,e in err.items():maes[k][0]+=float((e*v).sum());maes[k][1]+=float(v.sum())
            auc_cf.add(st['features'][...,5],((pn+dh-y).abs()>(pn-y).abs()).float(),v)
            auc_both.add(st['features'][...,5],((pn+corr['normal+ret_both']-y).abs()>(pn-y).abs()).float(),v)
        mae={k:s/max(c,1.) for k,(s,c) in maes.items()}
        out[str(sev)]={'mae_raw':mae,
                       'oracle_headroom_of_retrieval_arms':(mae['oracle{normal,cf}']-mae['oracle{all}'])
                                                          /max(mae['oracle{normal,cf}'],1e-12),
                       'loo_harm_auc_for_cf_harm':auc_cf.value(),
                       'loo_harm_auc_for_ret_both_harm':auc_both.value()}
    return out
