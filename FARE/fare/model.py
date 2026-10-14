"""One unified FARE model, with semantic fixed-checkpoint intervention paths."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .traffic import PropagationSupportEncoder
from .retrieval import HierarchicalRetriever
from .transfer import PropagationTransport
from .utility import SetConditionalSelector

class ForecastIntegrator(nn.Module):
    def __init__(self,support_dim,backbone_dim):
        super().__init__()
        self.token=nn.Linear(3,backbone_dim)
        self.correction=nn.Sequential(nn.Linear(support_dim+3,support_dim),nn.GELU(),nn.Linear(support_dim,1))
        # small initial correction, not an unconstrained independent forecast model
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
    def forward(self,batch,base,backbone,support,selection,mode='full'):
        pred0=base['prediction'];b,p,n=pred0.shape
        w=selection['weights'];vals=selection['selected_values']
        if mode in ('no_evidence','no_hidden_injection'):
            pred_hidden=pred0
        else:
            raw=torch.cat([vals.mean(-1),w.sum(-1).sum(3)[...,None]],-1)
            tok=self.token(raw)
            active=selection['selected_valid'].any((3,4))
            pred_hidden=backbone(batch['x'],batch['mask'],batch['tod'],batch['dow'],tok,active)['prediction']
        corr=selection['correction'].permute(0,2,1)
        if mode in ('no_evidence','no_forecast_correction'):corr=torch.zeros_like(corr)
        inp=torch.cat([support['embedding'],pred0.permute(0,2,1)[...,None],corr.permute(0,2,1)[...,None],
                       selection['null'][...,None]],-1)
        scale=1+0.2*self.correction(inp).squeeze(-1).tanh()
        candidate=pred_hidden+corr*scale.permute(0,2,1)
        return {'candidate_prediction':candidate,'hidden_prediction':pred_hidden,'forecast_delta':corr,
                'null_prediction':pred0}

class RiskController(nn.Module):
    def __init__(self,d,alpha_max=1.25):
        super().__init__();self.alpha_max=float(alpha_max)
        self.net=nn.Sequential(nn.Linear(d+5,d),nn.GELU(),nn.Linear(d,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias,float(torch.logit(torch.tensor(1.0/self.alpha_max))))
        self.gain=nn.Sequential(nn.Linear(d+5,d),nn.GELU(),nn.Linear(d,1))
        nn.init.zeros_(self.gain[-1].weight);nn.init.zeros_(self.gain[-1].bias)
    def forward(self,support,integrated,selection,mode='full',alpha_override=None):
        delta=(integrated['candidate_prediction']-integrated['null_prediction']).permute(0,2,1)
        inp=torch.cat([support['embedding'],delta[...,None],selection['null'][...,None],
                       selection['predicted_gain'][...,None],
                       delta.abs()[...,None],support['forward_coverage'][...,None]],-1)
        predicted_gain=self.gain(inp).squeeze(-1)
        raw=self.net(inp).squeeze(-1)+0.1*predicted_gain
        alpha=self.alpha_max*raw.sigmoid()
        if mode in ('no_evidence','no_risk_control'):
            alpha=torch.zeros_like(alpha) if mode=='no_evidence' else torch.ones_like(alpha)
        if alpha_override is not None:alpha=torch.full_like(alpha,float(alpha_override))
        prediction=integrated['null_prediction']+alpha.permute(0,2,1)*delta.permute(0,2,1)
        return {'prediction':prediction,'alpha':alpha,'gain_estimate':predicted_gain,
                'base_prediction':integrated['null_prediction']}

class FARE(nn.Module):
    def __init__(self,backbone,graph,cfg):
        super().__init__();self.backbone=backbone;self.graph=graph;self.cfg=cfg
        d=cfg['model']['hidden_dim'];p=cfg['model']['horizon']
        self.base_proj=nn.Linear(backbone.d,d)
        self.support=PropagationSupportEncoder(d,d,p)
        self.retriever=HierarchicalRetriever(d,cfg['retrieval']['coarse_k'],cfg['retrieval']['shortlist_k'])
        self.transfer=PropagationTransport(d,p)
        self.utility=SetConditionalSelector(d,p,cfg['retrieval']['final_k'])
        self.integrator=ForecastIntegrator(d,backbone.d)
        self.risk=RiskController(d,cfg['model']['alpha_max'])
    def train(self,mode=True):
        super().train(mode);self.backbone.train(False);return self
    @staticmethod
    def _empty_selection(base,k=1):
        b,p,n=base.shape;device=base.device
        return {'selected_ids':torch.zeros(b,n,k,device=device,dtype=torch.long),
                'selected_positions':torch.zeros(b,n,k,device=device,dtype=torch.long),
                'weights':torch.zeros(b,n,k,2,p,device=device),'null':torch.ones(b,n,p,device=device),
                'correction':torch.zeros(b,n,p,device=device),'predicted_gain':torch.zeros(b,n,p,device=device),
                'all_utility':torch.zeros(b,n,k,2,p,device=device),'selected_values':torch.zeros(b,n,k,2,p,device=device),
                'selected_valid':torch.zeros(b,n,k,2,p,device=device,dtype=torch.bool)}
    @staticmethod
    def edit_selection(selection,proposal,short,drop_slot=None,replace_slot=None,replace_position=None):
        """Weight-preserving intervention; reserved weight is transferred to Null for LOO."""
        sel={k:v.clone() if torch.is_tensor(v) else v for k,v in selection.items()}
        b,n,kk,t,p=sel['weights'].shape
        if drop_slot is not None:
            slot=int(drop_slot)
            if not 0<=slot<kk:raise ValueError('drop_slot out of range')
            removed=sel['weights'][:,:,slot].sum(2)
            sel['null']=sel['null']+removed
            sel['weights'][:,:,slot]=0
            sel['selected_valid'][:,:,slot]=False
        if replace_slot is not None:
            slot=int(replace_slot);pos=int(replace_position)
            if not 0<=slot<kk or not 0<=pos<short['ids'].shape[-1]:raise ValueError('replacement index out of range')
            newv=proposal['deltas'][:,:,pos]
            valid=proposal['available'][:,:,pos]
            original_weight=sel['weights'][:,:,slot].clone()
            removed=(original_weight*(~valid).float()).sum(2)
            sel['null']=sel['null']+removed
            sel['weights'][:,:,slot]=original_weight*valid.float()
            sel['selected_values'][:,:,slot]=newv
            sel['selected_valid'][:,:,slot]=valid
            sel['selected_ids'][:,:,slot]=short['ids'][:,:,pos]
        sel['correction']=(sel['weights']*sel['selected_values']).sum((2,3))
        return sel
    def forward(self,batch,memory,intervention='full',drop_slot=None,replace_slot=None,replace_position=None,
                alpha_override=None,purge=0,return_details=True):
        base=self.backbone(batch['x'],batch['mask'],batch['tod'],batch['dow'])
        hidden=self.base_proj(base['hidden'])
        no_evidence=intervention=='no_evidence'
        sup=self.support(batch['x'],batch['mask'],hidden,base['prediction'],self.graph,
                         neutral=intervention=='no_support',no_direction=intervention=='no_direction',
                         no_regime=intervention=='no_regime')
        if no_evidence:
            sel=self._empty_selection(base['prediction']);short=None;prop=None
        else:
            short=self.retriever(batch['x'],batch['mask'],sup,batch['origin'],memory,self.graph,purge=purge)
            if intervention in ('random_memory','shuffled_memory'):
                if intervention=='random_memory':
                    lawful=memory.valid_for(batch['origin'],batch['x'].device,purge=purge).float()
                    has=lawful.sum(-1)>0
                    lawful[~has,0]=1.0
                    ids=torch.multinomial(lawful,short['ids'].shape[1]*short['ids'].shape[2],replacement=True).view_as(short['ids'])
                    short['valid']=has[:,None,None].expand_as(short['valid'])
                else:
                    ids=short['ids'].roll(1,dims=1)
                    short['valid']=short['valid'].roll(1,dims=1)
                short['ids']=ids
            prop=self.transfer(short,memory,self.graph,sup,base['prediction'],
                               intervention=('no_alignment' if intervention=='no_propagation_alignment' else intervention) if intervention in ('no_residual','no_evolution','no_propagation_alignment','no_direction') else None)
            sel=self.utility(short,prop,sup,intervention=intervention if intervention in ('no_set_utility','no_counterfactual_utility','no_null') else None)
            if drop_slot is not None or replace_slot is not None:
                sel=self.edit_selection(sel,prop,short,drop_slot,replace_slot,replace_position)
        integrated=self.integrator(batch,base,self.backbone,sup,sel,mode=intervention)
        risk=self.risk(sup,integrated,sel,mode=intervention,alpha_override=alpha_override)
        out={'shortlist_ids':short['ids'] if short is not None else None,
             'prediction':risk['prediction'],'base_prediction':base['prediction'],
             'candidate_prediction':integrated['candidate_prediction'],
             'null_prediction':integrated['null_prediction'],'alpha':risk['alpha'],
             'gain_estimate':risk['gain_estimate'],'support':sup,'selection':sel}
        if return_details:out.update({'candidate_pool':short,'proposals':prop,'integrated':integrated})
        return out
