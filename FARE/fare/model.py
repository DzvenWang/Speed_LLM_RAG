"""Unified full FARE v4: support demand -> capability retrieval -> transport ->
set counterfactual selection -> STLA evidence integration -> risk control.

No target/future label is passed into model inference. The only Future available
inside forward belongs to historically completed, causally eligible episodes.
"""
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
        self.token=nn.Sequential(nn.Linear(3,backbone_dim),nn.LayerNorm(backbone_dim))
        self.adjust=nn.Sequential(nn.Linear(support_dim+3,support_dim),nn.GELU(),nn.Linear(support_dim,1))
        nn.init.zeros_(self.adjust[-1].weight);nn.init.zeros_(self.adjust[-1].bias)
    def forward(self,batch,base,backbone,support,selection,mode='full'):
        pred0=base['prediction']
        if mode in ('no_evidence','no_hidden_injection'):
            hidden=pred0
        else:
            vals=selection['selected_values'];w=selection['weights']
            stats=torch.cat([vals.mean(-1),w.sum((-1,-2))[...,None]],-1)
            token=self.token(stats)
            active=selection['selected_valid'].any((-1,-2))
            if hasattr(backbone,'decode_evidence') and 'prefix_cache' in base:
                hidden=backbone.decode_evidence(base['prefix_cache'],token,active)
            else:
                hidden=backbone(batch['x'],batch['mask'],batch['tod'],batch['dow'],token,active)['prediction']
        correction=selection['correction'].permute(0,2,1)
        if mode in ('no_evidence','no_forecast_correction'):correction=correction*0
        ctx=torch.cat([support['embedding'],pred0.permute(0,2,1)[...,None],
                       correction.permute(0,2,1)[...,None],selection['null'][...,None]],-1)
        scale=1+0.2*self.adjust(ctx).squeeze(-1).tanh()
        candidate=hidden+correction*scale.permute(0,2,1)
        return {'candidate_prediction':candidate,'hidden_prediction':hidden,
                'null_prediction':pred0,'forecast_delta':correction}

class RiskController(nn.Module):
    def __init__(self,d,alpha_max=1.25):
        super().__init__();self.alpha_max=float(alpha_max)
        self.net=nn.Sequential(nn.Linear(d+6,d),nn.GELU(),nn.Linear(d,1))
        self.gain=nn.Sequential(nn.Linear(d+6,d),nn.GELU(),nn.Linear(d,1))
        self.harm=nn.Sequential(nn.Linear(d+6,d),nn.GELU(),nn.Linear(d,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias,float(torch.logit(torch.tensor(1/self.alpha_max))))
        for v in (self.gain,self.harm):nn.init.zeros_(v[-1].weight);nn.init.zeros_(v[-1].bias)
    def forward(self,support,integrated,selection,mode='full',alpha_override=None):
        delta=(integrated['candidate_prediction']-integrated['null_prediction']).permute(0,2,1)
        feat=torch.cat([support['embedding'],delta[...,None],selection['null'][...,None],
                        selection['predicted_gain'][...,None],delta.abs()[...,None],
                        support['demand'][...,None],support['forward_coverage'][...,None]],-1)
        gain=self.gain(feat).squeeze(-1)
        harmful=self.harm(feat).squeeze(-1)
        alpha=self.alpha_max*self.net(feat).squeeze(-1).sigmoid()
        if mode=='no_evidence':alpha=alpha*0
        if mode=='no_risk_control':alpha=torch.ones_like(alpha)
        if alpha_override is not None:alpha=torch.full_like(alpha,float(alpha_override))
        pred=integrated['null_prediction']+alpha.permute(0,2,1)*delta.permute(0,2,1)
        return {'prediction':pred,'alpha':alpha,'gain_estimate':gain,
                'harm_logit':harmful,'base_prediction':integrated['null_prediction']}

class FARE(nn.Module):
    def __init__(self,backbone,graph,cfg):
        super().__init__();self.backbone=backbone;self.graph=graph;self.cfg=cfg
        d=int(cfg['model']['hidden_dim']);p=int(cfg['model']['horizon'])
        self.base_proj=nn.Linear(backbone.d,d)
        self.support=PropagationSupportEncoder(d,d,p)
        retr=cfg['retrieval']
        self.retriever=HierarchicalRetriever(d,retr['coarse_k'],retr['shortlist_k'],retr.get('chunk_size',4096))
        self.transfer=PropagationTransport(d,p,cfg.get('traffic',{}).get('max_temporal_lag',3))
        self.utility=SetConditionalSelector(d,p,retr['final_k'])
        self.integrator=ForecastIntegrator(d,backbone.d)
        self.risk=RiskController(d,cfg['model']['alpha_max'])
    def train(self,mode=True):
        super().train(mode);self.backbone.train(False);return self

    @staticmethod
    def _empty_selection(pred,k=1):
        b,p,n=pred.shape;dev=pred.device
        return {'selected_ids':torch.zeros(b,n,k,device=dev,dtype=torch.long),
                'selected_positions':torch.zeros(b,n,k,device=dev,dtype=torch.long),
                'weights':torch.zeros(b,n,k,2,p,device=dev),'null':torch.ones(b,n,p,device=dev),
                'correction':torch.zeros(b,n,p,device=dev),'predicted_gain':torch.zeros(b,n,p,device=dev),
                'all_utility':torch.zeros(b,n,k,2,p,device=dev),
                'all_benefit_logit':torch.zeros(b,n,k,2,p,device=dev),
                'selected_values':torch.zeros(b,n,k,2,p,device=dev),
                'selected_valid':torch.zeros(b,n,k,2,p,device=dev,dtype=torch.bool)}

    @staticmethod
    def edit_selection(selection,proposal,short,drop_slot=None,replace_slot=None,replace_position=None):
        sel={k:(v.clone() if torch.is_tensor(v) else v) for k,v in selection.items()}
        b,n,K,t,p=sel['weights'].shape
        if drop_slot is not None:
            slot=int(drop_slot)
            if not 0<=slot<K:raise ValueError('Invalid LOO slot')
            removed=sel['weights'][:,:,slot].sum(2)
            sel['null']=sel['null']+removed
            sel['weights'][:,:,slot]=0
            sel['selected_valid'][:,:,slot]=False
        if replace_slot is not None:
            slot=int(replace_slot)
            if not 0<=slot<K:raise ValueError('Invalid swap slot')
            if torch.is_tensor(replace_position):
                pos=replace_position.long().to(sel['weights'].device)
                if pos.shape!=(b,n):raise ValueError('Swap position tensor must be [B,N]')
            else:
                pos=torch.full((b,n),int(replace_position),device=sel['weights'].device,dtype=torch.long)
            if (pos<0).any() or (pos>=short['ids'].shape[-1]).any():
                raise ValueError('Invalid candidate position')
            offered=short['ids'].gather(2,pos[...,None]).squeeze(-1)
            allowed0=short['valid'].gather(2,pos[...,None]).squeeze(-1)
            duplicate=(sel['selected_ids']==offered[...,None]).any(-1)
            allowed=allowed0&(~duplicate)
            index=pos[...,None,None,None].expand(-1,-1,1,proposal['deltas'].shape[-2],proposal['deltas'].shape[-1])
            values=proposal['deltas'].gather(2,index).squeeze(2)
            good=proposal['available'].gather(2,index).squeeze(2)&allowed[...,None,None]
            old=sel['weights'][:,:,slot].clone()
            weight=torch.where(allowed[...,None,None],old*good.float(),old)
            sel['null']=sel['null']+(old-weight).sum(2)
            sel['weights'][:,:,slot]=weight
            sel['selected_values'][:,:,slot]=torch.where(allowed[...,None,None],values,sel['selected_values'][:,:,slot])
            sel['selected_valid'][:,:,slot]=torch.where(allowed[...,None,None],good,sel['selected_valid'][:,:,slot])
            sel['selected_ids'][:,:,slot]=torch.where(allowed,offered,sel['selected_ids'][:,:,slot])
            sel['selected_positions'][:,:,slot]=torch.where(allowed,pos,sel['selected_positions'][:,:,slot])
        sel['correction']=(sel['weights']*sel['selected_values']).sum((2,3))
        return sel

    def predict_with_selection(self,batch,prepared,selection,intervention='full',alpha_override=None):
        integrated=self.integrator(batch,prepared['base'],self.backbone,prepared['support'],selection,mode=intervention)
        risk=self.risk(prepared['support'],integrated,selection,mode=intervention,alpha_override=alpha_override)
        return {'prediction':risk['prediction'],'base_prediction':prepared['base']['prediction'],
                'null_prediction':integrated['null_prediction'],
                'candidate_prediction':integrated['candidate_prediction'],
                'alpha':risk['alpha'],'gain_estimate':risk['gain_estimate'],
                'harm_logit':risk['harm_logit'],'integrated':integrated,'selection':selection,
                'support':prepared['support']}

    def intervene_cached(self,batch,memory,full,mode,purge=0):
        """Exact fixed-parameter interventions with dependency-aware reuse.

        Upstream interventions must fall back to full forward; downstream ones
        reuse the SAME historical retrieval, transport and frozen Prefix.
        """
        if mode=='full':return full
        prepared=full['prepared']
        simple={'no_evidence','no_risk_control','no_hidden_injection','no_forecast_correction'}
        utility_modes={'no_null','no_set_utility','no_set_interaction','no_swap_utility',
                       'no_counterfactual_utility'}
        evidence_modes={'no_residual','no_evolution','no_residual_value','no_evolution_value',
                        'no_propagation_alignment'}
        if mode in simple:
            if mode=='no_evidence':sel=self._empty_selection(prepared['base']['prediction'])
            else:sel=full['selection']
            result=self.predict_with_selection(batch,prepared,sel,mode)
        elif mode in utility_modes|evidence_modes:
            short=prepared['candidate_pool']
            if mode in evidence_modes:
                proposals=self.transfer(short,memory,self.graph,prepared['support'],
                                        prepared['base']['prediction'],mode)
            else:proposals=prepared['proposals']
            selection=self.utility(short,proposals,prepared['support'],
                 mode if mode in utility_modes else None)
            newprep=dict(prepared,proposals=proposals)
            result=self.predict_with_selection(batch,newprep,selection,mode)
        else:
            return self.forward(batch,memory,intervention=mode,purge=purge,return_details=True)
        result.update({'prepared':prepared,'candidate_pool':prepared['candidate_pool'],
                       'proposals':prepared['proposals']})
        return result

    def forward(self,batch,memory,intervention='full',drop_slot=None,replace_slot=None,replace_position=None,
                alpha_override=None,purge=0,return_details=True):
        if memory.region_key_cpu is None:memory.attach_graph(self.graph)
        if hasattr(self.backbone,'encode_base'):
            base=self.backbone.encode_base(batch['x'],batch['mask'],batch['tod'],batch['dow'])
        else:base=self.backbone(batch['x'],batch['mask'],batch['tod'],batch['dow'])
        hidden=self.base_proj(base['hidden'])
        sup=self.support(batch['x'],batch['mask'],hidden,base['prediction'],self.graph,
                         neutral=intervention in ('no_support','no_support_demand'),
                         no_direction=intervention=='no_direction',
                         no_regime=intervention=='no_regime',
                         no_boundary=intervention=='no_boundary_support',
                         no_propagation_time=intervention=='no_propagation_time',
                         topology=batch.get('topology'))
        no_evidence=intervention=='no_evidence'
        if no_evidence:
            sel=self._empty_selection(base['prediction']);short=None;prop=None
        else:
            ret_mode=('state_similarity_only' if intervention in ('state_similarity_only','no_capability_matching')
                      else intervention if intervention in ('no_residual_information','no_evolution_information') else None)
            short=self.retriever(batch['x'],batch['mask'],sup,batch['origin'],memory,self.graph,purge,ret_mode)
            if intervention in ('random_memory','random_legal_memory','shuffled_memory','shuffled_legal_memory'):
                lawful=memory.valid_for(batch['origin'],batch['x'].device,purge)
                if intervention in ('random_memory','random_legal_memory'):
                    probs=lawful.float();empty=~lawful.any(-1);probs[empty,0]=1
                    ids=torch.multinomial(probs,short['ids'].shape[1]*short['ids'].shape[2],replacement=True)
                    ids=ids.reshape_as(short['ids'])
                    short['valid']=~empty[:,None,None].expand_as(short['valid'])
                else:
                    # Roll candidate EPISODES across query nodes, then reapply each query's legal time mask.
                    ids=short['ids'].roll(1,dims=1)
                    short['valid']=short['valid'].roll(1,dims=1)&lawful.gather(1,ids.reshape(ids.shape[0],-1)).reshape_as(ids)
                short['ids']=ids
                short['candidate_capability']=memory.fetch_node_keys(ids,batch['x'].device,'capability_cpu')
            transfer_mode=({'no_propagation_alignment':'no_propagation_alignment',
                            'no_direction':'no_direction',
                            'no_residual':'no_residual','no_evolution':'no_evolution',
                            'no_residual_information':'no_residual_information',
                            'no_evolution_information':'no_evolution_information',
                            'no_residual_value':'no_residual_value',
                            'no_evolution_value':'no_evolution_value'}).get(intervention)
            prop=self.transfer(short,memory,self.graph,sup,base['prediction'],transfer_mode)
            utility_mode=intervention if intervention in ('no_set_utility','no_set_interaction',
                                                          'no_swap_utility','no_counterfactual_utility',
                                                          'no_null','state_similarity_only') else None
            sel=self.utility(short,prop,sup,utility_mode)
            if drop_slot is not None or replace_slot is not None:
                sel=self.edit_selection(sel,prop,short,drop_slot,replace_slot,replace_position)
        prepared={'base':base,'support':sup,'candidate_pool':short,'proposals':prop}
        result=self.predict_with_selection(batch,prepared,sel,intervention,alpha_override)
        result['shortlist_ids']=short['ids'] if short is not None else None
        if return_details:
            result.update({'candidate_pool':short,'proposals':prop,'prepared':prepared})
        return result
