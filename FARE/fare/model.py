"""FARE v5: counterfactual-utility retrieval and safe horizon-wise evidence injection.

No legacy compatibility is promised; outputs retain standard train/evaluate fields.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .traffic import PropagationSupportEncoder
from .retrieval import HierarchicalRetriever
from .transfer import PropagationTransport
from .forecast_utility import ForecastUtilityRanker,ConditionalSetSelector,TemporalEvidenceEncoder

class ForecastIntegrator(nn.Module):
    def __init__(self,support_dim,backbone_dim,horizon):
        super().__init__()
        self.temporal=TemporalEvidenceEncoder(backbone_dim,horizon)
        self.context=nn.Linear(support_dim,backbone_dim)
        self.correction_gain=nn.Parameter(torch.tensor(.1))
    def forward(self,batch,base,backbone,support,selection,mode='full'):
        pred0=base['prediction']
        if mode in ('no_evidence','no_hidden_injection'):
            hidden=pred0
        else:
            # Full, temporally ordered [Residual, Evolution, validity] sequences.
            temporal={'deltas':selection['selected_values'],
                      'available':selection['selected_valid'],
                      'compatibility':torch.ones_like(selection['selected_values'][...,0,:])}
            # Compatibility is [B,N,K,H]; zero-selected evidence never becomes an active token.
            q=self.context(support['embedding'])
            token=self.temporal(temporal,q,selection['weights'])
            active=selection['selected_valid'].any((-1,-2)) & (selection['weights'].sum((-1,-2))>1e-8)
            if hasattr(backbone,'decode_evidence') and 'prefix_cache' in base:
                hidden=backbone.decode_evidence(base['prefix_cache'],token,active)
            else:
                hidden=backbone(batch['x'],batch['mask'],batch['tod'],batch['dow'],token,active)['prediction']
        correction=selection['correction'].permute(0,2,1)
        if mode in ('no_evidence','no_forecast_correction'):correction=correction*0
        candidate=hidden+self.correction_gain.tanh()*correction
        return {'candidate_prediction':candidate,'hidden_prediction':hidden,
                'null_prediction':pred0,'forecast_delta':correction}

class RiskController(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d+6,d),nn.GELU(),nn.Linear(d,1))
        self.gain=nn.Sequential(nn.Linear(d+6,d),nn.GELU(),nn.Linear(d,1))
        self.harm=nn.Sequential(nn.Linear(d+6,d),nn.GELU(),nn.Linear(d,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,-.5)
    def forward(self,support,integrated,selection,mode='full',alpha_override=None):
        delta=(integrated['candidate_prediction']-integrated['null_prediction']).permute(0,2,1)
        feat=torch.cat([support['embedding'],delta[...,None],selection['null'][...,None],
                        selection['predicted_gain'][...,None],delta.abs()[...,None],
                        support['demand'][...,None],support['forecast_deficit'][...,None]],-1)
        alpha=self.net(feat).squeeze(-1).sigmoid()
        if mode=='no_risk_control':alpha=torch.ones_like(alpha)
        if alpha_override is not None:alpha=torch.full_like(alpha,float(alpha_override))
        # Mathematical rejection: when all evidence mass is NULL, output is exactly base.
        alpha=alpha*(1-selection['null']).clamp(0,1)
        if mode=='no_evidence':alpha=alpha*0
        pred=integrated['null_prediction']+alpha.permute(0,2,1)*delta.permute(0,2,1)
        return {'prediction':pred,'alpha':alpha,'gain_estimate':self.gain(feat).squeeze(-1),
                'harm_logit':self.harm(feat).squeeze(-1),
                'base_prediction':integrated['null_prediction']}

class FARE(nn.Module):
    def __init__(self,backbone,graph,cfg):
        super().__init__();self.backbone=backbone;self.graph=graph;self.cfg=cfg
        d=int(cfg['model']['hidden_dim']);p=int(cfg['model']['horizon']);r=cfg['retrieval']
        self.base_proj=nn.Linear(backbone.d,d)
        self.support=PropagationSupportEncoder(d,d,p)
        self.deficit=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,1))
        self.retriever=HierarchicalRetriever(d,r['coarse_k'],r['shortlist_k'],r.get('chunk_size',4096))
        self.transfer=PropagationTransport(d,p,cfg.get('traffic',{}).get('max_temporal_lag',3))
        self.ranker=ForecastUtilityRanker(d,p)
        self.selector=ConditionalSetSelector(d,p,r['final_k'],r.get('temperature',.25))
        self.integrator=ForecastIntegrator(d,backbone.d,p)
        self.risk=RiskController(d)
    def train(self,mode=True):
        super().train(mode);self.backbone.train(False);return self

    @staticmethod
    def _empty_selection(pred):
        b,p,n=pred.shape;dev=pred.device
        z=torch.zeros(b,n,1,3,p,device=dev)
        return {'selected_ids':torch.zeros(b,n,1,device=dev,dtype=torch.long),
                'selected_positions':torch.zeros(b,n,1,device=dev,dtype=torch.long),
                'selected_values':z,'selected_valid':z.bool(),'weights':z,
                'null':torch.ones(b,n,p,device=dev),
                'correction':torch.zeros(b,n,p,device=dev),
                'predicted_gain':torch.zeros(b,n,p,device=dev)}
    def predict_with_selection(self,batch,prepared,selection,intervention='full',alpha_override=None):
        integ=self.integrator(batch,prepared['base'],self.backbone,prepared['support'],selection,intervention)
        risk=self.risk(prepared['support'],integ,selection,intervention,alpha_override)
        return {'prediction':risk['prediction'],'base_prediction':prepared['base']['prediction'],
                'null_prediction':integ['null_prediction'],'candidate_prediction':integ['candidate_prediction'],
                'alpha':risk['alpha'],'gain_estimate':risk['gain_estimate'],
                'harm_logit':risk['harm_logit'],'integrated':integ,'selection':selection,
                'support':prepared['support']}
    def candidate_single(self,batch,prepared,positions):
        """Exact no-retraining single-candidate intervention for TRAIN-only utility labels."""
        selected=self.selector(prepared['candidate_pool'],prepared['proposals'],prepared['support'],
                               prepared['rank'],forced_positions=positions,force_nonnull=True)
        return self.predict_with_selection(batch,prepared,selected,alpha_override=1.0)
    def intervene_cached(self,batch,memory,full,mode,purge=0):
        if mode=='full':return full
        if mode in ('no_evidence','no_risk_control','no_hidden_injection','no_forecast_correction'):
            selected=self._empty_selection(full['base_prediction']) if mode=='no_evidence' else full['selection']
            out=self.predict_with_selection(batch,full['prepared'],selected,mode)
            out.update({'prepared':full['prepared'],'candidate_pool':full.get('candidate_pool'),
                        'proposals':full.get('proposals')})
            return out
        return self.forward(batch,memory,intervention=mode,purge=purge)
    def forward(self,batch,memory,intervention='full',purge=0,return_details=True):
        if memory.region_key_cpu is None:memory.attach_graph(self.graph)
        if hasattr(self.backbone,'encode_base'):
            base=self.backbone.encode_base(batch['x'],batch['mask'],batch['tod'],batch['dow'])
        else:base=self.backbone(batch['x'],batch['mask'],batch['tod'],batch['dow'])
        hidden=self.base_proj(base['hidden'])
        sup=self.support(batch['x'],batch['mask'],hidden,base['prediction'],self.graph,
                         neutral=intervention=='no_support',no_direction=intervention=='no_direction',
                         topology=batch.get('topology'))
        sup['forecast_deficit_logits']=self.deficit(sup['embedding']).squeeze(-1)
        sup['forecast_deficit']=sup['forecast_deficit_logits'].sigmoid()
        if intervention=='no_evidence':
            short=None;prop=None;rank=None;selection=self._empty_selection(base['prediction'])
        else:
            mode='state_similarity_only' if intervention=='state_similarity_only' else None
            short=self.retriever(batch['x'],batch['mask'],sup,batch['origin'],memory,self.graph,purge,mode)
            if intervention in ('random_memory','shuffled_memory'):
                lawful=memory.valid_for(batch['origin'],batch['x'].device,purge)
                if intervention=='random_memory':
                    prob=lawful.float();empty=~lawful.any(-1)
                    prob[empty,0]=1
                    ids=torch.multinomial(prob,short['ids'].shape[1]*short['ids'].shape[2],replacement=True).reshape_as(short['ids'])
                    good=lawful.gather(1,ids.reshape(ids.shape[0],-1)).reshape_as(ids)
                else:
                    ids=short['ids'].roll(1,1)
                    good=lawful.gather(1,ids.reshape(ids.shape[0],-1)).reshape_as(ids)
                short['ids']=ids
                short['valid']=good
                short['scores']=torch.zeros_like(short['scores']) # NEVER reuse old candidate scores.
                short['candidate_capability']=memory.fetch_node_keys(ids,ids.device,'capability_cpu')
            trans=intervention if intervention in ('no_direction','no_residual','no_evolution',
                                                  'no_propagation_alignment') else None
            prop=self.transfer(short,memory,self.graph,sup,base['prediction'],trans)
            rank=self.ranker(short,prop,sup)
            if intervention=='matched_random':
                # Same retrieved legal shortlist: randomize SELECTION, not Memory legality.
                # Stateless hashed pseudo-random scores make repeated evaluations comparable.
                ii=short['ids'].long();rr=batch['row_id'].to(ii.device).long()[:,None,None]
                nn=torch.arange(ii.shape[1],device=ii.device)[None,:,None]
                hashed=(ii*1103515245+rr*12345+nn*2654435761)&0x7fffffff
                random_score=hashed.float()/2147483647.0
                rank=dict(rank,score=random_score[...,None].expand_as(rank['score']))
            if intervention=='candidate_prior':
                # Query-independent ranking control; evidence content remains identical.
                prior=prop['deltas'].float().abs().mean((-1,-2))
                rank=dict(rank,score=prior[...,None].expand_as(rank['score']))
            if intervention=='state_similarity_only':
                # Learned Utility ablation: actually rank with coarse similarity only.
                sim=short['scores'].float()[...,None]
                rank=dict(rank,score=sim.expand_as(rank['score']))
            selection=self.selector(short,prop,sup,rank,force_nonnull=False)
        prepared={'base':base,'support':sup,'candidate_pool':short,'proposals':prop,'rank':rank}
        out=self.predict_with_selection(batch,prepared,selection,intervention)
        out['shortlist_ids']=short['ids'] if short is not None else None
        if return_details:out.update({'candidate_pool':short,'proposals':prop,'rank':rank,'prepared':prepared})
        return out
