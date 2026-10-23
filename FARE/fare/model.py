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
        pred0=base.get('support_prediction',base['prediction'])
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
        if mode not in ('no_evidence','no_hidden_injection'):
            # Evidence delta must be relative to the SAME support-only reference.
            hidden=hidden+(pred0-base['prediction'])
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
                        selection['predicted_gain'][...,None],selection.get('predicted_harm',delta.abs())[...,None],
                        selection.get('predicted_sigma',support['demand'])[...,None],support['forecast_deficit'][...,None]],-1)
        alpha=self.net(feat).squeeze(-1).sigmoid()
        if mode=='no_risk_control':alpha=torch.ones_like(alpha)
        if alpha_override is not None:
            # Exact teacher do(g=constant): bypass learned risk and NULL shrinkage.
            alpha=torch.full_like(alpha,float(alpha_override))*(1-selection['null']).clamp(0,1)
        else:
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
        self.vulnerability=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,1))
        self.support_delta=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,1))
        nn.init.zeros_(self.support_delta[-1].weight)
        nn.init.zeros_(self.support_delta[-1].bias)
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
        return {'prediction':risk['prediction'],'support_prediction':prepared['base'].get('support_prediction',prepared['base']['prediction']),'base_prediction':prepared['base']['prediction'],
                'null_prediction':integ['null_prediction'],'candidate_prediction':integ['candidate_prediction'],
                'alpha':risk['alpha'],'gain_estimate':risk['gain_estimate'],
                'harm_logit':risk['harm_logit'],'integrated':integ,'selection':selection,
                'support':prepared['support']}
    def predict_support_only(self,batch):
        base=self.backbone.encode_base(batch['x'],batch['mask'],batch['tod'],batch['dow']) if hasattr(self.backbone,'encode_base') else self.backbone(batch['x'],batch['mask'],batch['tod'],batch['dow'])
        hidden=self.base_proj(base['hidden'])
        sup=self.support(batch['x'],batch['mask'],hidden,base['prediction'],self.graph,topology=batch.get('topology'))
        delta=self.support_delta(sup['embedding']).squeeze(-1).permute(0,2,1)
        return base['prediction']+delta

    def candidate_single(self,batch,prepared,positions):
        """Exact no-retraining single-candidate intervention for TRAIN-only utility labels."""
        selected=self.selector(prepared['candidate_pool'],prepared['proposals'],prepared['support'],
                               prepared['rank'],forced_positions=positions,force_nonnull=True)
        # A forced candidate has zero NULL mass wherever its evidence is valid.
        # When no stream is valid, the only admissible action is NULL.
        valid=selected['selected_valid'].any((2,3))
        selected['null']=torch.where(valid,torch.zeros_like(selected['null']),torch.ones_like(selected['null']))
        # Exactly one valid candidate has probability one for its active horizon;
        # there is no remaining learned NULL or ranker-mass attenuation.
        selected['weights']=selected['selected_valid'].float()
        count=selected['weights'].sum((2,3),keepdim=False).clamp_min(1)
        selected['weights']=selected['weights']/count[:,:,None,None,:]
        selected['correction']=(selected['weights']*selected['selected_values']).sum((2,3))
        return self.predict_with_selection(batch,prepared,selected,alpha_override=1.0)
    def intervene_cached(self,batch,memory,full,mode,purge=0):
        if mode=='full':return full
        if mode in ('no_evidence','support_only','no_risk_control','no_hidden_injection','no_forecast_correction','global_selection','no_failure','shuffled_sensing','temporal_mean','fixed_gate','audit_shared_horizon','audit_static_prior','audit_same_pool_similarity','audit_same_pool_random'):
            if mode=='temporal_mean':
                selected=dict(full['selection'])
                temporal=selected['selected_values'].clone()
                mean=(temporal*selected['selected_valid']).sum(-1,keepdim=True)/selected['selected_valid'].sum(-1,keepdim=True).clamp_min(1)
                selected['selected_values']=mean.expand_as(temporal)
            elif mode=='global_selection':
                prep=full['prepared']
                selected=self.selector(prep['candidate_pool'],prep['proposals'],prep['support'],prep['rank'],global_selection=True)
            elif mode == 'audit_shared_horizon':
                # Strict horizon-INDEPENDENT evidence selection: identical candidate,
                # branch and NULL mass at every horizon.  Require each retained
                # stream to be available at every horizon; unavailable streams
                # are assigned zero mass rather than using future labels.
                selected=dict(full['selection'])
                stream_valid=selected['selected_valid'].all(-1)
                shared=selected['weights'].float().mean(-1)*stream_valid.float()
                total=shared.sum((2,3),keepdim=True).clamp_min(1.0)
                shared=shared/total
                selected['weights']=shared[...,None].expand_as(selected['weights'])
                selected['null']=(1-shared.sum((2,3))).clamp(0,1)[...,None].expand_as(selected['null'])
                selected['correction']=(selected['weights']*selected['selected_values']).sum((2,3))
                for field in ('predicted_gain','predicted_harm','predicted_sigma'):
                    if field in selected:
                        selected[field]=selected[field].mean(-1,keepdim=True).expand_as(selected[field])
            elif mode in ('audit_static_prior','audit_same_pool_similarity','audit_same_pool_random'):
                # Only the policy changes; shortlist IDs, retrieval legality,
                # transport, and all trained weights are held fixed.
                prep=full['prepared'];short=prep['candidate_pool'];rank=prep['rank']
                if mode=='audit_static_prior':
                    # Historical OOF error & coverage summaries stored on each
                    # episode/node.  No current-query-conditioned proposal.
                    cap=short['candidate_capability'].float()
                    policy=cap[...,7]-0.1*cap[...,3].abs()
                elif mode=='audit_same_pool_similarity':
                    policy=short['scores'].float()
                else:
                    ids=short['ids'].long();rr=batch['row_id'].to(ids.device).long()[:,None,None]
                    node=torch.arange(ids.shape[1],device=ids.device)[None,:,None]
                    hashed=(ids*1103515245+rr*12345+node*2654435761)&0x7fffffff
                    policy=hashed.float()/2147483647.0
                overridden=dict(rank,score=policy[...,None].expand_as(rank['score']))
                selected=self.selector(short,prep['proposals'],prep['support'],overridden)
            elif mode in ('no_failure','shuffled_sensing'):
                prep=full['prepared'];prop=dict(prep['proposals'])
                z=prop['deltas'].clone();av=prop['available'].clone()
                if mode=='no_failure':
                    z[...,2,:]=0;av[...,2,:]=False
                else:
                    z[...,2,:]=z[...,2,:].roll(1,2)
                    av[...,2,:]=av[...,2,:].roll(1,2)
                prop['deltas']=z;prop['available']=av
                rank=self.ranker(prep['candidate_pool'],prop,prep['support'])
                selected=self.selector(prep['candidate_pool'],prop,prep['support'],rank)
            else:
                selected=self._empty_selection(full['base_prediction']) if mode in ('no_evidence','support_only') else full['selection']
            out=self.predict_with_selection(batch,full['prepared'],selected,'no_evidence' if mode=='support_only' else ('no_risk_control' if mode=='fixed_gate' else mode))
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
        if intervention in ('no_evidence','support_only','backbone_only'):
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
                                                  'no_propagation_alignment','ordinary_residual') else None
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
        # Query-only support branch is the common null/reference prediction.
        base['support_prediction']=base['prediction']+self.support_delta(sup['embedding']).squeeze(-1).permute(0,2,1)
        sup['vulnerability_logits']=self.vulnerability(sup['embedding']).squeeze(-1)
        prepared={'base':base,'support':sup,'candidate_pool':short,'proposals':prop,'rank':rank}
        out=self.predict_with_selection(batch,prepared,selection,'no_evidence' if intervention in ('support_only','backbone_only') else intervention)
        if intervention=='backbone_only':
            out['prediction']=base['prediction']
        out['support_prediction']=base['support_prediction']
        out['shortlist_ids']=short['ids'] if short is not None else None
        if return_details:out.update({'candidate_pool':short,'proposals':prop,'rank':rank,'prepared':prepared})
        return out
