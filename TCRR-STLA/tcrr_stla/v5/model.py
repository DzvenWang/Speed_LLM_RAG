"""TCRR-STLA v5: independent internal mechanism switches.

All teacher, adapter, replay and retrieval evidence is rebuilt under fresh v5 paths.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .counterfactual import StructuredCounterfactual
from .support import PairSupport
from .transport import SupportConditionedTransport
from .controller import ResponseGuidedController

@dataclass(frozen=True)
class MechanismMode:
    cf_enabled:bool=True
    support_enabled:bool=True
    c3_enabled:bool=True
    controller_mode:str='fixed'
    fixed_gate:float=.75

# Two-switch controls remain available for focused fixed-controller comparisons in v5
# comparisons. Canonical three-switch modes are M{C1}{C2}{C3}.
MODES={'M00':(False,False),'M10':(True,False),'M01':(False,True),'M11':(True,True)}
THREE_SWITCH_MODES={
    f'M{int(cf)}{int(support)}{int(c3)}':(cf,support,c3)
    for cf in (False,True) for support in (False,True) for c3 in (False,True)
}
ALL_MODES={**MODES,**THREE_SWITCH_MODES}

def resolve_mechanism_mode(mode,stage,controller_mode=None):
    """Resolve independent C1/C2/C3 switches without silent overrides.

    For canonical Mxyz: C3=0 -> no gate (gate=1), C3=1 -> learned
    correction-scale controller. Fixed 0.75 remains a distinct comparator,
    never conflated with turning C3 off. Legacy Mxy modes preserve prior
    explicit controller behavior for comparable C1/C2 experiments.
    """
    if mode in THREE_SWITCH_MODES:
        cf,support,c3=THREE_SWITCH_MODES[mode]
        expected='learned' if c3 else 'none'
        if controller_mode not in (None,'auto',expected):
            raise ValueError(f'{mode} requires controller_mode={expected}; '
                             f'got {controller_mode!r}. Use legacy Mxy for '
                             'fixed-gate or overridden-controller diagnostics.')
        return cf,support,c3,expected
    if mode in MODES:
        cf,support=MODES[mode]
        controller=('learned' if stage in ('controller','joint') else 'fixed') \
            if controller_mode in (None,'auto') else controller_mode
        if controller not in ('fixed','learned','none'):
            raise ValueError(f'Invalid controller mode {controller}')
        return cf,support,controller=='learned',controller
    raise ValueError(f'Expected one of {tuple(ALL_MODES)}, got {mode}')

class TCRRSTLAV5(nn.Module):
    def __init__(self,stla,cfg,neighbours=None):
        super().__init__()
        self.stla=stla
        cfg5=cfg.get('v5',{})
        latent=int(cfg['model'].get('latent_dim',16));hidden=int(cfg['model'].get('fusion_hidden',64))
        horizon=int(cfg['experiment']['horizon']);bound=float(cfg['model']['max_normalized_correction'])
        self.latent_proj=nn.Sequential(nn.Linear(stla.model_dim,latent),nn.GELU())
        self.c1=StructuredCounterfactual(horizon,latent,hidden,int(cfg5.get('temporal_hidden',16)))
        self.support_encoder=PairSupport(int(cfg5.get('support_hidden',32)))
        self.c2=SupportConditionedTransport(horizon,latent,self.support_encoder.out_dim,hidden,bound,
                         float(cfg5.get('rank_utility_weight',.35)),
                         float(cfg5.get('marginal_rank_weight',.15)))
        self.c3=ResponseGuidedController(latent,hidden,
            default_variant=str(cfg5.get('c3_variant','full')),
            spatial_mix=float(cfg5.get('c3_spatial_mix',.5)),
            horizon_group_size=int(cfg5.get('c3_horizon_group_size',4)))
        if neighbours is None:neighbours=torch.empty(0,dtype=torch.long)
        self.register_buffer('neighbours',neighbours.long())
        self.stage='c12';self.fixed_gate=float(cfg5.get('fixed_gate',.75))
        self.freeze_base()

    def freeze_base(self):
        for p in self.stla.parameters():p.requires_grad_(False)
        self.stla.eval()

    def set_stage(self,stage):
        if stage not in ('c12','controller','joint'):raise ValueError(stage)
        self.stage=stage
        for name,p in self.named_parameters():
            if name.startswith('stla.'):p.requires_grad_(False)
            elif name.startswith('c3.'):p.requires_grad_(stage!='c12')
            else:p.requires_grad_(stage!='controller')

    def train(self,mode=True):
        super().train(mode)
        self.stla.eval()
        return self

    def forward(self,x,mask,tod,dow,candidate_y=None,residual=None,candidate_scores=None,
                candidate_coverage=None,hist_distance=None,candidate_valid=None,
                normal_residual=None,teacher_base=None,candidate_histories=None,
                candidate_ids=None,candidate_source_valid=None,base_prediction=None,
                frozen_context=None,mode='M11',controller_mode=None,
                shuffled_negatives=False,support_control='correct',controller_variant=None):
        if base_prediction is None or frozen_context is None:
            with torch.no_grad():base_prediction,frozen_context=self.stla(x,mask,tod,dow)
        base=base_prediction.float();context=self.latent_proj(frozen_context.float())
        cf_on,support_on,c3_on,resolved_controller=resolve_mechanism_mode(
            mode,self.stage,controller_mode)
        if residual is None or normal_residual is None or candidate_valid is None or candidate_source_valid is None:
            raise ValueError('v5 requires fresh paired residual evidence, valid and source-valid masks')
        if candidate_scores is None or candidate_coverage is None:raise ValueError('Missing retrieval scores')
        cf=residual.float()-normal_residual.float()
        c1=self.c1(normal_residual.float(),cf,base,context,candidate_scores.float(),
                   candidate_coverage.float(),enabled=cf_on,
                   shuffled_negatives=shuffled_negatives)
        # Support must not leak into M00/M10: compute paired features but only use them when switch on.
        paired_support=self.support_encoder(x.float(),mask.float(),candidate_histories,
                                            candidate_source_valid,self.neighbours,control=support_control)
        c2=self.c2(base,context,c1['normal_component'],c1['cf_increment'],c1['cf_reliability'],
                   paired_support,candidate_scores.float(),candidate_coverage.float(),
                   candidate_valid,cf_enabled=cf_on,support_enabled=support_on,
                   cf_utility=c1['cf_incremental_utility'])
        control=self.c3(base,context,c2,mode=resolved_controller,
                        fixed_gate=self.fixed_gate,neighbours=self.neighbours,
                        variant=controller_variant)
        return {**c1,**c2,**control,'base_prediction':base,'query_context':context,
                'scale_choices':self.c3.scales,
                'support_context':paired_support['raw'].mean(1) if support_on else torch.zeros_like(paired_support['raw'].mean(1)),
                'support_pair_raw':paired_support['raw'] if support_on else torch.zeros_like(paired_support['raw']),
                'normal_residual':normal_residual.float(),'failure_residual':cf,
                'candidate_ids':candidate_ids,'residual_bound':self.c2.bound,
                'mechanism_mode':mode,'c1_enabled':cf_on,'c2_enabled':support_on,
                'c3_enabled':c3_on,'controller_mode':resolved_controller,
                'functional_neighbours':self.neighbours,
                'retrieval_prediction':base+c2['proposed_correction']}


    def forward_modes(self,x,mask,tod,dow,*,modes,base_prediction=None,
                      frozen_context=None,controller_mode='auto',
                      shuffled_negatives=False,support_control='correct',
                      controller_variant=None,**evidence):
        """Evaluation-only multi-mode pass with exact shared C1/C2 intermediates.

        Independent, unchanged model.forward() is still used for optimization;
        this API only removes redundant deterministic work on identical queries.
        The output dictionaries have the same keys as independent forward calls.
        No changes to retrieval, candidate shortlist, softmax, or interventions.
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('forward_modes is inference-only; call under torch.no_grad()')
        modes=tuple(modes)
        if not modes or len(modes)!=len(set(modes)):
            raise ValueError('Expected non-empty distinct mode names')
        if base_prediction is None or frozen_context is None:
            base_prediction,frozen_context=self.stla(x,mask,tod,dow)
        base=base_prediction.float()
        context=self.latent_proj(frozen_context.float())
        required=('residual','normal_residual','candidate_valid',
                  'candidate_source_valid','candidate_histories',
                  'candidate_scores','candidate_coverage')
        missing=[name for name in required if evidence.get(name) is None]
        if missing:raise ValueError(f'Missing paired inputs: {missing}')
        normal=evidence['normal_residual'].float()
        cf=evidence['residual'].float()-normal
        scores=evidence['candidate_scores'].float()
        coverage=evidence['candidate_coverage'].float()
        decoded={m:resolve_mechanism_mode(m,self.stage,controller_mode) for m in modes}
        c1_cache={}
        for cf_enabled in set(flag[0] for flag in decoded.values()):
            c1_cache[cf_enabled]=self.c1(normal,cf,base,context,scores,coverage,
                                        enabled=cf_enabled,
                                        shuffled_negatives=shuffled_negatives and cf_enabled)
        paired_support=self.support_encoder(x.float(),mask.float(),
            evidence['candidate_histories'],evidence['candidate_source_valid'],
            self.neighbours,control=support_control)
        support_raw=paired_support['raw']
        zeros_raw=torch.zeros_like(support_raw)
        c2_cache={}
        for cf_enabled,support_enabled,_,_ in decoded.values():
            key=(cf_enabled,support_enabled)
            if key not in c2_cache:
                c1=c1_cache[cf_enabled]
                c2_cache[key]=self.c2(base,context,c1['normal_component'],
                    c1['cf_increment'],c1['cf_reliability'],paired_support,
                    scores,coverage,evidence['candidate_valid'],
                    cf_enabled=cf_enabled,support_enabled=support_enabled,
                    cf_utility=c1['cf_incremental_utility'])
        outputs={}
        for mode,(cf_enabled,support_enabled,c3_enabled,resolved_controller) in decoded.items():
            c1=c1_cache[cf_enabled];c2=c2_cache[(cf_enabled,support_enabled)]
            control=self.c3(base,context,c2,mode=resolved_controller,
                fixed_gate=self.fixed_gate,neighbours=self.neighbours,
                variant=controller_variant)
            outputs[mode]={**c1,**c2,**control,'base_prediction':base,
                'query_context':context,'scale_choices':self.c3.scales,
                'support_context':support_raw.mean(1) if support_enabled else zeros_raw.mean(1),
                'support_pair_raw':support_raw if support_enabled else zeros_raw,
                'normal_residual':normal,'failure_residual':cf,
                'candidate_ids':evidence.get('candidate_ids'),
                'residual_bound':self.c2.bound,'mechanism_mode':mode,
                'c1_enabled':cf_enabled,'c2_enabled':support_enabled,
                'c3_enabled':c3_enabled,'controller_mode':resolved_controller,
                'functional_neighbours':self.neighbours,
                'retrieval_prediction':base+c2['proposed_correction']}
        return outputs
