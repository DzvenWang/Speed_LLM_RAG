"""Config -> model keyword arguments for the optional v0.2.3 mechanisms."""
from __future__ import annotations


def mechanism_kwargs(cfg):
    """v0.2.3 optional mechanisms (B dose ladder, A view consistency).

    Absent or disabled sections reproduce the v0.2.2 model exactly.
    """
    m=cfg['model'];kwargs={}
    dose=m.get('dose_ladder') or {}
    if dose.get('enabled',False):
        kwargs.update(dose_levels=tuple(dose.get('levels',(0.25,0.5,0.75,1.0))),
                      dose_margin=float(dose.get('margin',0.0)),
                      dose_temperature=float(dose.get('temperature',0.01)))
    view=m.get('view_consistency') or {}
    if view.get('enabled',False):
        if not dose.get('enabled',False):
            raise RuntimeError('view_consistency feeds the dose policy: enable dose_ladder '
                               '(use levels [1.0] for a consistency-only ablation)')
        kwargs.update(consistency_views=int(view.get('views',3)),
                      consistency_drop=float(view.get('drop_fraction',0.2)),
                      consistency_seed=int(view.get('seed',cfg['experiment'].get('seed',10))))
    return kwargs
