from __future__ import annotations
from pathlib import Path
from .registry import METHODS


def audit_repo_contract(root, method):
    root=Path(root); spec=METHODS[method]; repo=root/'vendor'/method
    checks=[]
    for rel in spec['official_entrypoints']:
        p=repo/rel; checks.append({'path':rel,'exists':p.exists()})
    return {'method':method,'repo_exists':repo.exists(),'entrypoints':checks,'all_entrypoints_present':all(c['exists'] for c in checks) if repo.exists() else False,**{k:v for k,v in spec.items() if k not in ('repo','official_entrypoints')}}
