#!/usr/bin/env python3
from __future__ import annotations
import tempfile
from pathlib import Path
import yaml

MATCHED_CONTROL_NATIVE = {
    'RAFT-PatchSTG': 'RAFT',
    'SARAF-PatchSTG': 'SARAF',
    'RAST-PatchSTG': 'RAST',
}


def resolve(cfg: dict, method: str):
    methods = cfg.get('methods', {})
    mc = methods.get(method)
    alias = MATCHED_CONTROL_NATIVE.get(method)
    if mc is None and alias is not None:
        native = methods.get(alias)
        if native is None:
            return None, alias, False
        mc = dict(native)
        mc['requires_frozen_backbone_for_final_forecast'] = True
    return mc, alias, bool(mc and mc.get('requires_frozen_backbone_for_final_forecast', False))


def main():
    # Simulates the exact failure condition from the PBS logs: native methods
    # exist while RAFT-PatchSTG/SARAF-PatchSTG have not yet been added to YAML.
    cfg = {
        'methods': {
            'RAFT': {'enabled': True, 'requires_frozen_backbone_for_final_forecast': False},
            'SARAF': {'enabled': True, 'requires_frozen_backbone_for_final_forecast': False},
            'RAST': {'enabled': True, 'requires_frozen_backbone_for_final_forecast': False},
        }
    }
    for method, native in MATCHED_CONTROL_NATIVE.items():
        mc, alias, need_patch = resolve(cfg, method)
        assert mc is not None, method
        assert alias == native
        assert need_patch is True

    # Explicit entries must still work and remain frozen-backbone controls.
    cfg['methods']['RAFT-PatchSTG'] = {
        'enabled': True,
        'requires_frozen_backbone_for_final_forecast': True,
        'formal_seeds': [10],
    }
    mc, alias, need_patch = resolve(cfg, 'RAFT-PatchSTG')
    assert mc['formal_seeds'] == [10]
    assert alias == 'RAFT'
    assert need_patch is True

    print('RETRIEVAL-PATCHSTG REGISTRATION SMOKE PASS')


if __name__ == '__main__':
    main()
