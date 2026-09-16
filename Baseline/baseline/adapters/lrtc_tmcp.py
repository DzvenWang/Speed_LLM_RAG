from __future__ import annotations
from .base import OfficialAdapter


class LRTCTMCPAdapter(OfficialAdapter):
    method = 'LRTC-TMCP'

    def compatibility_audit(self):
        return {
            'official_source_unmodified': True,
            'official_equations_externalized': True,
            'strict_h12_only': True,
            'cross_query_information': False,
            'future_information': False,
            'observed_value_protection': True,
            'downstream_forecaster': 'same frozen PatchSTG reference as ImputeFormer',
        }

    def prepare_bridge(self):
        return {
            'runner': 'baseline/runners/lrtc_tmcp_runner.py',
            'compatibility_core': 'baseline/compat/lrtc_tmcp.py',
            'official_notebook': 'vendor/LRTC-TMCP/lrtc-tmcp-demo.ipynb',
        }

    def run_command(self, phase: str):
        return f'python baseline/runners/lrtc_tmcp_runner.py --stage {phase} --seed 10'
