from .base import OfficialAdapter
class SLPFAdapter(OfficialAdapter):
    method='SLPF'
    def compatibility_audit(self):
        repo=self.ctx.project_root/'vendor'/self.method
        return {'entrypoint':str(repo/'model/run_SLPF.py'),'exists':(repo/'model/run_SLPF.py').exists(),'native_assumption':'one fixed sensed M versus unsensed M-prime partition','formal_protocol':'matched fixed-partial-sensing sub-track; do not claim native dynamic structured-mask support'}
    def prepare_bridge(self):
        return {'cache':str(self.ctx.cache_root),'track':'fixed_partial_sensing_subtrack','needs_easydict':True,'dynamic_main_table_allowed':False}
    def run_command(self, phase):
        raise RuntimeError('SLPF is intentionally gated to a matched fixed-node partial-sensing sub-track. Build that sub-track before invoking official run_SLPF.py.')
