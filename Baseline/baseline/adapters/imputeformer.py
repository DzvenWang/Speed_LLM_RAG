from .base import OfficialAdapter
class ImputeFormerAdapter(OfficialAdapter):
    method='ImputeFormer'
    def compatibility_audit(self):
        repo=self.ctx.project_root/'vendor'/self.method
        return {'entrypoint':str(repo/'experiments/run_spatiotemporal_imputation.py'),'exists':(repo/'experiments/run_spatiotemporal_imputation.py').exists(),'formal_protocol':'official ImputeFormer architecture reconstructs history only; observed values protected; completed history enters single Backbone-v1.1.7 frozen PatchSTG reference checkpoint'}
    def prepare_bridge(self):
        return {'cache':str(self.ctx.cache_root),'runner':'baseline/runners/imputeformer_runner.py','protect_observed_values':True,'frozen_backbone_json':str(self.ctx.frozen_backbone_json),'checkpoint_selection':'history-imputation Val only'}
    def run_command(self,phase):
        if phase!='formal': raise RuntimeError('ImputeFormer->PatchSTG v0.5 runner is formal-only.')
        if not self.ctx.frozen_backbone_json.exists(): raise RuntimeError('FROZEN_BACKBONE.json required.')
        return ['python','baseline/runners/imputeformer_runner.py','--stage','formal']
