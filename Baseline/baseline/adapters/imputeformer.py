from .base import OfficialAdapter
class ImputeFormerAdapter(OfficialAdapter):
    method='ImputeFormer'
    def compatibility_audit(self):
        repo=self.ctx.project_root/'vendor'/self.method
        return {'entrypoint':str(repo/'experiments/run_spatiotemporal_imputation.py'),'exists':(repo/'experiments/run_spatiotemporal_imputation.py').exists(),'formal_protocol':'reconstruct only artificially missing history; observed values remain protected; feed completed history to the same frozen numerical backbone'}
    def prepare_bridge(self):
        return {'cache':str(self.ctx.cache_root),'x_observed':'*_x_obs_norm.npy','target_complete':'*_x_full_norm.npy','mask':'*_history_observed.npy','protect_observed_values':True,'frozen_backbone_json':str(self.ctx.frozen_backbone_json)}
    def run_command(self, phase):
        if phase in ('forecast','formal') and not self.ctx.frozen_backbone_json.exists(): raise RuntimeError('ImputeFormer final forecasting requires FROZEN_BACKBONE.json.')
        raise RuntimeError('ImputeFormer official TSL data module requires a dedicated external dataset bridge; architecture source remains immutable.')
