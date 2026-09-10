from .base import OfficialAdapter


class GinARAdapter(OfficialAdapter):
    method = 'GinAR'

    def compatibility_audit(self):
        repo = self.ctx.project_root / 'vendor' / self.method
        paths = [repo / 'main_ginar.py', repo / 'model1/ginar_arch.py', repo / 'model1/ginar_cell.py']
        return {
            'entrypoints': [str(p) for p in paths],
            'exists': all(p.exists() for p in paths),
            'native_assumption': 'variable-missing multivariate forecasting with zeroed missing variables; official examples use 12 history and 12 future steps',
            'formal_protocol': 'official GinAR model1 architecture under an external TraffiDent runner; dynamic Protocol masks are used exactly; checkpoint selection is Val-only',
            'main_table_allowed': True,
        }

    def prepare_bridge(self):
        return {
            'cache': str(self.ctx.cache_root),
            'x': '*_x_obs_norm.npy',
            'y': '*_y_norm.npy',
            'time_features': ['*_tod_hist.npy', '*_dow_hist.npy'],
            'adjacency': 'adjacency.npy',
            'prediction_contract': 'raw_speed_Mx12xN',
            'runner': 'baseline/runners/ginar_runner.py',
            'test_access_before_evaluation': False,
        }

    def run_command(self, phase):
        if phase not in ('development', 'formal'):
            raise RuntimeError('GinAR runner is permitted only for development/formal before evaluation authorization.')
        return [str(self.ctx.project_root / 'baseline' / 'runners' / 'ginar_runner.py'), '--stage', phase]
