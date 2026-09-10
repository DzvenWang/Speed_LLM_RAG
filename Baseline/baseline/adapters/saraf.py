from .base import OfficialAdapter


class SARAFAdapter(OfficialAdapter):
    method = 'SARAF'

    def compatibility_audit(self):
        repo = self.ctx.project_root / 'vendor' / self.method
        paths = [repo / 'run.py', repo / 'models/SARAF.py', repo / 'layers/Retrieval.py']
        return {
            'entrypoints': [str(p) for p in paths],
            'exists': all(p.exists() for p in paths),
            'native_assumption': 'continuous multivariate forecasting with retrieval memory built from training windows',
            'stock_loop_issue': 'official Exp_Long_Term_Forecast builds Test retrieval and reports Test during training',
            'formal_protocol': 'official SARAF Model/RetrievalTool via external Val-only runner; Train-only memory; no Test dataset is instantiated before evaluation authorization',
            'main_table_allowed': True,
        }

    def prepare_bridge(self):
        return {
            'cache': str(self.ctx.cache_root),
            'x': '*_x_obs_norm.npy',
            'y': '*_y_norm.npy',
            'retrieval_memory': 'train_only',
            'query_split': 'val',
            'time_marks': 'derived from frozen benchmark time index and Protocol window_start',
            'prediction_contract': 'raw_speed_Mx12xN',
            'runner': 'baseline/runners/saraf_runner.py',
            'stock_run_py_allowed_for_formal': False,
            'test_access_before_evaluation': False,
        }

    def run_command(self, phase):
        if phase not in ('development', 'formal'):
            raise RuntimeError('SARAF runner is permitted only for development/formal before evaluation authorization.')
        return [str(self.ctx.project_root / 'baseline' / 'runners' / 'saraf_runner.py'), '--stage', phase]
