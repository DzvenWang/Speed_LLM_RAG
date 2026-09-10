from .base import OfficialAdapter


class RAFTAdapter(OfficialAdapter):
    method='RAFT'

    def compatibility_audit(self):
        repo=self.ctx.project_root/'vendor'/self.method
        paths=[repo/'run.py',repo/'models/RAFT.py',repo/'layers/Retrieval.py']
        return {
            'entrypoints':[str(p) for p in paths],
            'exists':all(p.exists() for p in paths),
            'native_assumption':'continuous multivariate forecasting with historical retrieval memory',
            'stock_loop_issue':'official experiment path builds/evaluates Test during training',
            'formal_protocol':'official RAFT Model/RetrievalTool via external Val-only runner; Train-only memory; Protocol-time overlap exclusion replaces contiguous-row self mask',
            'main_table_allowed':True,
        }

    def prepare_bridge(self):
        return {
            'cache':str(self.ctx.cache_root),
            'x':'*_x_obs_norm.npy',
            'y':'*_y_norm.npy',
            'retrieval_memory':'train_only_full_history',
            'query_signal':'incomplete_history',
            'prediction_contract':'raw_speed_Mx12xN',
            'runner':'baseline/runners/raft_runner.py',
            'stock_run_py_allowed_for_formal':False,
            'test_access_before_evaluation':False,
            'requires_retrieval_leakage_audit':True,
        }

    def run_command(self, phase):
        if phase not in ('development','formal'):
            raise RuntimeError('RAFT runner is permitted only for development/formal before evaluation authorization.')
        return [str(self.ctx.project_root/'baseline'/'runners'/'raft_runner.py'),'--stage',phase]
