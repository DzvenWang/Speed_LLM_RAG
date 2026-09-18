from __future__ import annotations

METHODS = {
    'ImputeFormer': {
        'category': 'reconstruction_first', 'table_group': 'main', 'venue': 'KDD 2024',
        'repo': 'https://github.com/tongnie/ImputeFormer.git', 'commit': '1c3868cf499951ac43b4fc7280593fbb745f58c7',
        'source_policy': 'exact', 'official_entrypoints': ['experiments/run_spatiotemporal_imputation.py'],
        'can_prepare_before_backbone_freeze': True, 'final_forecast_requires_frozen_backbone': True,
        'protocol': 'impute_history_then_same_frozen_backbone', 'execution_state': 'FORMAL_RUNNER_READY',
        'question': 'Is reconstruction-first sufficient before forecasting?',
        'compatibility_note': 'Official imputation architecture is retained; completed histories enter the same frozen PatchSTG reference used by Ours.',
    },
    'LRTC-TMCP': {
        'category': 'reconstruction_first_tensor_completion', 'table_group': 'main', 'venue': 'TRC 2024',
        'repo': 'https://github.com/peterchen96/LRTC-TMCP.git', 'commit': None, 'source_policy': 'resolve_once',
        'official_entrypoints': ['lrtc-tmcp-demo.ipynb', 'utils.py'],
        'can_prepare_before_backbone_freeze': True, 'final_forecast_requires_frozen_backbone': True,
        'protocol': 'causal_multiday_query_local_lrtc_tmcp_then_same_frozen_backbone',
        'execution_state': 'PRIMARY_SMOKE_REQUIRED',
        'question': 'Can recent TRC low-rank tensor completion recover enough history for accurate forecasting?',
        'compatibility_note': (
            'The official TMCP update equations are preserved in an external batched compatibility implementation. '
            'Each query uses 28 strictly earlier day-aligned H12 slices plus the current incomplete H12, forming a '
            'nodes x causal-days x query-slots tensor. No P12 future or hidden current value is exposed. '
            'Observed current values are protected exactly and the completed H12 is forecast by the same frozen PatchSTG reference as ImputeFormer.'
        ),
    },
    'SPRINT': {
        'category': 'traffic_specific_historical_reconstruction',
        'table_group': 'secondary',
        'venue': 'TRC 2025',
        'repo': 'https://github.com/ashishbhaskar/SPRINT.git', 'commit': None, 'source_policy': 'resolve_once',
        'official_entrypoints': ['README.md', 'pyproject.toml', 'src'],
        'can_prepare_before_backbone_freeze': True, 'final_forecast_requires_frozen_backbone': True,
        'protocol': 'causal_historical_day_context_then_same_frozen_backbone',
        'execution_state': 'CAUSAL_HISTORY_DEV_PENDING',
        'question': 'How far can a traffic-specific reconstruction method go when given its native recurring-day context?',
        'compatibility_note': (
            'SPRINT is an imputer, not a forecaster. The adapter supplies only earlier calendar days at the same '
            'clock-aligned H12 segment plus the current incomplete H12 query, restores current observed values exactly, '
            'and then uses the same frozen PatchSTG. Because SPRINT receives additional causal historical-day context, '
            'it is reported as a specialized reconstruction comparator rather than a strict H12 input-matched main-table baseline.'
        ),
    },
    'GinAR': {
        'category': 'missing_aware_forecasting', 'table_group': 'main', 'venue': 'KDD 2024',
        'repo': 'https://github.com/GestaltCogTeam/GinAR.git', 'commit': None, 'source_policy': 'resolve_once',
        'official_entrypoints': ['main_ginar.py', 'model1/ginar_arch.py', 'model1/ginar_cell.py'],
        'can_prepare_before_backbone_freeze': True, 'final_forecast_requires_frozen_backbone': False,
        'protocol': 'external_dynamic_view_runner_using_official_ginar_architecture', 'execution_state': 'FORMAL_STABILITY_PROBE_REQUIRED',
        'question': 'Is end-to-end variable-missing forecasting sufficient without external support?',
        'compatibility_note': 'Uses the pinned official GinAR architecture with TraffiDent dynamic-view tensors, 12->12 forecasting, graph supports from the frozen adjacency, and Val-only checkpoint selection.',
    },
    'RAFT': {
        'category': 'generic_retrieval', 'table_group': 'main', 'venue': 'ICML 2025',
        'repo': 'https://github.com/archon159/RAFT.git', 'commit': 'bfe2320f5d719d1e2a024330de3410b370bbb2a9', 'source_policy': 'exact',
        'official_entrypoints': ['run.py', 'models/RAFT.py', 'layers/Retrieval.py'],
        'can_prepare_before_backbone_freeze': True, 'final_forecast_requires_frozen_backbone': False,
        'protocol': 'external_dynamic_window_adapter_train_only_memory', 'execution_state': 'NO_LEAKAGE_SMOKE',
        'question': 'Is generic similarity-based historical retrieval sufficient?',
        'compatibility_note': 'Official RAFT architecture is reused through a Val-only external runner with Train-only retrieval memory.',
    },
    'RAST': {
        'category': 'traffic_rag', 'table_group': 'main', 'venue': 'AAAI 2026',
        'repo': 'https://github.com/RWLinno/RAST.git', 'commit': '4b48a47a002dd8d4151afe47be8f436f0c028a67', 'source_policy': 'exact',
        'official_entrypoints': ['experiments/train.py', 'src/arch/rast_arch.py', 'src/scripts/main/RAST_PEMS03.py'],
        'can_prepare_before_backbone_freeze': True, 'final_forecast_requires_frozen_backbone': False,
        'protocol': 'traffic_rag_native_external_dynamic_view_runner_train_only_retrieval', 'execution_state': 'FORMAL_RUNNER_READY',
        'question': 'Is traffic-specific spatiotemporal retrieval sufficient?',
        'compatibility_note': 'Closest traffic-RAG baseline; a separate RAST-PatchSTG matched-backbone adaptation remains appendix-only.',
    },
    'SARAF': {
        'category': 'retrieval_reliability', 'table_group': 'main', 'venue': 'KDD 2026',
        'repo': 'https://github.com/ShiqiaoZhou/SARAF.git', 'commit': None, 'source_policy': 'resolve_once',
        'official_entrypoints': ['run.py', 'models/SARAF.py', 'layers/Retrieval.py'],
        'can_prepare_before_backbone_freeze': True, 'final_forecast_requires_frozen_backbone': False,
        'protocol': 'external_val_only_stationarity_aware_retrieval_train_only_memory', 'execution_state': 'DEV_RUNNER_READY',
        'question': 'Is stationarity-aware robust retrieval sufficient without query-specific predictive-support reasoning?',
        'compatibility_note': 'Uses the official SARAF Model/RetrievalTool while preserving Train-only memory and Val-only selection.',
    },
    'LEAF': {
        'category': 'llm_enhanced_traffic', 'table_group': 'secondary', 'venue': 'ACL Findings 2025',
        'repo': 'https://github.com/YushengZhao/LEAF.git', 'commit': '650b9e915fd4cc213387e279e4e529acbe587b85', 'source_policy': 'exact',
        'official_entrypoints': ['main.py', 'dataset.py'], 'can_prepare_before_backbone_freeze': True,
        'final_forecast_requires_frozen_backbone': False, 'protocol': 'official_dual_branch_plus_llm_selector_reference',
        'execution_state': 'SHARED_TFP_BRANCH_RUNNER',
        'question': 'Does an existing traffic LLM selector explain the value of predictive-support reasoning?',
        'compatibility_note': 'Framework-level LLM reference with local Qwen2.5-1.5B selector adaptation.',
    },
    'ReMFT': {
        'category': 'llm_traffic_forecasting', 'table_group': 'main', 'venue': 'TRC 2026',
        'repo': 'https://github.com/IansSUn/ReMFT.git', 'commit': 'af61bef8ec872ba7b12e78675a39aeb0a20088a6', 'source_policy': 'exact',
        'official_entrypoints': ['run.ipynb'], 'can_prepare_before_backbone_freeze': True,
        'final_forecast_requires_frozen_backbone': False,
        'protocol': 'official_notebook_architecture_external_12step_val_only_runner', 'execution_state': 'RUNNER_READY_WITH_CHECKPOINT_CONSISTENT_BERT_DEPTH',
        'question': 'Does direct LLM-based traffic forecasting explain the gains of using an LLM specifically as a predictive-support reasoner?',
        'compatibility_note': 'Official ReMFT notebook architecture is loaded without executing stock data/train/test cells. History is adapted from 96 to benchmark H12; the frozen BERT runtime is aligned to the local pretrained checkpoint depth and then hard-validated; checkpoint selection is Val-only.',
    },
    'SLPF': {
        'category': 'partial_sensing', 'table_group': 'secondary', 'venue': 'TMLR 2025',
        'repo': 'https://github.com/zbliu98/SLPF.git', 'commit': '8f135a96d57651dc19564bd3b94508c1f92c04fb', 'source_policy': 'exact',
        'official_entrypoints': ['model/run_SLPF.py', 'lib/dataloader.py'], 'can_prepare_before_backbone_freeze': True,
        'final_forecast_requires_frozen_backbone': False, 'protocol': 'matched_fixed_partial_sensing_subtrack', 'execution_state': 'SUBTRACK_ONLY',
        'question': 'How does the method behave under fixed sensed/unsensed-node partial sensing?',
        'compatibility_note': 'Official SLPF assumes one fixed sensed/unsensed node partition and is not claimed as a native dynamic structured-mask main-table method.',
    },
    'MoGERNN': {
        'category': 'unobserved_location_forecasting', 'table_group': 'secondary', 'venue': 'TRC 2025',
        'repo': 'https://github.com/ZJU-TSELab/MoGERNN.git', 'commit': None, 'source_policy': 'resolve_once',
        'official_entrypoints': ['main.py', 'args.py', 'model/MoGERNN.py'], 'can_prepare_before_backbone_freeze': True,
        'final_forecast_requires_frozen_backbone': False,
        'protocol': 'official_mogernn_external_dynamic_view_batch1_runner', 'execution_state': 'DEV_AND_FORMAL_RUNNER_READY',
        'question': 'Can inductive unobserved-location forecasting handle incomplete sensing without external predictive support?',
        'compatibility_note': 'MoGERNN is natively designed for partially observed sensor networks. The external runner uses batch_size=1 so the official zero-based unknown-node detector sees each sample-specific TraffiDent mask independently; this adaptation is reported as a partial-sensing sub-track, not a native dynamic-mask main-table method.',
    },
}

MAIN_ORDER = ['ImputeFormer', 'LRTC-TMCP', 'GinAR', 'RAFT', 'RAST', 'SARAF', 'ReMFT']
SECONDARY_ORDER = ['SPRINT', 'SLPF', 'MoGERNN', 'LEAF']
ORDER = MAIN_ORDER + SECONDARY_ORDER

OPTIONAL_APPENDIX = {
    'RAST-PatchSTG': {'category': 'matched_backbone_control', 'status': 'secondary_controlled_experiment'},
    'LEAF-Qwen1.5B': {'category': 'llm_selector_adaptation', 'status': 'secondary_experiment'},
    'BiTGraph': {'category': 'missing_aware_forecasting', 'status': 'literature_or_appendix_only'},
    'PFRP': {'category': 'retrieval_forecasting', 'status': 'literature_or_appendix_only'},
}
