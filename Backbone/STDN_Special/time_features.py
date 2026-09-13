#!/usr/bin/env python3
"""Compatibility wrapper around the Backbone-wide temporal-feature contract.

STDN_Special no longer owns a separate parser. All models now use
``backbone.time_features``.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone.time_features import (  # noqa: E402,F401
    DATA_CONTRACT_REVISION,
    TIME_FEATURE_REVISION,
    ParsedTimeFeatures,
    audit_cache_time_features,
    audit_resolved_time_features,
    assert_resolved_time_features,
    resolve_time_features,
    time_feature_fingerprint,
)

REVISION = TIME_FEATURE_REVISION


def resolve_time_features_safe(bench, allow_index_fallback: bool = False):
    return resolve_time_features(bench, allow_index_fallback=allow_index_fallback)
