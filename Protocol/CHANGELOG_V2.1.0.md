# Protocol v2.1.0

- Primary family set changed to the five SupportBench general/core families: entry_random, sensor_outage, temporal_outage, regional_outage, directional_outage.
- Predictive regional/directional outages are excluded from primary Train/Val/Test and remain a separate SupportBench challenge track.
- Protocol names changed to `protocol_core5_s20/s40/s60/s90`.
- Paired group changed to `TraffiDent_Net_06_core5_multiseverity_v1` to prevent reuse of old manifests.
- Development validation uses `all_eligible` rather than the support-challenge pool, because the challenge pool does not contain entry_random/sensor_outage.
- Evaluation remains direct/unlocked.
