# Core-5 sensing protocol (v2.1.0)

The primary training/validation/evaluation benchmark uses exactly five general sensing families:

1. `entry_random`
2. `sensor_outage`
3. `temporal_outage`
4. `regional_outage`
5. `directional_outage`

`predictive_regional_outage` and `predictive_directional_outage` remain available in SupportBench as a separate predictive-support challenge track, but are excluded from the primary Backbone/PreSuRe protocol.

Primary training and validation use severity 0.40. Final evaluation uses severities 0.20/0.40/0.60/0.90 with the same five families. Family budgets remain balanced: development train 2,400/family, development/formal val 1,000/family, formal train 10,000/family, and evaluation 1,000/family at each severity.

The protocol name is `protocol_core5_sXX` and paired evaluation group is `TraffiDent_Net_06_core5_multiseverity_v1`, so old predictive-family artifacts are not reused accidentally.
