# Backbone Core-5 experiment protocol

Primary Train/Val/Test families:

1. entry_random
2. sensor_outage
3. temporal_outage
4. regional_outage
5. directional_outage

Training/validation use s40. Final evaluation uses s20/s40/s60/s90 with the same five families. Predictive regional/directional outages are intentionally excluded from primary metrics and can be studied later as a separate challenge track.

Expected balanced sizes: development train 12,000 (2,400/family), development/formal val 5,000 (1,000/family), formal train 50,000 (10,000/family), evaluation 5,000 per severity (1,000/family).
