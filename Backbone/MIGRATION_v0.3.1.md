# Migration to Backbone v0.3.1

Protocol v1.0.0 is unchanged. Do not rebuild the frozen Development Protocol.

v0.3.1 changes only experiment governance:
- cache manifest records exact Protocol split view hashes;
- every run writes `RUN_MANIFEST.json`;
- `best.pt` and `summary.json` carry the same run identity;
- screening/formal summaries reject untracked or mixed-identity results;
- `FROZEN_BACKBONE.json` stores the formal run identity;
- Backbone sizes are assertions on Protocol, never local sampling controls.

Because project version/config fingerprint changed, rebuild the Backbone consumer cache and rerun screening. Do not reuse v0.3.0 Backbone cache/results.
