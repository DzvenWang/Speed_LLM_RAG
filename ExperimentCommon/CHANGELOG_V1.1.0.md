# v1.1.0

- Bind training caches to Protocol v2.0.1 `protocol_s40` development/formal.
- Enforce Formal 50,000 rows, 50,000 unique groups, 10,000 rows per main family.
- Add four paired evaluation contracts: s20/s40/s60/s90.
- Enforce Protocol evaluation access gate before Test cache creation/validation.
- Replace legacy stage-only cache paths with dataset/protocol/severity/stage/full-stage-hash identity.
- Add local/Gadi build logging and cache status/preflight scripts.
- Keep tensor, metrics, time-feature, and prediction contracts unchanged.
