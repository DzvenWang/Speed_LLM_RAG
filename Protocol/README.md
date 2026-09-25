# Protocol v2.1.0 — Paired Multi-Severity Protocol

This project freezes the shared TraffiDent-Net-06 experiment protocol used by Backbone, Baseline and PreSuRe.

## Design

The current TraffiDent-SupportBench v1.2.6 Train catalogue is intentionally not a full `20/40/60/90 × five-family` grid. Therefore Protocol v2 does **not** invent missing Train sensing views.

The scientifically valid default is:

- `protocol_core5_s40`: Development + Formal + Evaluation. This is the primary training protocol.
- `protocol_core5_s20`: Evaluation only.
- `protocol_core5_s60`: Evaluation only.
- `protocol_core5_s90`: Evaluation only.
- All four Evaluation/Test protocols use the **same 5,000 base query/family pairs**. Only severity changes.
- Test manifests are generated and frozen now, but evaluation is directly readable once the frozen manifest/cache is built.

Main five families:

- `entry_random`
- `sensor_outage`
- `temporal_outage`
- `regional_outage`
- `directional_outage`

## Portable paths

No Windows or GADI path is hard-coded. If the repository layout is:

```text
Speed_LLM_RAG/
  Protocol/
  TraffiDent_SupportBench/
  data/
  artifacts/
```

no environment variable is required. If the project is stored in a non-standard layout, set `SPEED_LLM_RAG_ROOT`.

PowerShell:

```powershell
$env:SPEED_LLM_RAG_ROOT = "C:\path\to\Speed_LLM_RAG"
```

bash/GADI:

```bash
export SPEED_LLM_RAG_ROOT=/path/to/Speed_LLM_RAG
```

## Rebuild everything

From `Speed_LLM_RAG/Protocol`:

```powershell
conda activate TFP
python scripts\build_multiseverity_protocols.py --force
```

The same command works on Linux/GADI with `/` path separators:

```bash
python scripts/build_multiseverity_protocols.py --force
```

The wrapper:

1. preflights all four severity configs;
2. reports actual SupportBench Train-grid coverage;
3. rebuilds `protocol_core5_s40/development`;
4. rebuilds `protocol_core5_s40/formal`;
5. creates one frozen paired Test assignment from the 40% reference severity;
6. freezes `protocol_core5_s20/s40/s60/s90/evaluation` against exactly the same base pairs;
7. validates cross-severity pairing;
8. leaves the paired Test manifest frozen and directly accessible to validated consumers.

Logs are written to:

```text
Speed_LLM_RAG/Protocol_logs/ProtocolMultiSeverity.TraffiDent_Net_06.YYYYMMDD_HHMMSS.log
```

## Artifact layout

```text
Protocol_artifacts/TraffiDent_Net_06/
  _paired/
    TraffiDent_Net_06_core5_multiseverity_v1/
      evaluation_pairs.csv
      PAIRING_FROZEN.json
  protocol_core5_s20/
    evaluation/
  protocol_core5_s40/
    development/
    formal/
    evaluation/
  protocol_core5_s60/
    evaluation/
  protocol_core5_s90/
    evaluation/
```

## Test governance

Protocol v2.1.0 retains the v2.0.2 direct-access policy and removes the `EVALUATION_ACCESS_GRANTED.json` lock. A frozen evaluation manifest can be consumed directly. Integrity protection is still enforced through `FROZEN.json`, `stage_manifest.json`, stage hashes, view hashes, paired manifests, and downstream cache/contract validation.

The legacy `grant_evaluation_access.py`, `revoke_evaluation_access.py`, and `authorize_evaluation.py` entry points are retained only as no-op compatibility commands.

## Validation

```powershell
python scripts\validate_paired_evaluation.py
python scripts\status_all.py
```

## If severity-specific training is needed later

Do not make Protocol synthesize missing Train views. First expand and rebuild the SupportBench Train catalogue so every requested severity has all five training families. Then enable Development/Formal for those severity configs.


## Core-5 Formal balance

`protocol_core5_s40/formal` uses exactly 50,000 training views: 10,000 per primary family, with globally unique `view_group_id`.
