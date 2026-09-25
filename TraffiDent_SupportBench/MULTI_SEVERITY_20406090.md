# 20/40/60/90 missingness extension

This version keeps the historical Train view severity catalogue unchanged while expanding Validation/Test to 20%, 40%, 60%, and 90%. The 90% temporal-outage implementation expands affected-node coverage after the original 75% node-fraction ceiling is reached.

For Net_06 only:

```bash
qsub pbs/build_multiseverity_net06.pbs
```

Equivalent interactive command:

```bash
bash build_multiseverity_net06.sh
```

The audit must pass before building the severity-specific Protocol artifacts.
