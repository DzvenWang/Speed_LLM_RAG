# GADI deployment

The project contains no fixed GADI path.

Recommended layout:

```text
<root>/Speed_LLM_RAG/
  Protocol/
  TraffiDent_SupportBench/
  data/
  artifacts/
```

From `Protocol/`, direct execution works without setting a root because the parent directory is inferred:

```bash
conda activate TFP
python scripts/build_multiseverity_protocols.py --force
```

If the directory layout differs, set:

```bash
export SPEED_LLM_RAG_ROOT=/your/project/root/Speed_LLM_RAG
```

PBS templates are optional. Replace `PROJECT_CODE` in PBS resource directives for your allocation; runtime project/data paths are inferred and are not hard-coded.
