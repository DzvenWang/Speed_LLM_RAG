# Six backbone-independent experiment tracks

Implementation revision: `20260913_six_model_tracks_r1` (project version remains `0.6.0` so the report/contract identity is unchanged).

## Scientific placement

- Main dynamic-mask track: GinAR, RAFT, RAST-native, SARAF.
- LLM-specific secondary track: LEAF.
- Fixed partial-sensing secondary track: SLPF.
- LEAF and SLPF are not inserted into the dynamic-mask main table as if their original assumptions were identical.

## Shared data contract

All runners consume the canonical direct cache under:

`/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCache/TraffiDent_Net_06/{development,formal}`

No consumer-local `_cache` alias is used.

## Recommended launch sequence

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
conda activate TFP

python scripts/validate_cache.py --stage development
python scripts/validate_cache.py --stage formal
python scripts/verify_official_sources.py
python scripts/retrieval_leakage_audit.py --stage development
python scripts/retrieval_leakage_audit.py --stage formal

bash scripts/submit_independent_smokes.sh
```

After smoke logs are clean:

```bash
bash scripts/submit_independent_development.sh
```

This submits Development seed 10 for GinAR, RAFT, RAST-native, SARAF, LEAF branches, and SLPF fixed-partial sensing.

## LEAF second phase

The branch job intentionally does not require an LLM. It creates a deterministic selector payload that excludes Validation futures. After it finishes, submit the Qwen selector with a local model path:

```bash
qsub -v STAGE=development,SEED=10,QWEN_MODEL=/absolute/local/Qwen2.5-0.5B-Instruct \
  pbs/methods/leaf_selector.pbs
```

`QWEN_MODEL` must be a local Hugging Face-compatible checkpoint accessible on Gadi. The LEAF selector uses the isolated `/scratch/mh20/zw1165/envs/LEAF/bin/python` environment.

## Main Formal jobs after Development validation

```bash
for m in GinAR RAFT RAST SARAF; do
  for s in 10 20 30; do
    qsub -v METHOD=$m,SEED=$s pbs/methods/formal_main.pbs
  done
done
```

## Secondary Formal seed 10

```bash
bash scripts/submit_secondary_formal_seed10.sh
```

After LEAF formal branch training completes:

```bash
qsub -v STAGE=formal,SEED=10,QWEN_MODEL=/absolute/local/Qwen2.5-0.5B-Instruct \
  pbs/methods/leaf_selector.pbs
```

SLPF Formal is a fixed partial-sensing subtrack. Additional SLPF seeds can be launched with `-v SEED=20` and `-v SEED=30` if required.
