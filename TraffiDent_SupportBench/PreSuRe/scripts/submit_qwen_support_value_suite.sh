#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
PY=/scratch/mh20/zw1165/envs/TFP/bin/python
"$PY" scripts/qwen_reasoning_preflight.py --stage development
jid_data=$(qsub pbs/qwen_reasoning_dataset.pbs)
echo "DATASET=$jid_data"
jid_z05=$(qsub -W depend=afterok:"$jid_data" -v MODEL_SIZE=0.5b,MODE=zero_shot pbs/qwen_reasoner_probe.pbs)
jid_z15=$(qsub -W depend=afterok:"$jid_data" -v MODEL_SIZE=1.5b,MODE=zero_shot pbs/qwen_reasoner_probe.pbs)
jid_t05=$(qsub -W depend=afterok:"$jid_data" -v MODEL_SIZE=0.5b pbs/qwen_reasoner_train.pbs)
jid_t15=$(qsub -W depend=afterok:"$jid_data" -v MODEL_SIZE=1.5b pbs/qwen_reasoner_train.pbs)
echo "ZERO05=$jid_z05 ZERO15=$jid_z15 TRAIN05=$jid_t05 TRAIN15=$jid_t15"
jid_l05=$(qsub -W depend=afterok:"$jid_t05" -v MODEL_SIZE=0.5b,MODE=lora pbs/qwen_reasoner_probe.pbs)
jid_l15=$(qsub -W depend=afterok:"$jid_t15" -v MODEL_SIZE=1.5b,MODE=lora pbs/qwen_reasoner_probe.pbs)
echo "LORA05=$jid_l05 LORA15=$jid_l15"
jid_sum=$(qsub -W depend=afterok:"$jid_z05":"$jid_z15":"$jid_l05":"$jid_l15" pbs/qwen_reasoning_summary.pbs)
echo "SUMMARY=$jid_sum"
echo "Submitted full v0.8.0 Qwen support-value suite."
