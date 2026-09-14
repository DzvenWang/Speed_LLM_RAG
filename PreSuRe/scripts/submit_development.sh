#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
PY=/scratch/mh20/zw1165/envs/TFP/bin/python
mkdir -p /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_logs/out /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_exps
"$PY" scripts/preflight.py --stage development
jid(){ echo "$1" | sed 's/\.gadi-pbs$//'; }
J1=$(qsub -v STAGE=development,SEED=10 pbs/prepare_backbone_artifacts.pbs); echo "Prepare backbone artifacts -> $J1"
J2=$(qsub -W depend=afterok:$(jid "$J1") -v STAGE=development,SEED=10 pbs/build_retrieval_candidates.pbs); echo "Build retrieval + graph regions -> $J2"
for O in A B C E F D; do
  JO=$(qsub -W depend=afterok:$(jid "$J2") -v ORACLE_TYPE=$O,STAGE=development,SEED=10 pbs/oracle_support.pbs)
  echo "Oracle-$O -> $JO"
done
J0=$(qsub -W depend=afterok:$(jid "$J2") -v VARIANT=M0,STAGE=development,SEED=10 pbs/variant.pbs); echo "M0 -> $J0"
Jm1=$(qsub -W depend=afterok:$(jid "$J2") -v VARIANT=M1,STAGE=development,SEED=10 pbs/variant.pbs); echo "M1 -> $Jm1"
Jm2=$(qsub -W depend=afterok:$(jid "$Jm1") -v VARIANT=M2,STAGE=development,SEED=10 pbs/variant.pbs); echo "M2 -> $Jm2"
Jm3=$(qsub -W depend=afterok:$(jid "$Jm2") -v VARIANT=M3,STAGE=development,SEED=10 pbs/variant.pbs); echo "M3 horizon-utility -> $Jm3"
Jm4=$(qsub -W depend=afterok:$(jid "$Jm3") -v VARIANT=M4,STAGE=development,SEED=10 pbs/variant.pbs); echo "M4 region-horizon routing -> $Jm4"
Jm5=$(qsub -W depend=afterok:$(jid "$Jm4") -v VARIANT=M5,STAGE=development,SEED=10 pbs/variant.pbs); echo "M5 local safe gate -> $Jm5"
echo "PreSuRe v0.3.0 pipeline submitted: local-oracle diagnostics + M0-M5." 
echo "Key readout: Oracle-C/E/F/D and M2->M3->M4->M5." 
