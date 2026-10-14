#!/bin/bash
# Run from Speed_LLM_RAG root: bash EvalOnly/submit_all.sh --dry-run | --submit-backbone | --submit-baseline | --submit-all
set -euo pipefail
ACTION="${1:---dry-run}"
case "$ACTION" in --dry-run|--submit-backbone|--submit-baseline|--submit-all) ;; *) echo "Usage: bash EvalOnly/submit_all.sh [--dry-run|--submit-backbone|--submit-baseline|--submit-all]"; exit 2;; esac
ROOT="${SPEED_LLM_RAG_ROOT:-$PWD}"
[[ -d "$ROOT/Backbone" && -d "$ROOT/Baseline" ]] || { echo "Execute from Speed_LLM_RAG root or export SPEED_LLM_RAG_ROOT"; exit 2; }
[[ -f "$ROOT/Base_SupportBench/experiment_common/metrics.py" ]] || { echo "Missing Base_SupportBench/experiment_common/metrics.py"; exit 2; }
grep -q 'MAPE_PROTOCOL' "$ROOT/Base_SupportBench/experiment_common/metrics.py" || { echo "Base_SupportBench has not been updated for MAPE; aborting"; exit 2; }
SEED="${SEED:-10}"
TARGETS="${TARGETS:-speed flow}"
BACKBONE_MODELS="${BACKBONE_MODELS:-STAEformer HimNet BigST STDN PatchSTG STLAformer}"
BACKBONE_MODES="${BACKBONE_MODES:-incomplete complete}"
BASELINE_METHODS="${BASELINE_METHODS:-GinAR ImputeFormer LRTC-TMCP RAFT RAST SARAF ReMFT RAFT-PatchSTG RAFT-STLAformer RAST-PatchSTG RAST-STLAformer}"
REFERENCE_MODES="${REFERENCE_MODES:-incomplete complete}"
BACKBONE_EVAL_PBS="$ROOT/EvalOnly/pbs/backbone_eval_only.pbs"
BASELINE_EVAL_PBS="$ROOT/EvalOnly/pbs/baseline_eval_only.pbs"
[[ -f "$BACKBONE_EVAL_PBS" && -f "$BASELINE_EVAL_PBS" ]] || { echo "EvalOnly PBS scripts missing"; exit 2; }
COUNT=0; SKIP=0
submit() {
  local label="$1" name="$2" vars="$3" pbs="$4"
  COUNT=$((COUNT+1))
  if [[ "$ACTION" == --dry-run ]]; then
    printf '[DRY RUN] %s\n    qsub -N %s -v %q %q\n' "$label" "$name" "$vars" "$pbs"
  else
    local result
    result="$(qsub -N "$name" -v "$vars" "$pbs")"
    printf '[SUBMITTED] %-50s %s\n' "$label" "$result"
  fi
}
skip() { SKIP=$((SKIP+1)); printf '[SKIP] %s (no completed checkpoint found)\n' "$1"; }
# Find latest timestamp-named directory that contains a checkpoint; prefer not to follow stale LATEST pointers.
latest_backbone() {
  local root="$1" p
  [[ -d "$root" ]] || return 1
  while IFS= read -r p; do [[ -f "$p" ]] && { printf '%s\n' "$p"; return 0; }; done < <(find "$root" -mindepth 2 -maxdepth 2 -type f -name best.pt -path "*_seed${SEED}/best.pt" -print | sort -r)
  return 1
}
latest_formal_run() {
  local root="$1" d
  [[ -d "$root" ]] || return 1
  while IFS= read -r d; do
    if [[ -f "$d/best.pt" && -f "$d/summary.json" ]]; then
      # Exclude failed or interrupted runs, if the summary has a status marker.
      if grep -Eq '"status"[[:space:]]*:[[:space:]]*"COMPLETE"' "$d/summary.json"; then printf '%s\n' "$d"; return 0; fi
    fi
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -name "*_seed${SEED}" -print | sort -r)
  return 1
}
if [[ "$ACTION" == --dry-run || "$ACTION" == --submit-all || "$ACTION" == --submit-backbone ]]; then
  echo '======== BACKBONE eval-only ========'
  for t in $TARGETS; do
    for m in $BACKBONE_MODELS; do
      for mode in $BACKBONE_MODES; do
        stage=formal; [[ "$mode" == complete ]] && stage=formal_complete
        base="$ROOT/Backbone_exps/$t/$m/TraffiDent_Net_06/$stage"
        ckpt="$(latest_backbone "$base" || true)"
        label="BB $t/$m/$mode"
        if [[ -z "$ckpt" ]]; then skip "$label"; continue; fi
        short_model="${m:0:4}"; short_t="${t:0:1}"; short_mode="${mode:0:1}"
        name="BB_${short_t}_${short_model}_${short_mode}"
        submit "$label | $ckpt" "$name" "MODEL=$m,TARGET=$t,MODE=$mode,SEED=$SEED,CKPT=$ckpt" "$BACKBONE_EVAL_PBS"
      done
    done
  done
fi
if [[ "$ACTION" == --dry-run || "$ACTION" == --submit-all || "$ACTION" == --submit-baseline ]]; then
  echo '======== BASELINE eval-only ========'
  for t in $TARGETS; do
    for m in $BASELINE_METHODS; do
      refs='' # non-reference methods have one run; reference methods can have two backbones x two modes
      case "$m" in
        ImputeFormer|LRTC-TMCP) refs='STLAformer PatchSTG';;
        RAFT-STLAformer|RAST-STLAformer) refs='STLAformer';;
        RAFT-PatchSTG|RAST-PatchSTG) refs='PatchSTG';;
        GinAR|RAFT|RAST|SARAF|ReMFT) refs='native';;
        *) echo "Skipping unsupported evaluation method: $m"; continue;;
      esac
      for ref in $refs; do
        if [[ "$ref" == native ]]; then modes='incomplete'; else modes="$REFERENCE_MODES"; fi
        for mode in $modes; do
          if [[ "$ref" == native ]]; then
            parent="$ROOT/Baseline_exps/$t/$m/TraffiDent_Net_06/formal"
          else
            parent="$ROOT/Baseline_exps/$t/$m/$ref"
            [[ "$mode" == complete ]] && parent="$parent/complete"
            parent="$parent/TraffiDent_Net_06/formal"
          fi
          run="$(latest_formal_run "$parent" || true)"
          label="BL $t/$m/$ref/$mode"
          if [[ -z "$run" ]]; then skip "$label"; continue; fi
          # PBS job names short and safe; full metadata goes into the live log.
          short_m="$(printf '%s' "$m" | tr -cd '[:alnum:]' | cut -c1-5)"
          short_ref="${ref:0:1}"; short_mode="${mode:0:1}"; short_t="${t:0:1}"
          name="BL_${short_t}_${short_m}_${short_ref}${short_mode}"
          vars="METHOD=$m,TARGET=$t,SEED=$SEED,MODE=$mode,FORMAL_RUN_DIR=$run"
          [[ "$ref" == native ]] || vars+=",BACKBONE=$ref"
          submit "$label | $run" "$name" "$vars" "$BASELINE_EVAL_PBS"
        done
      done
    done
  done
fi
printf '\nTotal matched jobs: %d; skipped combinations: %d\n' "$COUNT" "$SKIP"
if [[ "$ACTION" == --dry-run ]]; then echo 'Dry run only; nothing was submitted. Use --submit-backbone, --submit-baseline or --submit-all after review.'; fi
