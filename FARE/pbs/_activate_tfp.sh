#!/bin/bash
# FARE PBS bootstrap: work under `set -euo pipefail` in Gadi non-interactive
# compute-node shells. Never source ~/.bashrc or /etc/bashrc: these are intended
# for interactive sessions and Gadi's /etc/bashrc may read BASHRCSOURCED before
# defining it (fatal with `set -u`).
#
# The project's expected layout is:
# /scratch/<project>/<user>/Speed_LLM_RAG and ../miniconda.
# Override the location with FARE_CONDA_SH=/absolute/path/to/conda.sh.

_fare_project_root="${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}"
_fare_repo_parent="$(dirname "$_fare_project_root")"

if ! declare -F conda >/dev/null 2>&1; then
    _fare_conda_found=0
    for _fare_candidate in \
        "${FARE_CONDA_SH:-}" \
        "$_fare_repo_parent/miniconda/etc/profile.d/conda.sh" \
        "$_fare_repo_parent/miniconda3/etc/profile.d/conda.sh" \
        "${HOME:-}/miniconda/etc/profile.d/conda.sh" \
        "${HOME:-}/miniconda3/etc/profile.d/conda.sh"; do
        if [[ -n "$_fare_candidate" && -f "$_fare_candidate" ]]; then
            # Some Conda versions are not safe to source/activate with nounset;
            # temporarily disable it, then restore the strict PBS behavior.
            set +u
            if source "$_fare_candidate"; then
                _fare_conda_found=1
            fi
            set -u
            if [[ "$_fare_conda_found" == 1 ]] && declare -F conda >/dev/null 2>&1; then
                printf 'FARE_CONDA_INIT_OK source=%s\n' "$_fare_candidate"
                break
            fi
        fi
    done
fi

if ! declare -F conda >/dev/null 2>&1; then
    printf '%s\n' \
      'FARE_CONDA_INIT_ERROR: conda.sh not loaded. Set FARE_CONDA_SH=/absolute/path/to/miniconda/etc/profile.d/conda.sh.' >&2
    return 2
fi

# Run activation with nounset disabled, but check its status explicitly.
set +u
if conda activate TFP; then
    _fare_activation_ok=1
else
    _fare_activation_ok=0
fi
set -u
if [[ "$_fare_activation_ok" != 1 || "${CONDA_DEFAULT_ENV:-}" != TFP ]]; then
    printf 'FARE_CONDA_ACTIVATE_ERROR: TFP activation failed (env=%s)\n' "${CONDA_DEFAULT_ENV:-unset}" >&2
    return 2
fi

printf 'FARE_CONDA_READY python=%s env=%s\n' "$(command -v python)" "$CONDA_DEFAULT_ENV"
unset _fare_project_root _fare_repo_parent _fare_candidate _fare_conda_found _fare_activation_ok 2>/dev/null || true
