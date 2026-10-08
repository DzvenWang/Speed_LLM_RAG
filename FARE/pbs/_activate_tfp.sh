#!/bin/bash
# Shared Conda bootstrap for PBS compute nodes.  A `conda` executable alone
# is insufficient: activation needs the shell function from conda.sh.
if [[ -f "${HOME}/.bashrc" ]]; then
    source "${HOME}/.bashrc" || true
fi
if ! declare -F conda >/dev/null 2>&1; then
    repo_parent="$(dirname "${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}")"
    for candidate in \
        "${FARE_CONDA_SH:-}" \
        "$repo_parent/miniconda/etc/profile.d/conda.sh" \
        "$repo_parent/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/miniconda/etc/profile.d/conda.sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh"; do
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            source "$candidate"
            break
        fi
    done
fi
if ! declare -F conda >/dev/null 2>&1; then
    echo 'FARE_CONDA_INIT_ERROR: conda shell function unavailable. Set FARE_CONDA_SH to the full path to conda.sh.' >&2
    return 2
fi
conda activate TFP
printf 'FARE_CONDA_READY python=%s env=%s\n' "$(command -v python)" "${CONDA_DEFAULT_ENV:-unknown}" 
