#!/usr/bin/env bash
# Portable PBS runtime. Submit from anywhere inside the repository tree.
# Optional overrides: SPEED_LLM_RAG_ROOT, TFP_ENV_ROOT.
_find_root() {
  local p
  for p in "${SPEED_LLM_RAG_ROOT:-}" "${PBS_O_WORKDIR:-}" "${PWD:-}"; do
    [[ -n "$p" && -d "$p" ]] || continue
    p="$(cd "$p" && pwd -P)"
    while :; do
      if [[ -d "$p/PCFT-STLA" && -d "$p/Backbone" && -d "$p/Base_SupportBench" ]]; then
        ROOT="$p"; return 0
      fi
      [[ "$p" == / ]] && break
      p="$(dirname "$p")"
    done
  done
  echo "ERROR: Cannot find Speed_LLM_RAG root. Submit qsub from the repo tree, or set SPEED_LLM_RAG_ROOT." >&2
  return 2
}
_find_python() {
  if [[ -n "${TFP_ENV_ROOT:-}" ]]; then
    PYTHON="$TFP_ENV_ROOT/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && "$(basename "$CONDA_PREFIX")" == TFP && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
  else
    if ! command -v conda >/dev/null 2>&1; then
      echo 'ERROR: conda not on PATH in PBS. Set TFP_ENV_ROOT (or submit with qsub -V from activated TFP).' >&2
      return 3
    fi
    eval "$(conda shell.bash hook)" || return 3
    conda activate TFP || return 3
    PYTHON="$CONDA_PREFIX/bin/python"
  fi
  [[ -x "$PYTHON" ]] || { echo "ERROR: Python not executable: $PYTHON" >&2; return 3; }
  local envname
  envname="$("$PYTHON" -c 'import sys,pathlib; print(pathlib.Path(sys.prefix).name)')"
  [[ "$envname" == TFP ]] || { echo "ERROR: Selected Python is not from the TFP environment: $PYTHON" >&2; return 3; }
  export PATH="$(dirname "$PYTHON"):$PATH"
}
_find_root
export SPEED_LLM_RAG_ROOT="$ROOT" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/PCFT-STLA:$ROOT/Base_SupportBench:${PYTHONPATH:-}"
_find_python
cd "$ROOT"
