#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
VENDOR="$ROOT/vendor"
PY=/scratch/mh20/zw1165/envs/TFP/bin/python
mkdir -p "$VENDOR"

clone_or_keep(){
  local name="$1"
  local url="$2"
  local ref="$3"
  local dst="$VENDOR/$name"
  # Existing tree-lock-no-git sources from v0.3.1 are intentionally reusable.
  if [[ -d "$dst" && ! -d "$dst/.git" && -f "$VENDOR/SOURCE_LOCK.json" ]]; then
    if "$PY" "$ROOT/scripts/source_lock.py" verify-one --root "$ROOT" --sources "$ROOT/configs/backbone_selection.yaml" --name "$name" >/dev/null 2>&1; then
      echo -e "$name\t$ref\t$url\tPRESERVED_TREE_LOCK"
      return 0
    fi
  fi
  if [[ ! -d "$dst/.git" ]]; then
    rm -rf "$dst"
    echo "Cloning $name from $url" >&2
    git clone "$url" "$dst" >&2
  fi
  git -C "$dst" fetch --all --tags --prune >/dev/null 2>&1 || true
  git -C "$dst" checkout --detach "$ref" >/dev/null
  git -C "$dst" reset --hard "$ref" >/dev/null
  find "$dst" -type d -name __pycache__ -prune -exec rm -rf {} + || true
  find "$dst" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete || true
  echo -e "$name\t$(git -C "$dst" rev-parse HEAD)\t$(git -C "$dst" remote get-url origin)\tCHECKED_OUT"
}

{
  echo -e "model\tcommit\tremote\tmode"
  clone_or_keep STAEformer https://github.com/XDZhelheim/STAEformer.git fc49d39b2f1a8e3cf37b6289d7240680e1690f3f
  clone_or_keep HimNet https://github.com/XDZhelheim/HimNet.git 2a9aa44d1864f6fb60bdf9d14bafd6c1681e81a5
  clone_or_keep BigST https://github.com/usail-hkust/BigST.git 830cb529354c0cda75c1660d1a0af2ffc7e66abb
  clone_or_keep STDN https://github.com/roarer008/STDN.git e24a60fa9485fb9805df9c171d67ab60840d757d
  clone_or_keep PatchSTG https://github.com/LMissher/PatchSTG.git 8b298b230227a304a082e05241c69dca3740a73d
  clone_or_keep STLAformer https://github.com/laibajiu/STLAformer.git f3eba25e2c63023a4834fd87d88da1c3347f16a9
} | tee "$VENDOR/OFFICIAL_COMMITS.tsv"

"$PY" "$ROOT/scripts/source_lock.py" create --root "$ROOT" --sources "$ROOT/configs/backbone_selection.yaml"
echo "Official repositories ready at exact frozen commits; SOURCE_LOCK.json created/extended."
