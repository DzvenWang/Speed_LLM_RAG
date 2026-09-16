#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
VENDOR="$ROOT/vendor"
PY=/scratch/mh20/zw1165/envs/TFP/bin/python
SOURCES="$ROOT/configs/official_sources.yaml"
mkdir -p "$VENDOR"

locked_commit(){
  local name="$1"
  "$PY" - "$VENDOR/SOURCE_LOCK.json" "$name" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); name=sys.argv[2]
if not p.exists(): raise SystemExit(1)
j=json.loads(p.read_text()); rec=j.get('repositories',{}).get(name)
if not rec: raise SystemExit(1)
print(rec['commit'])
PY
}

clone_or_keep(){
  local name="$1"
  local url="$2"
  local policy="$3"
  local requested="$4"
  local dst="$VENDOR/$name"

  if [[ -d "$dst" && ! -d "$dst/.git" && -f "$VENDOR/SOURCE_LOCK.json" ]]; then
    if "$PY" "$ROOT/scripts/source_lock.py" verify-one --root "$ROOT" --sources "$SOURCES" --name "$name" >/dev/null 2>&1; then
      local lc
      lc=$(locked_commit "$name")
      echo -e "$name\t$lc\t$url\tPRESERVED_TREE_LOCK"
      return 0
    fi
  fi

  if [[ ! -d "$dst/.git" ]]; then
    rm -rf "$dst"
    echo "Cloning $name from $url" >&2
    git clone "$url" "$dst" >&2
  fi
  git -C "$dst" fetch --all --tags --prune >/dev/null 2>&1 || true

  local target="$requested"
  if [[ "$policy" == "resolve_once" ]] && lc=$(locked_commit "$name" 2>/dev/null); then
    target="$lc"
  fi
  git -C "$dst" checkout --detach "$target" >/dev/null
  git -C "$dst" reset --hard "$target" >/dev/null
  find "$dst" -type d -name __pycache__ -prune -exec rm -rf {} + || true
  find "$dst" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete || true
  echo -e "$name\t$(git -C "$dst" rev-parse HEAD)\t$(git -C "$dst" remote get-url origin)\tCHECKED_OUT"
}

{
  echo -e "model\tcommit\tremote\tmode"
  clone_or_keep ImputeFormer https://github.com/tongnie/ImputeFormer.git exact 1c3868cf499951ac43b4fc7280593fbb745f58c7
  clone_or_keep LRTC-TMCP https://github.com/peterchen96/LRTC-TMCP.git resolve_once main
  clone_or_keep SLPF https://github.com/zbliu98/SLPF.git exact 8f135a96d57651dc19564bd3b94508c1f92c04fb
  clone_or_keep RAFT https://github.com/archon159/RAFT.git exact bfe2320f5d719d1e2a024330de3410b370bbb2a9
  clone_or_keep RAST https://github.com/RWLinno/RAST.git exact 4b48a47a002dd8d4151afe47be8f436f0c028a67
  clone_or_keep LEAF https://github.com/YushengZhao/LEAF.git exact 650b9e915fd4cc213387e279e4e529acbe587b85
  clone_or_keep GinAR https://github.com/GestaltCogTeam/GinAR.git resolve_once main
  clone_or_keep SARAF https://github.com/ShiqiaoZhou/SARAF.git resolve_once main
  clone_or_keep SPRINT https://github.com/ashishbhaskar/SPRINT.git resolve_once main
  clone_or_keep MoGERNN https://github.com/ZJU-TSELab/MoGERNN.git resolve_once main
  clone_or_keep ReMFT https://github.com/IansSUn/ReMFT.git exact af61bef8ec872ba7b12e78675a39aeb0a20088a6
} | tee "$VENDOR/OFFICIAL_COMMITS.tsv"

"$PY" "$ROOT/scripts/source_lock.py" create --root "$ROOT" --sources "$SOURCES"
echo "Official repositories ready and frozen by SOURCE_LOCK.json."
echo "LRTC-TMCP/GinAR/SARAF/SPRINT/MoGERNN use resolve-once: the first successful setup resolves main to an exact SHA; later runs reuse that SHA."
