#!/usr/bin/env bash
# usage: instantiate.sh <min|mid|max> <alias e.g. gemma-4-e4b-it> -> writes runs/sub_<dose>_<alias>/
set -euo pipefail; cd "$(dirname "$0")"; out="$(cd .. && pwd)/runs/sub_$1_$2"; rm -rf "$out"; cp -r "$1" "$out"; grep -rl MODEL_ALIAS "$out" | xargs sed -i "s/MODEL_ALIAS/$2/g"; echo "$out"
