#!/usr/bin/env bash
# llama-server for the swegemma harness (LOCAL_INFERENCE_URL=http://127.0.0.1:${PORT}/v1).
# Usage: ./run_server.sh e4b|e2b|12b|26b [GPU] [PORT] [SLOTS]
# Google QAT q4_0 GGUF, 32k ctx (harness max_model_len), thinking budget 4096 (harness default),
# --jinja so OpenAI tool_calls / reasoning_content are produced (smoke-tested 2026-10-22 with e4b).
set -euo pipefail
BIN=/home/happy/llama.cpp-cu13/build/bin/llama-server
case "${1:?model}" in
  e4b) MODEL=/home/happy/models/gemma4-e4b-qat/gemma-4-E4B_q4_0-it.gguf ;;
  e2b) MODEL=/home/happy/models/gemma4-e2b-qat/gemma-4-E2B_q4_0-it.gguf ;;
  12b) MODEL=/home/happy/models/gemma4-12b-qat/gemma-4-12b-it-qat-q4_0.gguf ;;
  26b) MODEL=/home/happy/models/gemma4-26b-a4b-qat/gemma-4-26B_q4_0-it.gguf ;;
  *) echo "unknown model $1" >&2; exit 1 ;;
esac
GPU=${2:-0}; PORT=${3:-8000}; SLOTS=${4:-1}
[ -r "$MODEL" ] || { echo "missing $MODEL" >&2; exit 1; }
sha256sum "$MODEL" > "$(dirname "$0")/logs/model_sha256_$1.txt" 2>/dev/null || true
exec env CUDA_VISIBLE_DEVICES="$GPU" "$BIN" -m "$MODEL" \
  -c "${CTX:-32768}" --reasoning-budget "${REASONING_BUDGET:-4096}" \
  --flash-attn on --jinja -t 8 -np "$SLOTS" --host 127.0.0.1 --port "$PORT" \
  --temp "${TEMP:-1.0}" --top-p "${TOP_P:-0.95}"
