#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Profile one chunk-reuse infer with PyTorch profiler + TensorBoard export.
#
# Prerequisites: same as run_measurement.sh (MODEL_PATH, conda/venv, GPU).
# Writes:
#   examples/service_demo/out/tensorboard/  - TensorBoard logdir (PYTORCH PROFILER tab)
#   /tmp/daser_load_profile.log             - vLLM log (grep "load profile report")
#
# Usage:
#   export MODEL_PATH=/path/to/model
#   source examples/service_demo/env.sh
#   bash examples/service_demo/run_profile_load.sh
#   bash examples/service_demo/view_profile_tensorboard.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${MODEL_PATH:?Set MODEL_PATH before running}"
PYTHON="${PYTHON:-python}"
VLLM_PORT="${VLLM_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.85}"
VLLM_LOG="${VLLM_LOG:-/tmp/daser_load_profile.log}"
TB_LOGDIR="${PROFILE_TB_LOGDIR:-$ROOT/examples/service_demo/out/tensorboard}"
mkdir -p "$TB_LOGDIR"

if ! "$PYTHON" -c "import tensorboard" 2>/dev/null \
  || ! "$PYTHON" -c "import torch_tb_profiler" 2>/dev/null; then
  echo "Installing profile deps (tensorboard + torch-tb-profiler)..."
  "$PYTHON" -m pip install -q "tensorboard>=2.14" "torch-tb-profiler>=0.4"
fi

source "$ROOT/examples/service_demo/env.sh"

KV_PROFILE=$(cat <<EOF
{"kv_connector":"DaserConnector","kv_connector_module_path":"daser.connector.daser_connector","kv_role":"kv_both","kv_connector_extra_config":{"socket_path":"/tmp/daser_chunk.sock","load_profile":true,"load_profile_tensorboard_dir":"${TB_LOGDIR}"}}
EOF
)

cleanup() {
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "python -m daser.server.*8081" 2>/dev/null || true
}
trap cleanup EXIT

rm -rf /tmp/daser_demo_chunk
: > "$VLLM_LOG"

echo "==> start vLLM (load_profile=true, tensorboard logdir=$TB_LOGDIR)"
nohup vllm serve "$MODEL_PATH" \
  --port "$VLLM_PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --no-enable-prefix-caching \
  --kv-transfer-config "$KV_PROFILE" \
  >>"$VLLM_LOG" 2>&1 &
for _ in $(seq 1 90); do
  curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 && break
  sleep 5
done

echo "==> start DaseR chunk-reuse server"
nohup "$PYTHON" -m daser.server \
  --host 127.0.0.1 --port 8081 \
  --vllm-base-url "http://127.0.0.1:${VLLM_PORT}" \
  --model-path "$MODEL_PATH" \
  --store-dir /tmp/daser_demo_chunk \
  --store-size 10gb \
  --socket-path /tmp/daser_chunk.sock \
  --cache-reuse-mode chunk \
  >>/tmp/daser_profile_server.log 2>&1 &
for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:8081/health >/dev/null 2>&1 && break
  sleep 1
done

echo "==> upload + warm infer (profile on chunk-hit load)"
"$PYTHON" examples/service_demo/demo.py --service-url http://127.0.0.1:8081 --ttft-only \
  >/tmp/daser_profile_demo_warm.log 2>&1 || true
"$PYTHON" examples/service_demo/demo.py --service-url http://127.0.0.1:8081 --ttft-only \
  >/tmp/daser_profile_demo.log 2>&1 || true

echo "==> load profile report (from vLLM log)"
grep -A 30 "load profile report" "$VLLM_LOG" || {
  echo "profile report not found; showing CONNECTOR tail:"
  grep CONNECTOR "$VLLM_LOG" | tail -30
}

TRACE_COUNT="$(find "$TB_LOGDIR" -name '*.pt.trace.json' 2>/dev/null | wc -l | tr -d ' ')"
echo "==> tensorboard traces: $TRACE_COUNT file(s) under $TB_LOGDIR"
echo "    view: bash examples/service_demo/view_profile_tensorboard.sh"
