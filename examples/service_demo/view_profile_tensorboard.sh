#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# View DaseR connector load profiles in TensorBoard (PYTORCH PROFILER tab).
#
# TensorBoard runs on the GPU server; use SSH port forwarding to open it in your
# local browser.
#
# Usage:
#   bash examples/service_demo/view_profile_tensorboard.sh
#   bash examples/service_demo/view_profile_tensorboard.sh --port 6006
#
# Local browser (separate terminal):
#   ssh -L 6006:127.0.0.1:6006 user@your-server
#   open http://127.0.0.1:6006/#pytorch_profiler

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGDIR="${PROFILE_TB_LOGDIR:-$ROOT/examples/service_demo/out/tensorboard}"
TB_PORT="${TB_PORT:-6006}"
HOST="$(hostname -f 2>/dev/null || hostname)"
USER_NAME="$(whoami)"
REL_LOGDIR="${LOGDIR#"$ROOT"/}"

if [[ "$LOGDIR" != /* ]]; then
  LOGDIR="$ROOT/$LOGDIR"
fi

if ! python -c "import tensorboard" 2>/dev/null; then
  echo "tensorboard is not installed. Install with:"
  echo "  pip install 'daser[profile]'"
  exit 1
fi

if ! python -c "import torch_tb_profiler" 2>/dev/null; then
  echo "torch-tb-profiler is required for the PYTORCH PROFILER tab."
  echo "Install with:"
  echo "  pip install 'daser[profile]'"
  echo "  # or: pip install torch-tb-profiler"
  echo ""
  echo "Then restart TensorBoard (stop the old process first)."
  exit 1
fi

if [ ! -d "$LOGDIR" ] || [ -z "$(find "$LOGDIR" -name '*.pt.trace.json' -print -quit 2>/dev/null)" ]; then
  echo "No TensorBoard trace under: $LOGDIR"
  echo ""
  echo "Generate profiles on the server:"
  echo "  export MODEL_PATH=/path/to/model"
  echo "  source examples/service_demo/env.sh"
  echo "  bash examples/service_demo/run_profile_load.sh"
  exit 1
fi

if [ "${1:-}" = "--port" ] && [ -n "${2:-}" ]; then
  TB_PORT="$2"
fi

echo "TensorBoard logdir: $LOGDIR"
echo "Traces in workspace: $REL_LOGDIR"
echo ""
echo "=== Open in your LOCAL browser (remote GPU server) ==="
echo "1. Keep this script running (starts TensorBoard on 127.0.0.1:${TB_PORT})."
echo "2. On your laptop, in another terminal:"
echo "     ssh -L ${TB_PORT}:127.0.0.1:${TB_PORT} ${USER_NAME}@${HOST}"
echo "3. Browser: http://127.0.0.1:${TB_PORT}/#pytorch_profiler"
echo ""
echo "Search for segments: daser::rope_delta  daser::gds_read  daser::index_copy"
echo ""

exec tensorboard --logdir "$LOGDIR" --host 127.0.0.1 --port "$TB_PORT"
