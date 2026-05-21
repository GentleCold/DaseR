# SPDX-License-Identifier: Apache-2.0
# Optional helpers for the service demo. Customize in your shell or copy:
#   export MODEL_PATH=/path/to/model
#   source examples/service_demo/env.sh
#
# When kvikio fails with GLIBCXX_3.4.31, prepend the active env's lib dir:
#   export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if [ -z "${MODEL_PATH:-}" ]; then
  echo "env.sh: set MODEL_PATH to your HuggingFace model directory" >&2
  return 1 2>/dev/null || exit 1
fi

if [ -z "${LD_LIBRARY_PATH:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
  elif [ -n "${VIRTUAL_ENV:-}" ]; then
    export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib"
  fi
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export VLLM_PORT="${VLLM_PORT:-8001}"

KV_BASELINE='{"kv_connector":"DaserConnector","kv_connector_module_path":"daser.connector.daser_connector","kv_role":"kv_both","kv_connector_extra_config":{"socket_path":"/tmp/daser_baseline.sock"}}'
KV_CHUNK='{"kv_connector":"DaserConnector","kv_connector_module_path":"daser.connector.daser_connector","kv_role":"kv_both","kv_connector_extra_config":{"socket_path":"/tmp/daser_chunk.sock"}}'
