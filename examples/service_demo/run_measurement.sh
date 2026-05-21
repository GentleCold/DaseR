#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the full issue #35 measurement flow (unit tests + sequential A/B infer).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${MODEL_PATH:?Set MODEL_PATH before running}"
PYTHON="${PYTHON:-python}"
VLLM_PORT="${VLLM_PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.85}"
TRIALS="${TRIALS:-5}"
JSON_OUT="${JSON_OUT:-/tmp/daser_demo_benchmark.json}"
VLLM_LOG="${VLLM_LOG:-/tmp/vllm_demo.log}"
TASK="${TASK:-Summarize both documents in two short sentences.}"

source "$ROOT/examples/service_demo/env.sh"

echo "==> Phase 0: unit tests"
"$PYTHON" -m pytest tests/examples/test_service_demo_metrics.py -q

echo "==> Phase 0: connector import"
"$PYTHON" -c "import daser.connector.daser_connector; print('connector ok')"

cleanup() {
  echo "==> cleanup"
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "python -m daser.server" 2>/dev/null || true
}
trap cleanup EXIT

rm -rf /tmp/daser_demo_baseline /tmp/daser_demo_chunk
: > "$VLLM_LOG"

wait_vllm() {
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "vLLM failed to start" >&2
  tail -30 "$VLLM_LOG" >&2
  return 1
}

start_vllm() {
  local kv_config="$1"
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  nohup vllm serve "$MODEL_PATH" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --no-enable-prefix-caching \
    --kv-transfer-config "$kv_config" \
    >>"$VLLM_LOG" 2>&1 &
  echo "vllm pid=$!"
  wait_vllm
}

start_daser() {
  local port="$1" store="$2" sock="$3" mode="$4"
  nohup "$PYTHON" -m daser.server \
    --host 127.0.0.1 --port "$port" \
    --vllm-base-url "http://127.0.0.1:${VLLM_PORT}" \
    --model-path "$MODEL_PATH" \
    --store-dir "$store" \
    --store-size 10gb \
    --socket-path "$sock" \
    --cache-reuse-mode "$mode" \
    >>"/tmp/daser_${port}.log" 2>&1 &
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "DaseR :${port} failed" >&2
  tail -20 "/tmp/daser_${port}.log" >&2
  return 1
}

run_phase() {
  local label="$1" url="$2" kv="$3"
  echo "==> Phase: ${label}"
  if [ "$label" != "baseline" ]; then
    start_vllm "$kv"
  fi
  ROOT="$ROOT" "$PYTHON" - "$url" "$label" "$TRIALS" "$TASK" <<'PY'
import json
import os
import sys
import httpx

root = os.environ["ROOT"]
sys.path.insert(0, os.path.join(root, "examples", "service_demo"))
import metrics as m
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

spec = spec_from_file_location("demo", Path(root) / "examples" / "service_demo" / "demo.py")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)

url, label, trials_s, task = sys.argv[1:5]
trials = int(trials_s)
client = httpx.Client(base_url=url, timeout=600.0)
doc_a, doc_b = mod._upload_docs(client)
latencies: list[float] = []
records: list[dict] = []
for i in range(trials):
    result, wall = mod._infer(
        client, doc_a, doc_b, task,
        trace_cache=True, gen_params=mod.TTFT_GEN_PARAMS,
    )
    hits = result.get("cache_hits") or []
    rec: dict = {
        "trial": i,
        "server_latency_ms": float(result["latency_ms"]),
        "client_wall_ms": wall,
        "prompt_tokens": int(result["prompt_tokens"]),
        "hit_count": len(hits),
    }
    if hits:
        rec["cache_summary"] = m.summarize_cache_hits(
            hits, block_tokens=16, num_layers=36,
        )
    latencies.append(rec["server_latency_ms"])
    records.append(rec)
payload = {
    "label": label,
    "url": url,
    "summary": m.trial_stats(latencies),
    "trials": records,
}
out = Path(f"/tmp/demo_benchmark_{label}.json")
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
}

echo "==> Phase 1: start vLLM (baseline socket) so DaseR can reach /v1/models"
start_vllm "$KV_BASELINE"

echo "==> Phase 2: start DaseR servers"
start_daser 8080 /tmp/daser_demo_baseline /tmp/daser_baseline.sock prefix
start_daser 8081 /tmp/daser_demo_chunk /tmp/daser_chunk.sock chunk

run_phase baseline "http://127.0.0.1:8080" "$KV_BASELINE"
run_phase chunk_reuse "http://127.0.0.1:8081" "$KV_CHUNK"

echo "==> Phase 3: combined summary"
"$PYTHON" - <<PY
import json
from pathlib import Path
baseline = json.loads(Path("/tmp/demo_benchmark_baseline.json").read_text())
chunk = json.loads(Path("/tmp/demo_benchmark_chunk_reuse.json").read_text())
out = {
    "config": {
        "model_path": "${MODEL_PATH}",
        "trials": ${TRIALS},
        "gpu_util": ${GPU_UTIL},
        "measurement": "ttft",
    },
    "baseline": baseline,
    "chunk_reuse": chunk,
    "delta_median_ms": chunk["summary"]["median_ms"] - baseline["summary"]["median_ms"],
    "delta_mean_ms": chunk["summary"]["mean_ms"] - baseline["summary"]["mean_ms"],
}
Path("${JSON_OUT}").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
print("wrote ${JSON_OUT}")
PY

echo "==> connector log (last 20 CONNECTOR lines)"
grep CONNECTOR "$VLLM_LOG" | tail -20 || true
echo "==> done"
