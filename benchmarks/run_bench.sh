#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

backend="all"
reuse_mode="chunk"
dataset="longbench"
model=""
store_dir=""
imdb=""
longbench_dir=""
datasets=""
max_samples="20"
gpu_id="auto"
gpu_util="0.85"
max_num_seqs="32"
max_num_batched_tokens="0"
max_inflight="32"
gen_max_tokens="128"
max_context_tokens="0"
evict="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) backend="$2"; shift 2 ;;
    --cache-reuse-mode) reuse_mode="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --store-dir) store_dir="$2"; shift 2 ;;
    --imdb) imdb="$2"; shift 2 ;;
    --longbench-dir) longbench_dir="$2"; shift 2 ;;
    --datasets) datasets="$2"; shift 2 ;;
    --max-samples) max_samples="$2"; shift 2 ;;
    --gpu-id) gpu_id="$2"; shift 2 ;;
    --gpu-util) gpu_util="$2"; shift 2 ;;
    --max-num-seqs) max_num_seqs="$2"; shift 2 ;;
    --max-num-batched-tokens) max_num_batched_tokens="$2"; shift 2 ;;
    --max-inflight) max_inflight="$2"; shift 2 ;;
    --gen-max-tokens) gen_max_tokens="$2"; shift 2 ;;
    --max-context-tokens) max_context_tokens="$2"; shift 2 ;;
    --evict) evict="true"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$model" || -z "$store_dir" ]]; then
  echo "--model and --store-dir are required" >&2
  exit 2
fi

run_id="$(date +%Y%m%d_%H%M%S)"
run_root="${store_dir%/}/run_${run_id}"
mkdir -p "$run_root"

prepare_args=(
  python benchmarks/bench_load.py
  --prepare-only
  --model "$model"
  --store-dir "$run_root"
  --cache-reuse-mode "$reuse_mode"
  --dataset "$dataset"
  --max-samples "$max_samples"
  --max-inflight "$max_inflight"
  --gen-max-tokens "$gen_max_tokens"
  --max-context-tokens "$max_context_tokens"
  --out "$run_root/prepare.json"
)
if [[ "$evict" == "true" ]]; then prepare_args+=(--evict); fi
if [[ -n "$imdb" ]]; then prepare_args+=(--imdb "$imdb"); fi
if [[ -n "$longbench_dir" ]]; then prepare_args+=(--longbench-dir "$longbench_dir"); fi
if [[ -n "$datasets" ]]; then prepare_args+=(--datasets "$datasets"); fi
"${prepare_args[@]}"

derived_l1="$(python - "$run_root/prepare.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["config"]["derived_l1_size_bytes"])
PY
)"
derived_l2="$(python - "$run_root/prepare.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["config"]["derived_l2_size_bytes"])
PY
)"

if [[ "$backend" == "all" ]]; then
  backends=(baseline lmcache daser-chunk daser-prefix)
else
  backends=("$backend")
fi

cleanup() {
  for pid_file in "$run_root"/*/pids.json; do
    [[ -f "$pid_file" ]] || continue
    python - "$pid_file" <<'PY'
import sys
from benchmarks.utils.servers import stop_from_pid_file

stop_from_pid_file(sys.argv[1])
PY
  done
}
trap cleanup EXIT

for be in "${backends[@]}"; do
  case "$be" in
    baseline)
      service_backend="vllm"
      service_reuse_mode="$reuse_mode"
      ;;
    vllm)
      service_backend="vllm"
      service_reuse_mode="$reuse_mode"
      be="baseline"
      ;;
    lmcache)
      service_backend="lmcache"
      service_reuse_mode="$reuse_mode"
      ;;
    daser)
      service_backend="daser"
      service_reuse_mode="$reuse_mode"
      ;;
    daser-chunk)
      service_backend="daser"
      service_reuse_mode="chunk"
      ;;
    daser-prefix)
      service_backend="daser"
      service_reuse_mode="prefix"
      ;;
    *)
      echo "unknown backend: $be" >&2
      exit 2
      ;;
  esac
  be_dir="$run_root/$be"
  mkdir -p "$be_dir"
  start_args=(
    python benchmarks/bench_start_servers.py
    --backend "$service_backend"
    --model "$model"
    --store-dir "$be_dir"
    --run-id "$run_id"
    --gpu-id "$gpu_id"
    --gpu-util "$gpu_util"
    --max-num-seqs "$max_num_seqs"
    --max-num-batched-tokens "$max_num_batched_tokens"
    --l1-size "$derived_l1"
    --l2-size "$derived_l2"
    --cache-reuse-mode "$service_reuse_mode"
  )
  if [[ "$evict" != "true" && ( "$service_backend" == "daser" || "$service_backend" == "lmcache" ) ]]; then
    start_args+=(--skip-l2)
  fi
  "${start_args[@]}"

  load_args=(
    python benchmarks/bench_load.py
    --manifest "$be_dir/manifest.json"
    --prepared-config "$run_root/prepare.json"
    --dataset "$dataset"
    --max-samples "$max_samples"
    --max-inflight "$max_inflight"
    --gen-max-tokens "$gen_max_tokens"
    --max-context-tokens "$max_context_tokens"
    --out "$be_dir/results.json"
  )
  if [[ "$evict" == "true" ]]; then load_args+=(--evict); fi
  if [[ -n "$imdb" ]]; then load_args+=(--imdb "$imdb"); fi
  if [[ -n "$longbench_dir" ]]; then load_args+=(--longbench-dir "$longbench_dir"); fi
  if [[ -n "$datasets" ]]; then load_args+=(--datasets "$datasets"); fi
  "${load_args[@]}"

  cleanup
done

echo "run_root=$run_root"
