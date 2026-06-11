#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

exec python benchmarks/run_bench.py "$@"
