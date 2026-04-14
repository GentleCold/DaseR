# Server Resources

Hardware topology, storage paths, and default test/model paths for the DaseR development server.

---

## GPU Inventory

| Index | Model | VRAM | PCIe Bus ID |
|-------|-------|------|-------------|
| 0 | NVIDIA H800 | 80 GB | `0000:09:00.0` |
| 1 | NVIDIA GeForce RTX 4090 | 24 GB | `0000:11:00.0` |
| 2 | NVIDIA H800 | 80 GB | `0000:33:00.0` |
| 3 | NVIDIA H800 | 80 GB | `0000:38:00.0` |
| 4 | NVIDIA H800 PCIe | 80 GB | `0000:3C:00.0` |

**Notes:**
- GPUs 0, 2, 3, 4 are H800s (SXM or PCIe); prefer these for large-model inference.
- GPU 1 (RTX 4090) has 24 GB VRAM; suitable for development and small-model testing.
- Default inference GPU: `cuda:0` (H800, 80 GB).

---

## NVMe Inventory

| Device | Model | Capacity | PCIe Bus ID | NUMA Node | Mount |
|--------|-------|----------|-------------|-----------|-------|
| `/dev/nvme1n1` | MEMBLAZE P6531DT 3.84 TB | 3.5 TiB | `0000:5c:00.0` | 0 | — |
| `/dev/nvme2n1` | MEMBLAZE P6531DT 3.84 TB | 3.5 TiB | `0000:5a:00.0` | 0 | — |
| `/dev/nvme3n1` | MEMBLAZE P6531DT 3.84 TB | 3.5 TiB | `0000:5b:00.0` | 0 | `/data` |
| `/dev/nvme4n1` | MEMBLAZE P6531DT 3.84 TB | 3.5 TiB | `0000:5b:00.0` | 0 | — |

**Notes:**
- All NVMes are MEMBLAZE P6531DT enterprise NVMe SSDs, all on NUMA node 0.
- `/dev/nvme3n1` is mounted at `/data` (btrfs); this is the primary data volume.
- `/dev/nvme1n1`, `/dev/nvme2n1`, `/dev/nvme4n1` are available for additional storage pools or RAID configurations.
- For GDS testing, use devices on NUMA node 0 paired with GPUs on the same node.

---

## NUMA Topology

```
Node 0: CPUs 0-47, 96-143  |  ~504 GB RAM
Node 1: CPUs 48-95, 144-191 |  ~504 GB RAM

Inter-node distance: 21  (local: 10)
```

All NVMe devices are on NUMA node 0. Prefer NUMA-local CPU pinning (`numactl --cpunodebind=0`) when running DaseR server with GDS workloads.

---

## Default Paths

| Resource | Path |
|----------|------|
| Default model | `/data/zwt/model/models/Qwen/Qwen3-8B` |
| Test working directory | `/data/zwt/daser_test/` |
| Data volume root | `/data/` |
| Python venv | `/data/zwt/vllm/` |

**Default model:** Qwen3-8B is the standard model for integration tests and benchmarks unless otherwise specified.

**Test directory:** `/data/zwt/daser_test/` is the scratch area for ring-buffer files, IPC sockets, and other test artifacts. Clean it between runs as needed.
