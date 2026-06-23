// SPDX-License-Identifier: Apache-2.0
//
// Standalone CUDA C++ microbenchmark for comparing pinned-host and staging
// restore paths into a fragmented, vLLM-like GPU KV cache.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error at ") +         \
                                     __FILE__ + ":" +                       \
                                     std::to_string(__LINE__) + ": " +       \
                                     cudaGetErrorString(_err));              \
        }                                                                    \
    } while (0)

struct Options {
    int device = 0;
    int layers = 36;
    int blocks = 64;
    int block_tokens = 16;
    int heads = 8;
    int head_dim = 128;
    int dtype_bytes = 2;
    int block_stride = 2;
    int pipeline_chunks = 4;
    int requests = 8;
    int warmup = 10;
    int iters = 50;
};

struct TimingResult {
    std::string name;
    float mean_ms = 0.0F;
    float min_ms = 0.0F;
    float p50_ms = 0.0F;
    float p90_ms = 0.0F;
    double effective_gbps = 0.0;
    double traffic_gbps = 0.0;
};

struct DeviceLayerTable {
    std::vector<std::uint8_t *> host_ptrs;
    std::uint8_t **device_ptrs = nullptr;
};

__global__ void scatter_kernel(const std::uint8_t *__restrict__ src,
                               std::uint8_t **__restrict__ kv_layers,
                               size_t total_bytes,
                               size_t per_block,
                               size_t per_slot,
                               int layers,
                               int block_stride,
                               int first_block) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t byte_idx = idx; byte_idx < total_bytes; byte_idx += stride) {
        const size_t local_block = byte_idx / per_slot;
        const size_t in_slot = byte_idx - local_block * per_slot;
        const size_t layer = in_slot / per_block;
        const size_t in_block = in_slot - layer * per_block;
        const size_t block = static_cast<size_t>(first_block) + local_block;
        const size_t dst_block = block * static_cast<size_t>(block_stride);
        kv_layers[layer][dst_block * per_block + in_block] = src[byte_idx];
    }
}

static void print_usage(const char *argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --device N          CUDA device index (default: 0)\n"
        << "  --layers N          Number of KV layers (default: 36)\n"
        << "  --blocks N          Restored blocks per iteration (default: 64)\n"
        << "  --block-tokens N    Tokens per KV block (default: 16)\n"
        << "  --heads N           KV heads (default: 8)\n"
        << "  --head-dim N        Head dimension (default: 128)\n"
        << "  --dtype-bytes N     Bytes per scalar (default: 2)\n"
        << "  --block-stride N    Destination block spacing (default: 2)\n"
        << "  --pipeline-chunks N Number of staging pipeline chunks (default: 4)\n"
        << "  --requests N        Requests in cross-request pipeline (default: 8)\n"
        << "  --warmup N          Warmup iterations (default: 10)\n"
        << "  --iters N           Timed iterations (default: 50)\n"
        << "  --help              Show this help text\n";
}

static int parse_int_arg(char **argv, int argc, int &idx) {
    if (idx + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[idx]);
    }
    ++idx;
    return std::stoi(argv[idx]);
}

static Options parse_args(int argc, char **argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--device") {
            opts.device = parse_int_arg(argv, argc, i);
        } else if (arg == "--layers") {
            opts.layers = parse_int_arg(argv, argc, i);
        } else if (arg == "--blocks") {
            opts.blocks = parse_int_arg(argv, argc, i);
        } else if (arg == "--block-tokens") {
            opts.block_tokens = parse_int_arg(argv, argc, i);
        } else if (arg == "--heads") {
            opts.heads = parse_int_arg(argv, argc, i);
        } else if (arg == "--head-dim") {
            opts.head_dim = parse_int_arg(argv, argc, i);
        } else if (arg == "--dtype-bytes") {
            opts.dtype_bytes = parse_int_arg(argv, argc, i);
        } else if (arg == "--block-stride") {
            opts.block_stride = parse_int_arg(argv, argc, i);
        } else if (arg == "--pipeline-chunks") {
            opts.pipeline_chunks = parse_int_arg(argv, argc, i);
        } else if (arg == "--requests") {
            opts.requests = parse_int_arg(argv, argc, i);
        } else if (arg == "--warmup") {
            opts.warmup = parse_int_arg(argv, argc, i);
        } else if (arg == "--iters") {
            opts.iters = parse_int_arg(argv, argc, i);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (opts.layers <= 0 || opts.blocks <= 0 || opts.block_tokens <= 0 ||
        opts.heads <= 0 || opts.head_dim <= 0 || opts.dtype_bytes <= 0 ||
        opts.block_stride <= 0 || opts.pipeline_chunks <= 0 ||
        opts.requests <= 0 || opts.warmup < 0 || opts.iters <= 0) {
        throw std::invalid_argument("all geometry values must be positive");
    }
    opts.pipeline_chunks = std::min(opts.pipeline_chunks, opts.blocks);
    return opts;
}

static size_t checked_mul(size_t lhs, size_t rhs, const char *label) {
    if (rhs != 0 && lhs > static_cast<size_t>(-1) / rhs) {
        throw std::overflow_error(std::string("size overflow while computing ") +
                                  label);
    }
    return lhs * rhs;
}

static size_t block_bytes(const Options &opts) {
    size_t bytes = static_cast<size_t>(2);  // key and value
    bytes = checked_mul(bytes, static_cast<size_t>(opts.block_tokens), "tokens");
    bytes = checked_mul(bytes, static_cast<size_t>(opts.heads), "heads");
    bytes = checked_mul(bytes, static_cast<size_t>(opts.head_dim), "head_dim");
    bytes = checked_mul(bytes, static_cast<size_t>(opts.dtype_bytes), "dtype");
    return bytes;
}

static size_t slot_bytes(const Options &opts) {
    return checked_mul(static_cast<size_t>(opts.layers), block_bytes(opts),
                       "slot bytes");
}

static size_t total_restore_bytes(const Options &opts) {
    return checked_mul(static_cast<size_t>(opts.blocks), slot_bytes(opts),
                       "restore bytes");
}

static int blocks_per_pipeline_chunk(const Options &opts) {
    return (opts.blocks + opts.pipeline_chunks - 1) / opts.pipeline_chunks;
}

static size_t pipeline_chunk_bytes(const Options &opts) {
    return checked_mul(static_cast<size_t>(blocks_per_pipeline_chunk(opts)),
                       slot_bytes(opts),
                       "pipeline chunk bytes");
}

static size_t destination_layer_bytes(const Options &opts) {
    const size_t block_count = checked_mul(static_cast<size_t>(opts.blocks - 1),
                                           static_cast<size_t>(opts.block_stride),
                                           "destination blocks");
    return checked_mul(block_count + 1, block_bytes(opts), "destination bytes");
}

static float percentile(std::vector<float> values, double q) {
    std::sort(values.begin(), values.end());
    const size_t idx = std::min(
        values.size() - 1,
        static_cast<size_t>(std::llround((values.size() - 1) * q)));
    return values[idx];
}

static void fill_host_pattern(std::uint8_t *data, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) {
        data[i] = static_cast<std::uint8_t>((i * 131U + 17U) & 0xffU);
    }
}

static DeviceLayerTable make_device_layer_table(
    const std::vector<std::uint8_t *> &kv_layers) {
    DeviceLayerTable table;
    table.host_ptrs = kv_layers;
    const size_t table_bytes = checked_mul(kv_layers.size(),
                                           sizeof(std::uint8_t *),
                                           "layer pointer table");
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&table.device_ptrs),
                          table_bytes));
    CUDA_CHECK(cudaMemcpy(table.device_ptrs,
                          table.host_ptrs.data(),
                          table_bytes,
                          cudaMemcpyHostToDevice));
    return table;
}

static void free_device_layer_table(DeviceLayerTable &table) {
    if (table.device_ptrs != nullptr) {
        CUDA_CHECK(cudaFree(table.device_ptrs));
        table.device_ptrs = nullptr;
    }
}

static void direct_h2d_scatter(const Options &opts,
                               std::uint8_t *host,
                               std::vector<std::uint8_t *> &kv_layers,
                               cudaStream_t stream) {
    const size_t per_block = block_bytes(opts);
    const size_t per_slot = slot_bytes(opts);
    for (int block = 0; block < opts.blocks; ++block) {
        const int dst_block = block * opts.block_stride;
        for (int layer = 0; layer < opts.layers; ++layer) {
            const size_t src_offset = static_cast<size_t>(block) * per_slot +
                                      static_cast<size_t>(layer) * per_block;
            const size_t dst_offset = static_cast<size_t>(dst_block) * per_block;
            CUDA_CHECK(cudaMemcpyAsync(kv_layers[layer] + dst_offset,
                                       host + src_offset,
                                       per_block,
                                       cudaMemcpyHostToDevice,
                                       stream));
        }
    }
}

static void staging_h2d_then_d2d_scatter(const Options &opts,
                                         std::uint8_t *host,
                                         std::uint8_t *staging,
                                         std::vector<std::uint8_t *> &kv_layers,
                                         cudaStream_t stream) {
    const size_t per_block = block_bytes(opts);
    const size_t per_slot = slot_bytes(opts);
    const size_t total_bytes = total_restore_bytes(opts);
    CUDA_CHECK(cudaMemcpyAsync(staging,
                               host,
                               total_bytes,
                               cudaMemcpyHostToDevice,
                               stream));
    for (int block = 0; block < opts.blocks; ++block) {
        const int dst_block = block * opts.block_stride;
        for (int layer = 0; layer < opts.layers; ++layer) {
            const size_t src_offset = static_cast<size_t>(block) * per_slot +
                                      static_cast<size_t>(layer) * per_block;
            const size_t dst_offset = static_cast<size_t>(dst_block) * per_block;
            CUDA_CHECK(cudaMemcpyAsync(kv_layers[layer] + dst_offset,
                                       staging + src_offset,
                                       per_block,
                                       cudaMemcpyDeviceToDevice,
                                       stream));
        }
    }
}

static void h2d_to_staging_only(const Options &opts,
                                std::uint8_t *host,
                                std::uint8_t *staging,
                                cudaStream_t stream) {
    const size_t total_bytes = total_restore_bytes(opts);
    CUDA_CHECK(cudaMemcpyAsync(staging,
                               host,
                               total_bytes,
                               cudaMemcpyHostToDevice,
                               stream));
}

static void launch_scatter_kernel(const Options &opts,
                                  const std::uint8_t *src,
                                  std::uint8_t **kv_layers,
                                  size_t bytes,
                                  int first_block,
                                  cudaStream_t stream) {
    const size_t per_block = block_bytes(opts);
    const size_t per_slot = slot_bytes(opts);
    const int threads = 256;
    const int max_blocks = 65535;
    int grid = static_cast<int>((bytes + threads - 1) / threads);
    grid = std::max(1, std::min(grid, max_blocks));
    scatter_kernel<<<grid, threads, 0, stream>>>(
        src,
        kv_layers,
        bytes,
        per_block,
        per_slot,
        opts.layers,
        opts.block_stride,
        first_block);
    CUDA_CHECK(cudaGetLastError());
}

static void kernel_scatter_only(const Options &opts,
                                std::uint8_t *staging,
                                std::uint8_t **kv_layers,
                                cudaStream_t stream) {
    launch_scatter_kernel(opts,
                          staging,
                          kv_layers,
                          total_restore_bytes(opts),
                          0,
                          stream);
}

static void staging_h2d_then_kernel_scatter(const Options &opts,
                                            std::uint8_t *host,
                                            std::uint8_t *staging,
                                            std::uint8_t **kv_layers,
                                            cudaStream_t stream) {
    h2d_to_staging_only(opts, host, staging, stream);
    kernel_scatter_only(opts, staging, kv_layers, stream);
}

static void mapped_host_kernel_scatter(const Options &opts,
                                       std::uint8_t *mapped_host_device_ptr,
                                       std::uint8_t **kv_layers,
                                       cudaStream_t stream) {
    launch_scatter_kernel(opts,
                          mapped_host_device_ptr,
                          kv_layers,
                          total_restore_bytes(opts),
                          0,
                          stream);
}

static void pipelined_staging_kernel_scatter(const Options &opts,
                                             std::uint8_t *host,
                                             std::vector<std::uint8_t *> &staging_slots,
                                             std::uint8_t **kv_layers,
                                             cudaStream_t h2d_stream,
                                             cudaStream_t kernel_stream,
                                             std::vector<cudaEvent_t> &copy_ready_events,
                                             std::vector<cudaEvent_t> &consume_done_events) {
    const size_t per_slot = slot_bytes(opts);
    const int chunk_blocks = blocks_per_pipeline_chunk(opts);
    for (int chunk = 0; chunk < opts.pipeline_chunks; ++chunk) {
        const int first_block = chunk * chunk_blocks;
        if (first_block >= opts.blocks) {
            break;
        }
        const int blocks_this_chunk =
            std::min(chunk_blocks, opts.blocks - first_block);
        const size_t bytes_this_chunk =
            checked_mul(static_cast<size_t>(blocks_this_chunk),
                        per_slot,
                        "pipeline chunk copy");
        const int slot = chunk % static_cast<int>(staging_slots.size());

        if (chunk >= static_cast<int>(staging_slots.size())) {
            CUDA_CHECK(cudaStreamWaitEvent(h2d_stream,
                                           consume_done_events[slot],
                                           0));
        }
        CUDA_CHECK(cudaMemcpyAsync(staging_slots[slot],
                                   host + static_cast<size_t>(first_block) * per_slot,
                                   bytes_this_chunk,
                                   cudaMemcpyHostToDevice,
                                   h2d_stream));
        CUDA_CHECK(cudaEventRecord(copy_ready_events[slot], h2d_stream));
        CUDA_CHECK(cudaStreamWaitEvent(kernel_stream, copy_ready_events[slot], 0));
        launch_scatter_kernel(opts,
                              staging_slots[slot],
                              kv_layers,
                              bytes_this_chunk,
                              first_block,
                              kernel_stream);
        CUDA_CHECK(cudaEventRecord(consume_done_events[slot], kernel_stream));
    }
}

static void cross_request_pipelined_staging_kernel_scatter(
    const Options &opts,
    std::uint8_t *host,
    std::vector<std::uint8_t *> &staging_slots,
    std::uint8_t **kv_layers,
    cudaStream_t h2d_stream,
    cudaStream_t kernel_stream,
    std::vector<cudaEvent_t> &copy_ready_events,
    std::vector<cudaEvent_t> &consume_done_events) {
    for (int request = 0; request < opts.requests; ++request) {
        pipelined_staging_kernel_scatter(opts,
                                         host,
                                         staging_slots,
                                         kv_layers,
                                         h2d_stream,
                                         kernel_stream,
                                         copy_ready_events,
                                         consume_done_events);
    }
}

template <typename Fn>
static TimingResult time_path(const std::string &name,
                              const Options &opts,
                              size_t effective_bytes,
                              double traffic_multiplier,
                              cudaStream_t stream,
                              Fn fn) {
    for (int i = 0; i < opts.warmup; ++i) {
        fn();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> times_ms;
    times_ms.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        cudaEvent_t start;
        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
        fn();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        times_ms.push_back(elapsed_ms);
    }

    const float sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0F);
    const float mean_ms = sum / static_cast<float>(times_ms.size());
    const float min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    const double seconds = static_cast<double>(mean_ms) / 1000.0;

    TimingResult result;
    result.name = name;
    result.mean_ms = mean_ms;
    result.min_ms = min_ms;
    result.p50_ms = percentile(times_ms, 0.50);
    result.p90_ms = percentile(times_ms, 0.90);
    result.effective_gbps = static_cast<double>(effective_bytes) / seconds / 1e9;
    result.traffic_gbps = result.effective_gbps * traffic_multiplier;
    return result;
}

template <typename Fn>
static TimingResult time_dual_stream_path(const std::string &name,
                                          const Options &opts,
                                          size_t effective_bytes,
                                          double traffic_multiplier,
                                          cudaStream_t h2d_stream,
                                          cudaStream_t kernel_stream,
                                          Fn fn) {
    for (int i = 0; i < opts.warmup; ++i) {
        fn();
    }
    CUDA_CHECK(cudaStreamSynchronize(h2d_stream));
    CUDA_CHECK(cudaStreamSynchronize(kernel_stream));

    std::vector<float> times_ms;
    times_ms.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i) {
        cudaEvent_t start;
        cudaEvent_t h2d_done;
        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&h2d_done));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, h2d_stream));
        CUDA_CHECK(cudaStreamWaitEvent(kernel_stream, start, 0));
        fn();
        CUDA_CHECK(cudaEventRecord(h2d_done, h2d_stream));
        CUDA_CHECK(cudaStreamWaitEvent(kernel_stream, h2d_done, 0));
        CUDA_CHECK(cudaEventRecord(stop, kernel_stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(h2d_done));
        CUDA_CHECK(cudaEventDestroy(stop));
        times_ms.push_back(elapsed_ms);
    }

    const float sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0F);
    const float mean_ms = sum / static_cast<float>(times_ms.size());
    const float min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    const double seconds = static_cast<double>(mean_ms) / 1000.0;

    TimingResult result;
    result.name = name;
    result.mean_ms = mean_ms;
    result.min_ms = min_ms;
    result.p50_ms = percentile(times_ms, 0.50);
    result.p90_ms = percentile(times_ms, 0.90);
    result.effective_gbps = static_cast<double>(effective_bytes) / seconds / 1e9;
    result.traffic_gbps = result.effective_gbps * traffic_multiplier;
    return result;
}

static void print_result(const TimingResult &result) {
    std::cout << "| " << result.name << " | "
              << std::fixed << std::setprecision(3) << result.mean_ms << " | "
              << result.min_ms << " | " << result.p50_ms << " | "
              << result.p90_ms << " | " << std::setprecision(2)
              << result.effective_gbps << " | " << result.traffic_gbps
              << " |\n";
}

int main(int argc, char **argv) {
    try {
        const Options opts = parse_args(argc, argv);
        CUDA_CHECK(cudaSetDevice(opts.device));

        cudaDeviceProp prop {};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.device));

        const size_t per_block = block_bytes(opts);
        const size_t per_slot = slot_bytes(opts);
        const size_t restore_bytes = total_restore_bytes(opts);
        const size_t pipeline_bytes = pipeline_chunk_bytes(opts);
        const size_t per_layer_dst = destination_layer_bytes(opts);

        std::uint8_t *host = nullptr;
        std::uint8_t *mapped_host_device_ptr = nullptr;
        std::uint8_t *staging = nullptr;
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&host),
                                 restore_bytes,
                                 cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void **>(&mapped_host_device_ptr),
            host,
            0));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&staging), restore_bytes));
        fill_host_pattern(host, restore_bytes);

        std::vector<std::uint8_t *> pipeline_staging_slots(2);
        for (auto &slot : pipeline_staging_slots) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&slot), pipeline_bytes));
        }

        std::vector<std::uint8_t *> kv_layers(static_cast<size_t>(opts.layers));
        for (int layer = 0; layer < opts.layers; ++layer) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&kv_layers[layer]),
                                  per_layer_dst));
            CUDA_CHECK(cudaMemset(kv_layers[layer], 0, per_layer_dst));
        }
        DeviceLayerTable layer_table = make_device_layer_table(kv_layers);

        cudaStream_t stream;
        cudaStream_t h2d_stream;
        cudaStream_t kernel_stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaStreamCreate(&h2d_stream));
        CUDA_CHECK(cudaStreamCreate(&kernel_stream));

        std::vector<cudaEvent_t> copy_ready_events(pipeline_staging_slots.size());
        std::vector<cudaEvent_t> consume_done_events(pipeline_staging_slots.size());
        for (size_t idx = 0; idx < pipeline_staging_slots.size(); ++idx) {
            CUDA_CHECK(cudaEventCreateWithFlags(&copy_ready_events[idx],
                                                cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&consume_done_events[idx],
                                                cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(consume_done_events[idx], h2d_stream));
        }

        std::cout << "device: " << opts.device << " (" << prop.name << ")\n"
                  << "layers: " << opts.layers << "\n"
                  << "blocks: " << opts.blocks << "\n"
                  << "block_stride: " << opts.block_stride << "\n"
                  << "pipeline_chunks: " << opts.pipeline_chunks << "\n"
                  << "requests: " << opts.requests << "\n"
                  << "block_bytes: " << per_block << "\n"
                  << "slot_bytes: " << per_slot << "\n"
                  << "restored_bytes: " << restore_bytes << "\n\n";

        const TimingResult direct = time_path(
            "direct_h2d_scatter",
            opts,
            restore_bytes,
            1.0,
            stream,
            [&]() { direct_h2d_scatter(opts, host, kv_layers, stream); });

        const TimingResult staging_result = time_path(
            "staging_h2d_then_d2d_scatter",
            opts,
            restore_bytes,
            2.0,
            stream,
            [&]() {
                staging_h2d_then_d2d_scatter(opts, host, staging, kv_layers, stream);
            });

        const TimingResult staging_kernel = time_path(
            "staging_h2d_then_kernel_scatter",
            opts,
            restore_bytes,
            2.0,
            stream,
            [&]() {
                staging_h2d_then_kernel_scatter(
                    opts, host, staging, layer_table.device_ptrs, stream);
            });

        const TimingResult h2d_only = time_path(
            "h2d_to_staging_only",
            opts,
            restore_bytes,
            1.0,
            stream,
            [&]() { h2d_to_staging_only(opts, host, staging, stream); });

        h2d_to_staging_only(opts, host, staging, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const TimingResult kernel_only = time_path(
            "kernel_scatter_only",
            opts,
            restore_bytes,
            1.0,
            stream,
            [&]() {
                kernel_scatter_only(opts, staging, layer_table.device_ptrs, stream);
            });

        const TimingResult pipelined = time_dual_stream_path(
            "pipelined_staging_kernel_scatter",
            opts,
            restore_bytes,
            2.0,
            h2d_stream,
            kernel_stream,
            [&]() {
                pipelined_staging_kernel_scatter(
                    opts,
                    host,
                    pipeline_staging_slots,
                    layer_table.device_ptrs,
                    h2d_stream,
                    kernel_stream,
                    copy_ready_events,
                    consume_done_events);
            });

        const TimingResult cross_request_pipelined = time_dual_stream_path(
            "cross_request_pipelined_staging_kernel_scatter",
            opts,
            checked_mul(restore_bytes,
                        static_cast<size_t>(opts.requests),
                        "cross-request effective bytes"),
            2.0,
            h2d_stream,
            kernel_stream,
            [&]() {
                cross_request_pipelined_staging_kernel_scatter(
                    opts,
                    host,
                    pipeline_staging_slots,
                    layer_table.device_ptrs,
                    h2d_stream,
                    kernel_stream,
                    copy_ready_events,
                    consume_done_events);
            });

        const TimingResult mapped_host_kernel = time_path(
            "mapped_host_kernel_scatter",
            opts,
            restore_bytes,
            1.0,
            stream,
            [&]() {
                mapped_host_kernel_scatter(
                    opts, mapped_host_device_ptr, layer_table.device_ptrs, stream);
            });

        std::cout << "| path | mean ms | min ms | p50 ms | p90 ms | effective GB/s | traffic GB/s |\n";
        std::cout << "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n";
        print_result(direct);
        print_result(staging_result);
        print_result(staging_kernel);
        print_result(h2d_only);
        print_result(kernel_only);
        print_result(pipelined);
        print_result(cross_request_pipelined);
        print_result(mapped_host_kernel);

        for (size_t idx = 0; idx < pipeline_staging_slots.size(); ++idx) {
            CUDA_CHECK(cudaEventDestroy(copy_ready_events[idx]));
            CUDA_CHECK(cudaEventDestroy(consume_done_events[idx]));
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaStreamDestroy(h2d_stream));
        CUDA_CHECK(cudaStreamDestroy(kernel_stream));
        free_device_layer_table(layer_table);
        for (auto *layer : kv_layers) {
            CUDA_CHECK(cudaFree(layer));
        }
        for (auto *slot : pipeline_staging_slots) {
            CUDA_CHECK(cudaFree(slot));
        }
        CUDA_CHECK(cudaFree(staging));
        CUDA_CHECK(cudaFreeHost(host));
        return 0;
    } catch (const std::exception &exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
