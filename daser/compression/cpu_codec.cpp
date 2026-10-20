// SPDX-License-Identifier: Apache-2.0

// Offline feasibility encoder for the existing DKVSLOT1 3+3-bit format.
// The Python boundary validates geometry and buffer lifetimes. There is no
// allocator, thread pool, hash computation, or floating-point conversion in
// encode(): BF16 NaNs, signed zeros, and subnormals remain literal bits.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace {
constexpr uint32_t alignment = 4096;
constexpr uint32_t tile_scalars = 256;

void put32(uint8_t* p, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) p[i] = value >> (8 * i);
}

void put64(uint8_t* p, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) p[i] = value >> (8 * i);
}

uint32_t packed_bytes(uint32_t count) {
    return (uint64_t(count) * 3 + 7) / 8;
}

struct Plane {
    uint32_t escapes, raw_escapes, start, symbols, prefixes, escape_start, end;
};

// Streaming writers assign each byte once, including partial final bytes.
// They never OR into stale output and never write past the packed stream.
class Bits {
    uint8_t* output;
    uint32_t pending = 0;
    unsigned used = 0;
public:
    explicit Bits(uint8_t* output) : output(output) {}
    void append(uint8_t value) {
        pending |= uint32_t(value) << used;
        used += 3;
        if (used >= 8) {
            *output++ = pending;
            pending >>= 8;
            used -= 8;
        }
    }
    void finish() {
        if (used) *output = pending;
    }
};

struct Encoder {
    uint32_t scalars, raw_bytes, tiles;
    std::array<uint8_t, alignment> header;
    std::vector<std::array<uint8_t, 256>> lookup;
    std::vector<Plane> planes;

    Encoder(uint32_t count, uint32_t scalars, uint32_t raw_bytes,
            const uint8_t* tables, const uint8_t* template_header)
        : scalars(scalars), raw_bytes(raw_bytes),
          tiles((scalars + tile_scalars - 1) / tile_scalars),
          lookup(count), planes(count) {
        std::memcpy(header.data(), template_header, alignment);
        for (uint32_t p = 0; p < count; ++p) {
            lookup[p].fill(15);
            for (uint8_t i = 0; i < 15; ++i) lookup[p][tables[p * 15 + i]] = i;
        }
    }

    uint32_t encode(const uint8_t* raw, uint8_t* output, uint64_t slot_id,
                    uint32_t* mode) {
        uint64_t cursor = alignment;
        // First pass reads only the high bytes. The complete slot decides RAW
        // fallback; unlike the GPU writer no individual plane has a cap.
        for (size_t p = 0; p < planes.size(); ++p) {
            auto& plane = planes[p];
            const auto& table = lookup[p];
            const uint8_t* input = raw + p * uint64_t(scalars) * 2;
            uint32_t escapes = 0, raw_escapes = 0;
            for (uint32_t i = 0; i < scalars; ++i) {
                uint8_t code = table[input[2 * i + 1]];
                escapes += code >= 7;
                raw_escapes += code >= 14;
            }
            uint64_t end = cursor + scalars + packed_bytes(scalars) +
                           8 * uint64_t(tiles + 1) +
                           packed_bytes(escapes) + raw_escapes;
            // Early fallback is exact: subsequent plane lengths are positive.
            if (end > raw_bytes) {
                std::memcpy(output, raw, raw_bytes);
                *mode = 0;
                return raw_bytes;
            }
            plane = {escapes, raw_escapes, uint32_t(cursor),
                     uint32_t(cursor + scalars),
                     uint32_t(cursor + scalars + packed_bytes(scalars)),
                     uint32_t(end - packed_bytes(escapes) - raw_escapes),
                     uint32_t(end)};
            cursor = end;
        }
        const uint32_t length = (cursor + alignment - 1) & ~(uint64_t(alignment) - 1);
        if (length > raw_bytes) {
            std::memcpy(output, raw, raw_bytes);
            *mode = 0;
            return raw_bytes;
        }
        std::memcpy(output, header.data(), alignment);
        put64(output + 24, slot_id);
        put64(output + 48, length);
        for (size_t p = 0; p < planes.size(); ++p) {
            const auto& plane = planes[p];
            const auto& table = lookup[p];
            const uint8_t* input = raw + p * uint64_t(scalars) * 2;
            // The template already supplies layer, K/V, scalar/tile counts and
            // reserved fields. Descriptor uint32 fields may be unaligned.
            uint8_t* descriptor = output + 120 + p * 48;
            put32(descriptor + 12, plane.start);
            put32(descriptor + 16, plane.end - plane.start);
            put32(descriptor + 20, plane.start);
            put32(descriptor + 24, plane.symbols);
            put32(descriptor + 28, plane.prefixes);
            put32(descriptor + 32, plane.escape_start);
            put32(descriptor + 36, plane.escapes);
            uint8_t* prefixes = output + plane.prefixes;
            uint8_t* raw_prefixes = prefixes + 4 * (tiles + 1);
            put32(prefixes, 0);
            put32(raw_prefixes, 0);
            Bits symbols(output + plane.symbols);
            Bits escapes(output + plane.escape_start);
            uint8_t* raw_stream = output + plane.escape_start + packed_bytes(plane.escapes);
            uint32_t escape_count = 0, raw_count = 0;
            // Second pass emits literal lows, both code streams, and prefixes
            // directly into the caller's output, reusing only per-plane counts.
            for (uint32_t i = 0; i < scalars; ++i) {
                output[plane.start + i] = input[2 * i];
                uint8_t high = input[2 * i + 1];
                uint8_t code = table[high];
                symbols.append(std::min<uint8_t>(code, 7));
                if (code >= 7) {
                    ++escape_count;
                    escapes.append(code >= 14 ? 7 : code - 7);
                    if (code >= 14) raw_stream[raw_count++] = high;
                }
                if ((i + 1) % tile_scalars == 0 || i + 1 == scalars) {
                    uint32_t tile = (i / tile_scalars) + 1;
                    put32(prefixes + 4 * tile, escape_count);
                    put32(raw_prefixes + 4 * tile, raw_count);
                }
            }
            symbols.finish();
            escapes.finish();
        }
        std::memset(output + cursor, 0, length - cursor);
        *mode = 1;
        return length;
    }
};
}  // namespace

extern "C" {
void* daser_cpu_encoder_create(uint32_t planes, uint32_t scalars,
                               uint32_t raw_bytes, const uint8_t* tables,
                               const uint8_t* header) noexcept {
    try {
        return new Encoder(planes, scalars, raw_bytes, tables, header);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}

void daser_cpu_encoder_destroy(void* context) noexcept {
    delete static_cast<Encoder*>(context);
}

uint64_t daser_cpu_encoder_workspace_bytes(void* context) noexcept {
    const auto& encoder = *static_cast<Encoder*>(context);
    return sizeof(Encoder) +
           encoder.lookup.capacity() * sizeof(std::array<uint8_t, 256>) +
           encoder.planes.capacity() * sizeof(Plane);
}

uint32_t daser_cpu_encoder_encode(void* context, const uint8_t* raw,
                                 uint8_t* output, uint64_t slot_id,
                                 uint32_t* mode) noexcept {
    return static_cast<Encoder*>(context)->encode(raw, output, slot_id, mode);
}
}
