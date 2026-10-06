#pragma once

// Ported from mihailescu2m/llama.cpp (MIT), commit d1762fc.

#include <algorithm>
#include <cstdint>

// Retune a wave cap only if it actually enables pair partitioning. In particular, a valid forced
// cap may exceed the double-buffered default: clamping it DOWN adds masked passes, not partitioning.
inline uint32_t llama_moe_stream_partition_cap(uint64_t n_touch, uint32_t n_slots,
                                              uint32_t n_used, uint32_t max_waves, uint32_t cap) {
    if (max_waves == 0 || n_slots <= n_used || n_used == 0) {
        return cap;
    }
    const uint32_t cap_max = (n_slots - n_used) / 2;
    if (cap_max < n_used) {
        return cap;
    }
    const uint64_t want = (n_touch + max_waves - 1) / max_waves;
    const uint32_t candidate = (uint32_t) std::clamp<uint64_t>(want, n_used, cap_max);
    return candidate > cap && (n_touch + candidate - 1) / candidate <= max_waves ? candidate : cap;
}
