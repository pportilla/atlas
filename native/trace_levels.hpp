// Incoming edges of the triangular reducer graph. Contributions to a column
// depend only on earlier levels, so columns at one level have independent sums.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace atlas {

// Output array sizes: width+1 starts, offsets[width] parents and coefficients,
// width columns, and width+1 level boundaries. Return the number of levels.
inline uint32_t trace_levels(uint32_t width, const uint32_t* offsets, const uint32_t* targets,
                             const uint32_t* coefficients, uint32_t* starts, uint32_t* parents,
                             uint32_t* factors, uint32_t* order, uint32_t* levels) {
    std::vector<uint32_t> depth(width), cursor(width);
    std::fill_n(starts, width + 1, 0);
    std::fill_n(levels, width + 1, 0);
    uint32_t count = 0;
    for (uint32_t j = 0; j < width; ++j) {
        const uint32_t next = depth[j] + 1;
        count = std::max(count, next);
        ++levels[next];
        for (uint32_t t = offsets[j]; t < offsets[j + 1]; ++t) {
            const uint32_t dst = targets[t];
            if (dst <= j || dst >= width) return 0;
            depth[dst] = std::max(depth[dst], next);
            ++starts[dst + 1];
        }
    }
    for (uint32_t j = 0; j < width; ++j) {
        starts[j + 1] += starts[j];
        cursor[j] = starts[j];
    }
    for (uint32_t j = 0; j < width; ++j) {
        for (uint32_t t = offsets[j]; t < offsets[j + 1]; ++t) {
            const uint32_t at = cursor[targets[t]]++;
            parents[at] = j;
            factors[at] = coefficients[t];
        }
    }
    for (uint32_t d = 0; d < count; ++d) {
        levels[d + 1] += levels[d];
        cursor[d] = levels[d];
    }
    for (uint32_t j = 0; j < width; ++j) order[cursor[depth[j]]++] = j;
    return count;
}

}  // namespace atlas
