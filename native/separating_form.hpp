#pragma once

#include <vector>

namespace atlas {
// Choose a small prospective separating direction for the shape representation
// of points in a finite algebra. Each caller checks the extra coordinate.
inline std::vector<long> separating_weights(size_t n, size_t trial) {
    std::vector<long> weights(n);
    for (size_t j = 0; j < n; ++j)
        weights[j] = 1 + static_cast<long>((trial + 1) * (j + 1) * 7 % (2 * trial + 5));
    return weights;
}
}
