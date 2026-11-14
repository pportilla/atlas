#pragma once

#include "local_basis.hpp"
#include <algorithm>
#include <cstddef>
#include <vector>

namespace atlas::local {

// Each equation a*x_i + h(free coordinates) eliminates its own coordinate
// over QQ, independently of the monomial order and of the nonzero scalar a.
// A pivot occurs in exactly one monomial of the entire set of equations.
inline bool polynomial_graph(const Basis& rows, std::size_t width) {
    if (rows.empty() || rows.size() > width) return false;
    std::vector<std::size_t> occurrences(width), linear_row(width, rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (const auto& [powers, coefficient] : rows[i]) {
            if (powers.size() != width || coefficient == 0) return false;
            std::size_t variables = 0, axis = 0;
            for (std::size_t j = 0; j < width; ++j) {
                if (powers[j] == 0) continue;
                ++occurrences[j];
                ++variables;
                axis = j;
            }
            if (variables == 1 && powers[axis] == 1) linear_row[axis] = i;
        }
    }
    std::vector<bool> covered(rows.size());
    for (std::size_t j = 0; j < width; ++j)
        if (occurrences[j] == 1 && linear_row[j] < rows.size()) covered[linear_row[j]] = true;
    return std::all_of(covered.begin(), covered.end(), [](bool pivot) { return pivot; });
}

} // namespace atlas::local
