#pragma once

#include "local_basis.hpp"
#include <stdexcept>

namespace atlas::local {
struct PointOutsideScheme : std::invalid_argument {
    PointOutsideScheme() : std::invalid_argument("The point must lie on the scheme.") {}
};
struct DimensionResult {
    // A dimension-only solve may leave tangent metadata at -1.
    int dimension, embedding_dimension, jacobian_rank;
    bool smooth;
};

inline DimensionResult ambient_space_at_point(int width) {
    return {width, width, 0, true};
}

DimensionResult dimension_at_point(const Basis& ideal, const std::vector<std::string>& names,
                                   const std::vector<mpq_class>& point, bool lex, bool smoothness,
                                   bool prefer_modular = false, int certified_rank = -1);
int monomial_support_dimension(std::vector<std::vector<int>> supports, int width);
std::optional<int> dimension_power_coordinates(const Basis& ideal, const std::vector<std::string>& names,
                                               const std::vector<mpq_class>& point, bool lex, bool prefer_modular);
}
