#pragma once

#include "rational_sparse_json.hpp"

namespace atlas::monodromy {

// Decide membership in the nearby Milnor eigenvalue set for rational
// pole values. Each input string keys a root (or null) and the tested shifts.
Json::Value local_eigenvalues(const sparse_json::Terms& terms, std::size_t width,
                              const Json::Value& values);

}  // namespace atlas::monodromy
