#pragma once
#include "local_basis.hpp"
#include <json/value.h>

namespace atlas::local {
// Exact homogeneous pieces of the extended tangent quotient for a binary map.
std::optional<Json::Value> ae_graded_surface(const Basis& map);
}
