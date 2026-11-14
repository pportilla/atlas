#pragma once

#include "local_basis.hpp"
#include <json/value.h>

namespace atlas::local {
// Exact graph elimination for a map admitting a positive source grading.
std::optional<Json::Value> ae_graded(const Basis& components, const std::vector<std::string>& names);
}
