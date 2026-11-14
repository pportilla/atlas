#pragma once

#include "local_basis.hpp"
#include <json/value.h>

namespace atlas::local {
// The components are ordered and centered, and all belong to the same source ring.
std::optional<Json::Value> ae_special(const Basis& components, const std::vector<std::string>& names);
}
