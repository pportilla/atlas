#pragma once

#include "local_basis.hpp"
#include <json/value.h>

namespace atlas::local {
// A certified contact order permits a finite-generation test on weighted pieces.
std::optional<Json::Value> ae_graded(const Basis& components, const std::vector<std::string>& names,
                                     std::optional<unsigned> contact_order = std::nullopt);
}
