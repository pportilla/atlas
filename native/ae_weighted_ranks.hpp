#pragma once

#include "local_basis.hpp"
#include <json/value.h>

namespace atlas::local {

// The positive source weights grade every component of the map. The certified
// contact order k means m^k M is contained in (f) M for M = R^p / df R^n.
std::optional<Json::Value> ae_weighted_ranks(const Basis& map, const Powers& weights,
                                            unsigned contact_order);

}
