#pragma once

#include "rational_sparse_json.hpp"

namespace atlas::monodromy {

// Zeta functions on generic codimension-two coordinate strata where exactly
// two simple smooth branches meet with finite contact.
Json::Value transverse_contacts(const sparse_json::Terms& f, size_t width);

}
