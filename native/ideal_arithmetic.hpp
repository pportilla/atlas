#pragma once
#include "local_basis.hpp"

namespace atlas::ideal_arithmetic {
// Ordered pairwise polynomial products, retaining multiplicity, scale and zero rows.
local::Basis generator_products(const local::Basis& left, const local::Basis& right, size_t width);
// Unreduced generators over QQ, with empty zero ideals and singleton unit ideals.
local::Basis sum(local::Basis left, local::Basis right);
local::Basis product(const local::Basis& left, const local::Basis& right, size_t width);
}
