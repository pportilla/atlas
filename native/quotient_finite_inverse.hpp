#pragma once

#include "local_basis.hpp"

namespace atlas::quotient_ring {

// Decide invertibility for small tensor products of univariate monomial or
// binomial quotients. A false return leaves the decision to the general kernel.
// When true, result contains the canonical inverse or is empty for a nonunit.
bool finite_inverse(const local::Basis& basis, const std::vector<local::Powers>& heads,
                    const local::Polynomial& element, size_t width,
                    std::optional<local::Polynomial>& result);

} // namespace atlas::quotient_ring
