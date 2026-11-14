#pragma once

#include "local_basis.hpp"

namespace atlas::quotient_ring {

// A Bezout inverse when the element and a defining relation use one variable.
// An empty result leaves the general unit test to the caller.
std::optional<local::Polynomial> fast_inverse(const local::Polynomial& element,
                                               const local::Basis& basis, size_t width, bool lex);
std::optional<local::Polynomial> univariate_power(const local::Polynomial& element,
                                                   const local::Basis& basis,
                                                   const mpz_class& exponent);
}
