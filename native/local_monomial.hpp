#pragma once

#include "local_basis.hpp"
#include <algorithm>
#include <cstddef>

namespace atlas::local {

// Both monomials belong to the same coordinate ring.
inline bool monomial_divides(const Powers& divisor, const Powers& target) {
    for (std::size_t i = 0; i < divisor.size(); ++i)
        if (divisor[i] > target[i]) return false;
    return true;
}

inline mpz_class total_degree(const Powers& powers) {
    mpz_class sum = 0;
    for (const auto& exponent : powers) sum += exponent;
    return sum;
}

// The local algorithms use zero as the degree bound for the empty polynomial.
inline mpz_class max_total_degree(const Polynomial& polynomial) {
    mpz_class largest = 0;
    for (const auto& term : polynomial)
        largest = std::max(largest, total_degree(term.first));
    return largest;
}

}
