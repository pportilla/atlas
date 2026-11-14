#pragma once

#include "local_division.hpp"

namespace atlas::local {
NormalForm normal_form_at_point(const Basis& generators, const Polynomial& query,
        const std::vector<std::string>& names, const std::vector<mpq_class>& point,
        bool lex, bool prefer_modular, bool known_basis);

// Mora weak division for standard-basis completion. The divisors need
// not yet form a standard basis. All polynomials use original coordinates.
Polynomial mora_remainder_at_point(const Basis& reducers, const Polynomial& query,
        const std::vector<mpq_class>& point, bool lex);
}
