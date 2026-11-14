#pragma once

#include "local_basis.hpp"
#include <cstddef>
#include <optional>

namespace atlas::local {

struct NormalForm {
    Polynomial unit;
    Polynomial remainder;
    bool weak;
};

NormalForm normal_form(const Basis& basis, const Polynomial& centered,
                       std::size_t width, bool lex);

// For a finite quotient, omit all terms in the nilpotent power of the
// maximal ideal. Return nullopt when full translation is preferable.
std::optional<Polynomial> finite_query_jet(const Basis& basis, const Polynomial& original,
                                           const std::vector<mpq_class>& point, bool lex);

}
