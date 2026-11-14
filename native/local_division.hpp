#pragma once

#include "local_basis.hpp"
#include <cstddef>
#include <functional>
#include <optional>

namespace atlas::local {

struct NormalForm {
    Polynomial unit;
    Polynomial remainder;
    bool weak;
};

NormalForm normal_form(const Basis& basis, const Polynomial& centered,
                       std::size_t width, bool lex, std::optional<mpz_class> known_cutoff = std::nullopt);
std::optional<mpz_class> finite_cutoff(const std::vector<Powers>& heads, std::size_t width);

// Compute the basis and remainder together when the centered generators admit
// packed homogeneous division. The query arrives in the original coordinates.
using LocalStage = std::function<void(const char* operation, const char* label)>;
std::optional<Polynomial> homogeneous_normal_form(const Basis& centered, const Polynomial& original,
    const std::vector<mpq_class>& point, bool lex, bool prefer_modular = false, const LocalStage& stage = {});

// For a finite quotient, omit all terms in the nilpotent power of the
// maximal ideal. Return nullopt when full translation is preferable.
std::optional<Polynomial> finite_query_jet(const Basis& basis, const Polynomial& original,
                                           const std::vector<mpq_class>& point, bool lex);

// Truncate the Taylor expansion to degrees strictly below cutoff.
Polynomial point_jet_below(const mpz_class& cutoff, const Polynomial& original,
                           const std::vector<mpq_class>& point);

}
