#pragma once
#include "local_basis.hpp"

namespace atlas::ideal_powers {
// A fully expanded monic generator, retained in FLINT's sparse representation
// until the caller formats it or requests a GMP polynomial for more arithmetic.
class PrincipalBasis {
    struct Storage;
    std::unique_ptr<Storage> storage;
public:
    PrincipalBasis(const local::Polynomial& f,size_t width,bool lex,
                   const mpz_class& exponent,bool symbolic);
    ~PrincipalBasis();
    PrincipalBasis(PrincipalBasis&&) noexcept;
    PrincipalBasis& operator=(PrincipalBasis&&) noexcept;
    local::Basis materialize() const;
    std::vector<std::string> expressions(const std::vector<std::string>& names) const;
};
}
