#pragma once

#include <flint/fmpq_mpoly_factor.h>
#include "local_basis.hpp"

#include <string>
#include <vector>

namespace atlas {

std::vector<std::pair<local::Polynomial, mpz_class>> marked_factors(
    const local::Polynomial& f, const local::Polynomial& modulus, size_t width);

// Monic gcd of several polynomials in the same marked coefficient field.
local::Polynomial marked_gcd(const local::Basis& rows, const local::Polynomial& modulus, size_t width);

// Cancel the common monic geometric polynomial over the chart's marked field.
local::Basis marked_cofactors(const local::Polynomial& a,const local::Polynomial& b,
                              const local::Polynomial& modulus,size_t width);

// Return monic factors in the order of ctx and the corresponding rational unit.
const char* ordered_factors(fmpq_mpoly_factor_t out, const fmpq_mpoly_t polynomial,
                            const fmpq_mpoly_ctx_t ctx);

void rational_squarefree_part(fmpq_mpoly_t out, const fmpq_mpoly_t f, const fmpq_mpoly_ctx_t ctx);

// A marked field uses one additional, last variable. Its rational lex basis can
// be requested lazily when the divisor really needs arithmetic modulo the field.
std::string factor_multiplicity(const fmpq_mpoly_t value, const fmpq_mpoly_t divisor,
                                slong axes, std::vector<fmpq_mpoly_struct*> field_basis,
                                const fmpq_mpoly_ctx_t ctx);

}
