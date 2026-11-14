#pragma once

#include <flint/fmpq_mpoly_factor.h>

#include <string>
#include <vector>

namespace atlas {

// Return monic factors in the order of ctx and the corresponding rational unit.
const char* ordered_factors(fmpq_mpoly_factor_t out, const fmpq_mpoly_t polynomial,
                            const fmpq_mpoly_ctx_t ctx);

// A marked field uses one additional, last variable. Its rational lex basis can
// be requested lazily when the divisor really needs arithmetic modulo the field.
std::string factor_multiplicity(const fmpq_mpoly_t value, const fmpq_mpoly_t divisor,
                                slong axes, std::vector<fmpq_mpoly_struct*> field_basis,
                                const fmpq_mpoly_ctx_t ctx);

}
