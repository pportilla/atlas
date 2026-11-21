#pragma once
#include "local_basis.hpp"
#include "flint_polynomial.hpp"
#include <flint/fmpq_mpoly_factor.h>

namespace atlas {
bool triton_roots_active();
bool triton_linear_factors(fmpq_mpoly_factor_t out, const fmpq_mpoly_t f, const fmpq_mpoly_ctx_t ctx,
                           bool squarefree = false);
bool triton_common_linear_factors(fmpq_mpoly_t common, fmpq_mpoly_t a, fmpq_mpoly_t b,
                                  const fmpq_mpoly_ctx_t ctx);
// A binomial gives an independent geometric progression for each input monomial.
bool triton_binomial_division(local::Polynomial* quotient, local::Polynomial& remainder,
                              const local::Polynomial& f, const local::Polynomial& divisor, bool lex);
} // namespace atlas
