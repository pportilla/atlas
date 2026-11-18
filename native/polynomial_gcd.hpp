#pragma once
#include <flint/fmpq_mpoly.h>

namespace atlas {
// The monic associate in this context's monomial order, including gcd(0,0)=0.
void monic_polynomial_gcd(fmpq_mpoly_t out, const fmpq_mpoly_t a,
                          const fmpq_mpoly_t b, const fmpq_mpoly_ctx_t ctx);
bool try_gcd_image(fmpq_mpoly_t out, const fmpq_mpoly_t a,
                   const fmpq_mpoly_t b, const fmpq_mpoly_ctx_t ctx);
}
