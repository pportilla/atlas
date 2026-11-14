#pragma once

#include "flint_polynomial.hpp"

namespace atlas {

void lowest_form(fmpq_mpoly_t out, const fmpq_mpoly_t f, const std::vector<slong>& axes,
                 const fmpq_mpoly_ctx_t ctx);
std::string support_profile(const fmpq_mpoly_t f, const fmpq_mpoly_ctx_t ctx);
std::string singular_subspaces(const fmpq_mpoly_t f, const std::vector<std::vector<slong>>& axes,
                               const fmpq_mpoly_ctx_t ctx);
std::string repeated_directions(const fmpq_mpoly_t f, slong first, slong second,
                                const fmpq_mpoly_ctx_t ctx);
std::vector<PolyValue> reduced_minors(const std::vector<fmpq_mpoly_struct*>& matrix, slong rows,
                                      slong rank, std::vector<fmpq_mpoly_struct*> basis,
                                      const fmpq_mpoly_ctx_t ctx);

}
