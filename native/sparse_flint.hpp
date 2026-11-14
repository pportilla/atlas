#pragma once
#include "flint_polynomial.hpp"
#include "local_basis.hpp"
#include <flint/fmpq.h>

namespace atlas::local {

// Direct interchange between sparse GMP polynomials and FLINT's QQ lex ring.
// The two representations have opposite term iteration order.
struct SparseLexRing {
    fmpq_mpoly_ctx_t ctx;
    Exponents powers;
    fmpq_t coefficient;

    explicit SparseLexRing(size_t width) : powers(width) {
        fmpq_mpoly_ctx_init(ctx, width, ORD_LEX);
        fmpq_init(coefficient);
    }
    SparseLexRing(const SparseLexRing&) = delete;
    ~SparseLexRing() { fmpq_clear(coefficient); fmpq_mpoly_ctx_clear(ctx); }

    void write(fmpq_mpoly_t out, const Polynomial& f) {
        fmpq_mpoly_zero(out, ctx);
        fmpq_mpoly_fit_length(out, f.size(), ctx);
        for (auto it = f.rbegin(); it != f.rend(); ++it) {
            fmpq_set_mpq(coefficient, it->second.get_mpq_t());
            for (size_t j = 0; j < powers.entries.size(); ++j)
                fmpz_set_mpz(powers.entries[j], it->first[j].get_mpz_t());
            fmpq_mpoly_push_term_fmpq_fmpz(out, coefficient, powers.entries.data(), ctx);
        }
        fmpq_mpoly_reduce(out, ctx);
    }

    Polynomial read(const fmpq_mpoly_t f) {
        Polynomial result;
        for (slong i = 0; i < fmpq_mpoly_length(f, ctx); ++i) {
            powers.read(f, i, ctx);
            Powers e(powers.entries.size());
            for (size_t j = 0; j < e.size(); ++j)
                fmpz_get_mpz(e[j].get_mpz_t(), powers.entries[j]);
            fmpq_mpoly_get_term_coeff_fmpq(coefficient, f, i, ctx);
            auto term = result.try_emplace(result.begin(), std::move(e));
            fmpq_get_mpq(term->second.get_mpq_t(), coefficient);
        }
        return result;
    }
};
}
