#pragma once
#include "flint_polynomial.hpp"
#include "local_basis.hpp"
#include <flint/fmpq.h>

namespace atlas::local {

// Direct interchange between sparse GMP polynomials and FLINT's ordered QQ rings.
struct SparseFlintRing {
    fmpq_mpoly_ctx_t ctx;
    Exponents powers;
    fmpq_t coefficient;

    explicit SparseFlintRing(size_t width, ordering_t order = ORD_LEX) : powers(width) {
        fmpq_mpoly_ctx_init(ctx, width, order);
        fmpq_init(coefficient);
    }
    SparseFlintRing(const SparseFlintRing&) = delete;
    ~SparseFlintRing() { fmpq_clear(coefficient); fmpq_mpoly_ctx_clear(ctx); }

    void write(fmpq_mpoly_t out, const Polynomial& f) {
        fmpq_mpoly_zero(out, ctx);
        fmpq_mpoly_fit_length(out, f.size(), ctx);
        for (auto it = f.rbegin(); it != f.rend(); ++it) {
            fmpq_set_mpq(coefficient, it->second.get_mpq_t());
            for (size_t j = 0; j < powers.entries.size(); ++j)
                fmpz_set_mpz(powers.entries[j], it->first[j].get_mpz_t());
            fmpq_mpoly_push_term_fmpq_fmpz(out, coefficient, powers.entries.data(), ctx);
        }
        // Reverse map iteration already gives descending lexicographic order.
        if (fmpq_mpoly_ctx_ord(ctx) != ORD_LEX) fmpq_mpoly_sort_terms(out,ctx);
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
using SparseLexRing = SparseFlintRing;
}
