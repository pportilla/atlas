#pragma once

#include "groebner.hpp"
#include <flint/fmpq_mpoly.h>

namespace atlas {

struct Exponents {
    std::vector<Z> values;
    std::vector<fmpz*> entries;

    explicit Exponents(slong n) : values(n) {
        for (auto& x : values) entries.push_back(&x.v);
    }

    void read(const fmpq_mpoly_t p, slong term, const fmpq_mpoly_ctx_t ctx) {
        fmpq_mpoly_get_term_exp_fmpz(entries.data(), p, term, ctx);
    }
};

struct PolyValue {
    fmpq_mpoly_t p;
    const fmpq_mpoly_ctx_struct* ctx;
    explicit PolyValue(const fmpq_mpoly_ctx_struct* c) : ctx(c) { fmpq_mpoly_init(p, ctx); }
    ~PolyValue() { fmpq_mpoly_clear(p, ctx); }
    PolyValue(PolyValue&& other) noexcept : PolyValue(other.ctx) {
        fmpq_mpoly_swap(p, other.p, ctx);
    }
    PolyValue(const PolyValue&) = delete;
};

} // namespace atlas
