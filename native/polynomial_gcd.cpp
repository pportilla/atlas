#include "polynomial_gcd.hpp"
#include "flint_polynomial.hpp"
#include <flint/mpoly.h>

namespace atlas {
namespace {
// A leading-term cancellation sometimes gives the GCD itself. Every common
// factor divides the difference, so one exact division certifies maximality.
// Scalar contents of the inputs are units over QQ.
bool try_difference(fmpq_mpoly_t out, const fmpq_mpoly_t a, const fmpq_mpoly_t b,
                    const fmpq_mpoly_ctx_t ctx) {
    const auto* f = a->zpoly;
    const auto* g = b->zpoly;
    if (f->bits != g->bits || !mpoly_monomial_equal(f->exps,g->exps,
            mpoly_words_per_exp(f->bits,ctx->zctx->minfo))) return false;
    PolyValue d(ctx);
    if (fmpz_equal(f->coeffs,g->coeffs)) {
        fmpz_mpoly_sub(d.p->zpoly,f,g,ctx->zctx);
    } else {
        Z minus_lc;
        fmpz_neg(&minus_lc.v,f->coeffs);
        fmpz_mpoly_scalar_fmma(d.p->zpoly,f,g->coeffs,g,&minus_lc.v,ctx->zctx);
    }
    if (!d.p->zpoly->length) {
        fmpq_mpoly_make_monic(out,a,ctx);
        return true;
    }
    if (d.p->zpoly->length > std::min(f->length,g->length) / 2) return false;
    fmpq_one(d.p->content);
    fmpq_mpoly_reduce(d.p,ctx);
    PolyValue quotient(ctx);
    if (!fmpq_mpoly_divides(quotient.p,a,d.p,ctx)) return false;
    fmpq_mpoly_make_monic(out,d.p,ctx);
    return true;
}

// For two exact squares, each irreducible multiplicity in their GCD is
// twice its multiplicity in the GCD of the roots.
bool try_squares(fmpq_mpoly_t out, const fmpq_mpoly_t a, const fmpq_mpoly_t b,
                 const fmpq_mpoly_ctx_t ctx) {
    PolyValue x(ctx), y(ctx);
    if (!fmpq_mpoly_sqrt(x.p,a,ctx) || !fmpq_mpoly_sqrt(y.p,b,ctx)) return false;
    if (!try_gcd_image(out,x.p,y.p,ctx) && !fmpq_mpoly_gcd(out,x.p,y.p,ctx)) return false;
    fmpq_mpoly_mul(out,out,out,ctx);
    return true;
}
}

void monic_polynomial_gcd(fmpq_mpoly_t out, const fmpq_mpoly_t a,
                          const fmpq_mpoly_t b, const fmpq_mpoly_ctx_t ctx) {
    const auto na = a->zpoly->length, nb = b->zpoly->length;
    if (na > 1 && nb > 1) {
        // Bound speculative divisions and square-root extraction. High-degree
        // sparse operands retain FLINT's general GCD algorithms.
        if (std::max(na,nb) <= 256 && fmpq_mpoly_ctx_nvars(ctx) <= 8 &&
            a->zpoly->bits <= FLINT_BITS && b->zpoly->bits <= FLINT_BITS &&
            fmpq_mpoly_total_degree_fits_si(a,ctx) && fmpq_mpoly_total_degree_fits_si(b,ctx)) {
            const auto da = fmpq_mpoly_total_degree_si(a,ctx);
            const auto db = fmpq_mpoly_total_degree_si(b,ctx);
            const auto degree = std::max(da,db);
            if (degree <= 16 && try_difference(out,a,b,ctx)) return;
            if (degree <= 8 && !(da % 2) && !(db % 2) && try_squares(out,a,b,ctx)) return;
        }
        if (try_gcd_image(out,a,b,ctx)) return;
    }
    // FLINT returns a monic rational GCD for every nonzero result.
    if (!fmpq_mpoly_gcd(out,a,b,ctx)) throw Limit("FLINT could not compute the polynomial gcd.");
}
}
