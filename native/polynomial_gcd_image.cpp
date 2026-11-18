#include "polynomial_gcd.hpp"
#include "flint_polynomial.hpp"
#include <flint/nmod.h>
#include <flint/nmod_poly.h>
#include <array>
#include <numeric>

namespace atlas {
namespace {
constexpr ulong prime = 65537;
constexpr slong max_width = 16, max_box = 8192;

struct Image {
    nmod_poly_t p;
    slong degree = -1, leading = 0;
    Image() { nmod_poly_init(p, prime); }
    ~Image() { nmod_poly_clear(p); }
    Image(const Image&) = delete;
};

// x_i -> t^weight[i]. A dense box uses mixed-radix weights, so its
// monomials have distinct images. The diagonal x_i -> t can certify units.
struct Projection {
    slong width, size = 1;
    bool box = true;
    std::array<slong, max_width> radix{}, weight{};

    bool prepare(const fmpz_mpoly_t a, const fmpz_mpoly_t b, const fmpz_mpoly_ctx_t ctx) {
        if (width < 2 || width > max_width || !a->length || !b->length ||
            a->length > max_box || b->length > max_box ||
            a->bits > FLINT_BITS || b->bits > FLINT_BITS) return false;
        std::array<slong, max_width> left{}, right{};
        fmpz_mpoly_degrees_si(left.data(), a, ctx);
        fmpz_mpoly_degrees_si(right.data(), b, ctx);
        slong total = 0;
        for (slong i = 0; i < width; ++i) {
            const slong degree = std::max(left[i], right[i]);
            if (left[i] < 0 || right[i] < 0 || degree >= max_box) return false;
            total += degree;
            radix[i] = degree + 1;
            weight[i] = size;
            if (box && size <= max_box / radix[i]) size *= radix[i];
            else box = false;
        }
        // A short dense univariate GCD is still cheap with some empty slots.
        const slong slots_per_term = size <= 256 ? 4 : 2;
        if (box && size <= slots_per_term * std::min(a->length, b->length)) return true;
        if (a->length > 64 || b->length > 64 || total > 256) return false;
        box = false;
        weight.fill(1);
        size = total + 1;
        return true;
    }

    bool encode(Image& image, const fmpz_mpoly_t f, const fmpz_mpoly_ctx_t ctx) const {
        nmod_poly_fit_length(image.p, size);
        std::fill_n(image.p->coeffs, size, 0UL);
        std::array<ulong, max_width> powers;
        for (slong i = 0; i < f->length; ++i) {
            const fmpz coeff = f->coeffs[i];
            if (COEFF_IS_MPZ(coeff)) return false;
            fmpz_mpoly_get_term_exp_ui(powers.data(), f, i, ctx);
            const slong k = std::inner_product(weight.begin(), weight.begin() + width, powers.begin(), slong(0));
            if (k > image.degree) {
                image.degree = k;
                image.leading = coeff; // Used only for an injective box.
            }
            slong residue = coeff % static_cast<slong>(prime);
            if (residue < 0) residue += prime;
            ulong sum = image.p->coeffs[k] + residue;
            image.p->coeffs[k] = sum >= prime ? sum - prime : sum;
        }
        image.p->length = image.degree + 1;
        _nmod_poly_normalise(image.p);
        return true;
    }

    bool lift(fmpq_mpoly_t h, const nmod_poly_t image, ulong scale, const fmpq_mpoly_ctx_t ctx) const {
        // A single centered lift is speculative. Bound the work before exact divisions.
        if (!scale) return false;
        std::array<ulong, max_width> powers;
        for (slong k = 0; k < image->length; ++k) {
            if (!image->coeffs[k]) continue;
            if (h->zpoly->length >= 512) return false;
            for (slong j = 0; j < width; ++j) powers[j] = (k / weight[j]) % radix[j];
            slong coeff = nmod_mul(image->coeffs[k], scale, image->mod);
            if (coeff > static_cast<slong>(prime / 2)) coeff -= static_cast<slong>(prime);
            fmpz_mpoly_push_term_si_ui(h->zpoly, coeff, powers.data(), ctx->zctx);
        }
        fmpz_mpoly_sort_terms(h->zpoly, ctx->zctx);
        fmpq_one(h->content);
        fmpq_mpoly_reduce(h, ctx);
        return true;
    }
};
}

bool try_gcd_image(fmpq_mpoly_t out, const fmpq_mpoly_t a, const fmpq_mpoly_t b,
                   const fmpq_mpoly_ctx_t ctx) {
    Projection map{fmpq_mpoly_ctx_nvars(ctx)};
    if (!map.prepare(a->zpoly, b->zpoly, ctx->zctx)) return false;
    Image f, g, h;
    if (!map.encode(f, a->zpoly, ctx->zctx) || !map.encode(g, b->zpoly, ctx->zctx)) return false;
    // If one primitive input keeps its weighted degree modulo p, so must each
    // of its factors: the degrees add in a product and can only drop under the
    // substitution. The modular univariate GCD degree then bounds the weighted
    // degree of every common divisor. All weights are positive.
    if (f.degree != nmod_poly_degree(f.p) && g.degree != nmod_poly_degree(g.p)) return false;
    nmod_poly_gcd(h.p, f.p, g.p);
    if (nmod_poly_degree(h.p) == 0) {
        fmpq_mpoly_one(out, ctx);
        return true;
    }
    if (!map.box) return false;
    // The primitive GCD's encoded leading coefficient divides both leading
    // coefficients. If the lift divides both inputs, its degree equals the
    // bound (the mixed-radix decoding preserves the leading nonzero term).
    const ulong scale = std::gcd(f.leading, g.leading) % prime;
    PolyValue candidate(ctx), quotient(ctx);
    if (!map.lift(candidate.p, h.p, scale, ctx) ||
        !fmpq_mpoly_divides(quotient.p, a, candidate.p, ctx) ||
        !fmpq_mpoly_divides(quotient.p, b, candidate.p, ctx)) return false;
    fmpq_mpoly_make_monic(out, candidate.p, ctx);
    return true;
}
} // namespace atlas
