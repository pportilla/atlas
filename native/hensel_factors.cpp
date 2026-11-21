// Lift monic coprime polynomial factors from F_p[x] to (Z/p^2Z)[x].
#include "local_io.hpp"
#include "rational_sparse_json.hpp"
#include "sparse_flint.hpp"
#include <flint/fmpz_mod_mpoly.h>
#include <flint/fmpz_mod_poly.h>
#include <optional>

namespace atlas {
namespace {
using local::Polynomial;
using local::Powers;

mpz_class residue(const mpz_class& value, const mpz_class& modulus) {
    mpz_class reduced;
    mpz_mod(reduced.get_mpz_t(), value.get_mpz_t(), modulus.get_mpz_t());
    return reduced;
}

Polynomial reduced(const Polynomial& input, const mpz_class& modulus) {
    Polynomial result;
    for (const auto& [e, c] : input) {
        if (c.get_den() != 1)
            throw std::invalid_argument("Hensel lifting requires integer polynomial coefficients.");
        auto r = residue(c.get_num(), modulus);
        if (r != 0)
            result.emplace(e, std::move(r));
    }
    return result;
}

bool is_monic(const Polynomial& f) { return !f.empty() && f.rbegin()->second == 1; }

// All field arithmetic is univariate, even when the session parent has more coordinates.
Polynomial project(const Polynomial& f, size_t axis, size_t width) {
    Polynomial result;
    for (const auto& [e, c] : f)
        result.emplace(Powers{width ? e[axis] : mpz_class(0)}, c);
    return result;
}

Polynomial restore(const Polynomial& f, size_t axis, size_t width) {
    Polynomial result;
    for (const auto& [e, c] : f) {
        Powers exponent(width);
        if (width)
            exponent[axis] = e[0];
        result.emplace(std::move(exponent), c);
    }
    return result;
}

struct ModRing {
    fmpz_mod_mpoly_ctx_t ctx;
    explicit ModRing(const mpz_class& p) {
        Z modulus;
        fmpz_set_mpz(&modulus.v, p.get_mpz_t());
        fmpz_mod_mpoly_ctx_init(ctx, 1, ORD_LEX, &modulus.v);
    }
    ~ModRing() { fmpz_mod_mpoly_ctx_clear(ctx); }
};

struct ModPoly {
    fmpz_mod_mpoly_t f;
    const fmpz_mod_mpoly_ctx_struct* ctx;
    explicit ModPoly(const ModRing& ring) : ctx(ring.ctx) { fmpz_mod_mpoly_init(f, ctx); }
    ModPoly(const ModPoly&) = delete;
    ~ModPoly() { fmpz_mod_mpoly_clear(f, ctx); }
    void set(const Polynomial& poly) {
        fmpz_mod_mpoly_zero(f, ctx);
        Z exponent, coefficient;
        for (auto it = poly.rbegin(); it != poly.rend(); ++it) {
            fmpz_set_mpz(&exponent.v, it->first[0].get_mpz_t());
            fmpz_set_mpz(&coefficient.v, mpq_numref(it->second.get_mpq_t()));
            fmpz_mod_mpoly_push_term_fmpz_ffmpz(f, &coefficient.v, &exponent.v, ctx);
        }
    }
    Polynomial read() const {
        Polynomial result;
        Z exponent, coefficient;
        fmpz* exps[] = {&exponent.v};
        for (slong i = 0; i < f->length; ++i) {
            fmpz_mod_mpoly_get_term_exp_fmpz(exps, f, i, ctx);
            fmpz_mod_mpoly_get_term_coeff_fmpz(&coefficient.v, f, i, ctx);
            mpz_class e, c;
            fmpz_get_mpz(e.get_mpz_t(), &exponent.v);
            fmpz_get_mpz(c.get_mpz_t(), &coefficient.v);
            result.emplace(Powers{std::move(e)}, std::move(c));
        }
        return result;
    }
};

// Euclid on sparse F_p[x] supports arbitrarily large exponents. The inverse is
// only computed modulo g, which keeps the necessary Bézout coefficient bounded.
void inverse_mod(ModPoly& inverse, const ModPoly& h, const ModPoly& g, const ModRing& ring) {
    const auto& ctx = ring.ctx;
    if (fmpz_mod_mpoly_is_one(g.f, ctx)) {
        fmpz_mod_mpoly_zero(inverse.f, ctx);
        return;
    }
    auto dense = [&](const ModPoly& polynomial) {
        if (!fmpz_mod_mpoly_degrees_fit_si(polynomial.f, ctx))
            return false;
        const slong degree = fmpz_mod_mpoly_degree_si(polynomial.f, 0, ctx);
        return degree <= 65536 && degree / 4 <= polynomial.f->length;
    };
    if (dense(g) && dense(h)) {
        // FLINT specializes dense univariate Euclid, including half-GCD at large degree.
        const auto& field = ctx->ffinfo;
        fmpz_mod_poly_t a, b, i;
        fmpz_mod_poly_init(a, field);
        fmpz_mod_poly_init(b, field);
        fmpz_mod_poly_init(i, field);
        fmpz_mod_mpoly_get_fmpz_mod_poly(a, h.f, 0, ctx);
        fmpz_mod_mpoly_get_fmpz_mod_poly(b, g.f, 0, ctx);
        fmpz_mod_poly_rem(a, a, b, field);
        bool coprime = !fmpz_mod_poly_is_zero(a, field) && fmpz_mod_poly_invmod(i, a, b, field);
        if (coprime)
            fmpz_mod_mpoly_set_fmpz_mod_poly(inverse.f, i, 0, ctx);
        fmpz_mod_poly_clear(a, field);
        fmpz_mod_poly_clear(b, field);
        fmpz_mod_poly_clear(i, field);
        if (!coprime)
            throw std::invalid_argument("The two Hensel factors must be coprime modulo the prime.");
        return;
    }
    ModPoly old(ring), r(ring), s(ring), t(ring), q(ring), next(ring), tmp(ring);
    fmpz_mod_mpoly_set(old.f, h.f, ctx);
    fmpz_mod_mpoly_set(r.f, g.f, ctx);
    fmpz_mod_mpoly_one(s.f, ctx);
    while (!fmpz_mod_mpoly_is_zero(r.f, ctx)) {
        fmpz_mod_mpoly_divrem(q.f, next.f, old.f, r.f, ctx);
        fmpz_mod_mpoly_swap(old.f, r.f, ctx);
        fmpz_mod_mpoly_swap(r.f, next.f, ctx);
        fmpz_mod_mpoly_mul(tmp.f, q.f, t.f, ctx);
        fmpz_mod_mpoly_sub(tmp.f, s.f, tmp.f, ctx);
        fmpz_mod_mpoly_swap(s.f, t.f, ctx);
        fmpz_mod_mpoly_swap(t.f, tmp.f, ctx);
    }
    if (!fmpz_mod_mpoly_is_fmpz(old.f, ctx))
        throw std::invalid_argument("The two Hensel factors must be coprime modulo the prime.");
    Z c;
    fmpz_mod_mpoly_get_fmpz(&c.v, old.f, ctx);
    if (!fmpz_invmod(&c.v, &c.v, fmpz_mod_mpoly_ctx_modulus(ctx)))
        throw std::invalid_argument("The two Hensel factors must be coprime modulo the prime.");
    fmpz_mod_mpoly_scalar_mul_fmpz(inverse.f, s.f, &c.v, ctx);
}

Polynomial cpu_product(const Polynomial& a, const Polynomial& b) {
    local::SparseFlintRing ring(1);
    PolyValue x(ring.ctx), y(ring.ctx), product(ring.ctx);
    ring.write(x.p, a);
    ring.write(y.p, b);
    fmpq_mpoly_mul(product.p, x.p, y.p, ring.ctx);
    return ring.read(product.p);
}

std::optional<Polynomial> error_polynomial(const Polynomial& f, const Polynomial& product,
                                           const mpz_class& p) {
    Polynomial difference = f, error;
    for (const auto& [e, c] : product)
        difference[e] -= c;
    for (const auto& [e, c] : difference) {
        const auto value = c.get_num();
        if (c.get_den() != 1 || !mpz_divisible_p(value.get_mpz_t(), p.get_mpz_t()))
            return std::nullopt;
        auto corrected = residue(value / p, p);
        if (corrected != 0)
            error.emplace(e, std::move(corrected));
    }
    return error;
}

using Lift = std::pair<Polynomial, Polynomial>;

Lift lift(const Polynomial& g, const Polynomial& h, const Polynomial& error, const mpz_class& p,
          const ModRing& ring, const ModPoly& G, const ModPoly& H, const ModPoly& inverse) {
    ModPoly e(ring), correction_g(ring), correction_h(ring), product(ring), q(ring);
    e.set(error);
    fmpz_mod_mpoly_mul(product.f, inverse.f, e.f, ring.ctx);
    fmpz_mod_mpoly_divrem(q.f, correction_g.f, product.f, G.f, ring.ctx);
    fmpz_mod_mpoly_mul(product.f, H.f, correction_g.f, ring.ctx);
    fmpz_mod_mpoly_sub(product.f, e.f, product.f, ring.ctx);
    if (!fmpz_mod_mpoly_divides(correction_h.f, product.f, G.f, ring.ctx))
        throw std::logic_error("The Hensel correction did not satisfy its finite-field equation.");
    Lift answer{g, h};
    for (const auto& [ex, c] : correction_g.read())
        answer.first[ex] += p * c;
    for (const auto& [ex, c] : correction_h.read())
        answer.second[ex] += p * c;
    return answer;
}

void validate_monic_degrees(const Polynomial& f, const Polynomial& g, const Polynomial& h) {
    if (!is_monic(g) || !is_monic(h))
        throw std::invalid_argument("Both Hensel factors must be monic modulo the prime.");
    if (!is_monic(f) || f.rbegin()->first[0] != g.rbegin()->first[0] + h.rbegin()->first[0])
        throw std::invalid_argument("The target modulo the squared prime must be monic with degree equal "
                                    "to the sum of the factor degrees.");
}

Json::Value hensel(const Json::Value& request) {
    const auto op = request["operation"].asString();
    const size_t width = request["width"].asUInt();
    const mpz_class p(request["prime"].asString(), 10);
    if (p < 2)
        throw std::invalid_argument("The Hensel modulus must be a positive prime.");
    const mpz_class modulus = p * p;
    Json::Value result;
    Z prime;
    fmpz_set_mpz(&prime.v, p.get_mpz_t());
    if (!fmpz_is_prime(&prime.v))
        throw std::invalid_argument("The Hensel modulus must be prime.");
    if (op != "prepare" && op != "compute" && op != "lift")
        throw std::invalid_argument("Unknown polynomial Hensel operation.");
    const size_t input_width = op == "lift" ? 1 : width;
    auto f = reduced(local::decode_row(request["polynomial"].asString(), input_width), modulus);
    auto g = reduced(local::decode_row(request["left"].asString(), input_width), p);
    auto h = reduced(local::decode_row(request["right"].asString(), input_width), p);
    size_t axis = 0;
    if (op != "lift") {
        std::optional<size_t> active;
        for (const auto* poly : {&f, &g, &h})
            for (const auto& [e, c] : *poly)
                for (size_t j = 0; j < width; ++j)
                    if (e[j] != 0) {
                        if (active && *active != j)
                            throw std::invalid_argument(
                                "The Hensel target and factors must have one common variable.");
                        active = j;
                    }
        axis = active.value_or(0);
        f = project(f, axis, width);
        g = project(g, axis, width);
        h = project(h, axis, width);
    } else {
        axis = request["axis"].asUInt();
        if (width && axis >= width)
            throw std::invalid_argument("Invalid Hensel coordinate.");
    }
    validate_monic_degrees(f, g, h);
    if (op == "prepare") {
        result["axis"] = static_cast<Json::UInt>(axis);
        result["polynomial"] = local::encode_row(f);
        result["left"] = local::encode_row(g);
        result["right"] = local::encode_row(h);
        return result;
    }
    auto product = op == "compute" ? cpu_product(g, h) : local::decode_row(request["product"].asString(), 1);
    ModRing ring(p);
    ModPoly G(ring), H(ring), inverse(ring);
    G.set(g);
    H.set(h);
    inverse_mod(inverse, H, G, ring);
    for (int attempt = 0; attempt < 2; ++attempt) {
        auto error = error_polynomial(f, product, p);
        if (error) {
            auto [a, b] = lift(g, h, *error, p, ring, G, H, inverse);
            // This independent CPU product detects incorrect device coefficients
            // even when their difference is divisible by p. Residues lie in [0,p²).
            if (is_monic(a) && is_monic(b) && a.rbegin()->first == g.rbegin()->first &&
                b.rbegin()->first == h.rbegin()->first && reduced(cpu_product(a, b), modulus) == f) {
                result["left"] = local::encode_row(restore(a, axis, width));
                result["right"] = local::encode_row(restore(b, axis, width));
                result["modulus"] = modulus.get_str();
                result["recovered"] = attempt != 0;
                return result;
            }
        }
        if (op == "lift" && attempt == 0) {
            auto exact = cpu_product(g, h);
            if (exact != product) {
                product.swap(exact);
                continue;
            }
        }
        throw std::invalid_argument(
            "The supplied Hensel factors do not multiply to the target modulo the prime.");
    }
    throw std::logic_error("Unable to certify the Hensel factor lift.");
}
} // namespace
} // namespace atlas
extern "C" const char* atlas_hensel_factors(const char* request) noexcept {
    return atlas::sparse_json::invoke(request, atlas::hensel);
}
