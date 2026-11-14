#include "quotient_inverse.hpp"
#include "sparse_flint.hpp"
#include <algorithm>
#include <flint/fmpq.h>
#include <flint/fmpq_poly.h>

namespace atlas::quotient_ring {
namespace {
struct Dense {
    fmpq_poly_t value;
    Dense() { fmpq_poly_init(value); }
    ~Dense() { fmpq_poly_clear(value); }
    Dense(const Dense&) = delete;
};

struct Rational {
    fmpq_t value;
    Rational() { fmpq_init(value); }
    ~Rational() { fmpq_clear(value); }
    Rational(const Rational&) = delete;
};

// Sparse inputs of very high degree belong on the sparse arithmetic path.
bool dense_univariate(const local::Polynomial& row, size_t variable, Dense& out) {
    for (const auto& term : row) {
        const auto& powers = term.first;
        if (powers[variable] > 8192) return false;
        for (size_t i = 0; i < powers.size(); ++i)
            if (i != variable && powers[i] != 0) return false;
    }
    Rational q;
    for (const auto& [powers, coefficient] : row) {
        fmpq_set_mpq(q.value, coefficient.get_mpq_t());
        fmpq_poly_set_coeff_fmpq(out.value, powers[variable].get_si(), q.value);
    }
    return true;
}

local::Polynomial sparse(const Dense& dense, size_t variable, size_t width) {
    local::Polynomial result;
    Rational q;
    local::Powers powers(width);
    for (slong degree = 0; degree < fmpq_poly_length(dense.value); ++degree) {
        fmpq_poly_get_coeff_fmpq(q.value, dense.value, degree);
        if (fmpq_is_zero(q.value)) continue;
        powers[variable] = degree;
        fmpq_get_mpq(result[powers].get_mpq_t(), q.value);
    }
    return result;
}

// If g = a*q + c is a defining relation and c is a nonzero scalar,
// then -q/c is a Bezout inverse. Lex division suffices for this shortcut.
std::optional<local::Polynomial> relation_inverse(const local::Polynomial& element,
                                                   const local::Basis& basis, size_t width) {
    const local::Powers origin(width);
    local::SparseLexRing ring(width);
    PolyValue divisor(ring.ctx), relation(ring.ctx), quotient(ring.ctx), remainder(ring.ctx);
    ring.write(divisor.p, element);
    for (const auto& row : basis) {
        if (!row.count(origin) && !element.count(origin)) continue;
        // This is a shortcut. Avoid expanding a huge quotient just to find
        // whether its remainder happens to be a constant.
        if (row.size() > 64 || width > 8 ||
            std::any_of(row.begin(), row.end(), [](const auto& term) {
                return std::any_of(term.first.begin(), term.first.end(), [](const auto& k) { return k > 32; });
            })) continue;
        ring.write(relation.p, row);
        fmpq_mpoly_divrem(quotient.p, remainder.p, relation.p, divisor.p, ring.ctx);
        if (fmpq_mpoly_is_zero(remainder.p, ring.ctx) || !fmpq_mpoly_is_fmpq(remainder.p, ring.ctx))
            continue;
        Rational c;
        fmpq_mpoly_get_fmpq(c.value, remainder.p, ring.ctx);
        fmpq_neg(c.value, c.value);
        fmpq_mpoly_scalar_div_fmpq(quotient.p, quotient.p, c.value, ring.ctx);
        return ring.read(quotient.p);
    }
    return std::nullopt;
}
} // namespace

std::optional<local::Polynomial> fast_inverse(const local::Polynomial& element,
                                               const local::Basis& basis, size_t width, bool) {
    size_t variable = width;
    for (const auto& term : element) {
        for (size_t i = 0; i < width; ++i) {
            if (term.first[i] == 0) continue;
            if (variable != width && variable != i) return relation_inverse(element, basis, width);
            variable = i;
        }
    }
    if (variable == width) return std::nullopt;
    Dense a;
    if (!dense_univariate(element, variable, a)) return std::nullopt;
    for (const auto& relation : basis) {
        Dense modulus, gcd, inverse, cofactor;
        if (!dense_univariate(relation, variable, modulus) || fmpq_poly_degree(modulus.value) < 1)
            continue;
        fmpq_poly_xgcd(gcd.value, inverse.value, cofactor.value, a.value, modulus.value);
        if (!fmpq_poly_is_one(gcd.value)) continue;

        return sparse(inverse, variable, width);
    }
    return std::nullopt;
}

// In a univariate quotient, FLINT keeps one denominator for the entire
// polynomial. Reduce each square and product before the next exponent bit.
std::optional<local::Polynomial> univariate_power(const local::Polynomial& element,
                                                   const local::Basis& basis,
                                                   const mpz_class& exponent) {
    if (basis.size() != 1 || exponent < 2) return std::nullopt;
    Dense modulus, power, result, product;
    if (!dense_univariate(basis.front(), 0, modulus) ||
        !dense_univariate(element, 0, power) || fmpq_poly_degree(modulus.value) < 1)
        return std::nullopt;
    fmpq_poly_set(result.value, power.value);
    for (size_t bit = mpz_sizeinbase(exponent.get_mpz_t(), 2) - 1; bit > 0; --bit) {
        fmpq_poly_mul(product.value, result.value, result.value);
        fmpq_poly_rem(result.value, product.value, modulus.value);
        if (mpz_tstbit(exponent.get_mpz_t(), bit - 1)) {
            fmpq_poly_mul(product.value, result.value, power.value);
            fmpq_poly_rem(result.value, product.value, modulus.value);
        }
        if (fmpq_poly_is_zero(result.value)) break;
    }
    return sparse(result, 0, 1);
}
} // namespace atlas::quotient_ring
