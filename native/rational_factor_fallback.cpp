// A finite rational factor search for sparse polynomials outside FLINT's
// factorization range. Gauss's lemma reduces to primitive integer factors.
#include <gmpxx.h>
#include "polynomial_factors.hpp"
#include "polynomial_gcd.hpp"
#include "primary_decomposition.hpp"
#include "sparse_flint.hpp"
#include <algorithm>
#include <map>

namespace atlas {
namespace {
using local::Polynomial;
using local::Powers;
using Factors = std::map<Polynomial, mpz_class>;

Polynomial primitive(Polynomial f) {
    mpz_class denominator = 1, content = 0;
    for (const auto& [e, c] : f)
        denominator = lcm(denominator, c.get_den());
    for (auto& [e, c] : f) {
        c *= denominator;
        content = gcd(content, c.get_num());
    }
    if (f.rbegin()->second < 0) content = -content;
    for (auto& [e, c] : f)
        c /= content;
    return f;
}

void add_monic(Factors& result, Polynomial f, const mpz_class& power) {
    const auto leading = f.rbegin()->second;
    for (auto& [e, c] : f)
        c /= leading;
    result[std::move(f)] += power;
}

// X^m+h is Eisenstein at any prime occurring exactly once in h. Certify
// existence by a coordinate valuation of one or by gcd(h,partial h)=1.
bool eisenstein(const Polynomial& f, const Powers& degrees, local::SparseLexRing& ring) {
    const size_t n = degrees.size();
    for (size_t axis = 0; axis < n; ++axis) {
        if (!degrees[axis]) continue;
        Powers head(n);
        head[axis] = degrees[axis];
        const auto leading = f.find(head);
        if (leading == f.end() || std::any_of(f.begin(), f.end(), [&](const auto& term) {
                return term.first != head && term.first[axis] != 0;
            }))
            continue;
        if (head[axis] == 1) return true;
        Polynomial constant = f;
        constant.erase(head);
        if (constant.empty() || (constant.size() == 1 && constant.begin()->first == Powers(n))) continue;
        PolyValue h(ring.ctx), derivative(ring.ctx), common(ring.ctx);
        ring.write(h.p, constant);
        for (size_t j = 0; j < n; ++j) {
            if (!degrees[j] || j == axis) continue;
            auto min =
                std::min_element(constant.begin(), constant.end(),
                                 [&](const auto& a, const auto& b) { return a.first[j] < b.first[j]; });
            if (min->first[j] == 1) return true;
            fmpq_mpoly_derivative(derivative.p, h.p, j, ring.ctx);
            if (fmpq_mpoly_gcd(common.p, h.p, derivative.p, ring.ctx) &&
                fmpq_mpoly_is_one(common.p, ring.ctx))
                return true;
        }
    }
    return false;
}

// Kronecker substitution maps exponents in the component-degree box to
// distinct indices. The substitution is multiplicative, and the degree
// box of every integral factor fits in that of f, so its coefficients
// inject unchanged. Mignotte's bound gives |g_i| <= 2^D * ||f||_1
// for each primitive factor, with D the degree of the univariate image.
// Bound the *length* of its sparse description, without forming 2^D.
struct FactorBox {
    Powers radices;
    mpz_class max_degree = 0, max_code_bits = 0;

    FactorBox(const Polynomial& f, Powers degrees) : radices(std::move(degrees)) {
        for (auto& radix : radices)
            ++radix;
        mpz_class norm = 0;
        for (const auto& [e, c] : f) {
            mpz_class index = 0;
            for (size_t j = e.size(); j-- > 0;)
                index = index * radices[j] + e[j];
            max_degree = std::max(max_degree, index);
            norm += abs(c.get_num());
        }
        const mpz_class degree_bits = mpz_sizeinbase(max_degree.get_mpz_t(), 2) + 1;
        const mpz_class coefficient_bits = max_degree + mpz_sizeinbase(norm.get_mpz_t(), 2) + 2;
        // A gamma-encoded positive integer of b bits takes 2b-1 bits.
        max_code_bits = 1 + (max_degree + 1) * 2 * (degree_bits + coefficient_bits);
    }

    // Read an Elias gamma integer from the low end of the word. The
    // distinguished final 1 makes this a self-delimiting term sequence.
    static bool read(mpz_class& code, mpz_class& value) {
        if (code <= 1) return false;
        const auto zeros = mpz_scan1(code.get_mpz_t(), 0);
        code >>= zeros + 1;
        if (mpz_sizeinbase(code.get_mpz_t(), 2) <= zeros || code == 0) return false;
        mpz_fdiv_r_2exp(value.get_mpz_t(), code.get_mpz_t(), zeros);
        mpz_setbit(value.get_mpz_t(), zeros);
        code >>= zeros;
        return true;
    }

    Polynomial decode(mpz_class code) const {
        Polynomial candidate;
        mpz_class previous = -1, index, coefficient;
        while (code > 1) {
            if (!read(code, index) || !read(code, coefficient)) return {};
            --index;
            if (index <= previous || 2 * index > max_degree) return {};
            previous = index;
            Powers e(radices.size());
            for (size_t j = 0; j < e.size(); ++j) {
                e[j] = index % radices[j];
                index /= radices[j];
            }
            if (index) return {};
            if (mpz_odd_p(coefficient.get_mpz_t()))
                coefficient = (coefficient + 1) / 2;
            else
                coefficient = -coefficient / 2;
            candidate.emplace(std::move(e), std::move(coefficient));
        }
        if (previous <= 0 || candidate.rbegin()->second <= 0) return {};
        mpz_class content = 0;
        for (const auto& [e, c] : candidate)
            content = gcd(content, c.get_num());
        return content == 1 ? candidate : Polynomial{};
    }
};

// Lexicographic division in Z[x]. A primitive integral factor has an
// integral quotient by Gauss's lemma. Exponents retain their GMP width.
Polynomial quotient_if_exact(Polynomial remainder, const Polynomial& divisor) {
    const auto& [head, leading] = *divisor.rbegin();
    Polynomial quotient;
    while (!remainder.empty()) {
        auto [shift, coefficient] = *remainder.rbegin();
        Powers powers = shift;
        for (size_t j = 0; j < powers.size(); ++j) {
            if (powers[j] < head[j]) return {};
            powers[j] -= head[j];
        }
        coefficient /= leading;
        if (coefficient.get_den() != 1) return {};
        quotient.emplace(powers, coefficient);
        for (const auto& [e, c] : divisor) {
            auto key = powers;
            for (size_t j = 0; j < key.size(); ++j)
                key[j] += e[j];
            auto entry = remainder.try_emplace(std::move(key), 0).first;
            if ((entry->second -= coefficient * c) == 0) remainder.erase(entry);
        }
    }
    return quotient;
}

void split(Polynomial f, local::SparseLexRing& ring, Factors& result) {
    const size_t n = f.begin()->first.size();
    Powers common = f.begin()->first, degrees(n);
    for (const auto& [e, c] : f)
        for (size_t j = 0; j < n; ++j) {
            common[j] = std::min(common[j], e[j]);
            degrees[j] = std::max(degrees[j], e[j]);
        }
    if (common != Powers(n)) {
        Polynomial rest;
        for (const auto& [e, c] : f) {
            Powers reduced = e;
            for (size_t j = 0; j < n; ++j)
                reduced[j] -= common[j];
            rest.emplace(std::move(reduced), c);
        }
        for (size_t j = 0; j < n; ++j)
            if (common[j]) {
                Powers variable(n);
                variable[j] = 1;
                result[Polynomial{{std::move(variable), 1}}] += common[j];
            }
        if (rest.size() > 1) split(std::move(rest), ring, result);
        return;
    }
    if (eisenstein(f, degrees, ring)) {
        add_monic(result, std::move(f), 1);
        return;
    }
    FactorBox box(f, std::move(degrees));
    for (mpz_class code = 2; mpz_sizeinbase(code.get_mpz_t(), 2) <= box.max_code_bits; ++code) {
        const auto candidate = box.decode(code);
        if (candidate.empty()) continue;
        auto quotient = quotient_if_exact(f, candidate);
        if (!quotient.empty()) {
            split(candidate, ring, result);
            split(std::move(quotient), ring, result);
            return;
        }
    }
    add_monic(result, std::move(f), 1);
}
} // namespace

void exhaustive_rational_factors(fmpq_mpoly_factor_t out, const fmpq_mpoly_t f,
                                 const fmpq_mpoly_ctx_t ctx) {
    // FLINT's input order can differ from lex. Convert explicitly before
    // using lex division, and normalize each returned factor in caller order.
    const slong n = fmpq_mpoly_ctx_nvars(ctx);
    local::SparseFlintRing input(n, fmpq_mpoly_ctx_ord(ctx));
    local::SparseLexRing lex(n);
    Factors factors;
    split(primitive(input.read(f)), lex, factors);
    out->num = 0;
    fmpq_mpoly_get_term_coeff_fmpq(out->constant, f, 0, ctx);
    fmpq_mpoly_factor_fit_length(out, factors.size(), ctx);
    for (const auto& [factor, power] : factors) {
        input.write(out->poly + out->num, factor);
        fmpq_mpoly_make_monic(out->poly + out->num, out->poly + out->num, ctx);
        fmpz_set_mpz(out->exp + out->num, power.get_mpz_t());
        ++out->num;
    }
}

// In a UFD (a) intersection (b) is generated by lcm(a,b). Elimination and
// exact long division over sparse QQ use arbitrary exponents throughout.
void gcd_by_intersection(fmpq_mpoly_t out, const fmpq_mpoly_t a, const fmpq_mpoly_t b,
                         const fmpq_mpoly_ctx_t ctx) {
    if (fmpq_mpoly_is_zero(a, ctx) || fmpq_mpoly_is_zero(b, ctx)) {
        const auto other = fmpq_mpoly_is_zero(a, ctx) ? b : a;
        if (fmpq_mpoly_is_zero(other, ctx))
            fmpq_mpoly_zero(out, ctx);
        else
            fmpq_mpoly_make_monic(out, other, ctx);
        return;
    }
    const auto width = fmpq_mpoly_ctx_nvars(ctx);
    local::SparseFlintRing ring(width, fmpq_mpoly_ctx_ord(ctx));
    const auto left = primitive(ring.read(a)), right = primitive(ring.read(b));
    const auto common = primary::intersection_lex({left}, {right}, width);
    if (common.size() != 1)
        throw std::logic_error("Intersection of two nonzero principal ideals is not principal.");
    const auto g = quotient_if_exact(primary::multiply(left, right), primitive(common.front()));
    if (g.empty()) throw std::logic_error("The primitive lcm does not divide the product.");
    ring.write(out, g);
    fmpq_mpoly_make_monic(out, out, ctx);
}

void integer_gcd_by_intersection(fmpz_mpoly_t out, const fmpz_mpoly_t a, const fmpz_mpoly_t b,
                                 const fmpz_mpoly_ctx_t ctx) {
    local::SparseFlintRing ring(fmpz_mpoly_ctx_nvars(ctx), fmpz_mpoly_ctx_ord(ctx));
    PolyValue left(ring.ctx), right(ring.ctx), g(ring.ctx);
    auto lift = [&](PolyValue& target, const fmpz_mpoly_t value) {
        fmpz_mpoly_set(target.p->zpoly, value, ctx);
        fmpq_one(target.p->content);
        fmpq_mpoly_reduce(target.p, ring.ctx);
    };
    lift(left, a);
    lift(right, b);
    Z content;
    fmpz_gcd(&content.v, fmpq_numref(left.p->content), fmpq_numref(right.p->content));
    gcd_by_intersection(g.p, left.p, right.p, ring.ctx);
    fmpz_mpoly_scalar_mul_fmpz(out, g.p->zpoly, &content.v, ctx);
}
} // namespace atlas
