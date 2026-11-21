// Exact native planning, reconstruction and fallbacks for factorization and division.
#include "triton_factor_division.hpp"
#include "local_io.hpp"
#include "polynomial_factors.hpp"
#include "polynomial_gcd.hpp"
#include "rational_sparse_json.hpp"
#include "sparse_flint.hpp"
#include <flint/fmpq_poly.h>
#include <climits>
#include <set>

namespace atlas {
namespace {
using local::Polynomial;
using local::Powers;
constexpr int prime = 8191;
using RootScreen = int (*)(const int*, int, int*, int);
using BinomialSteps = int (*)(const int*, int, int, int*);
thread_local RootScreen screen = nullptr;
thread_local BinomialSteps steps = nullptr;

struct Rational {
    fmpq_t v;
    Rational() { fmpq_init(v); }
    ~Rational() { fmpq_clear(v); }
};

struct Candidate {
    slong axis;
    mpq_class root;
    slong power;
};

std::vector<Candidate> candidates(const fmpq_mpoly_t f, const fmpq_mpoly_ctx_t ctx) {
    const slong n = fmpq_mpoly_ctx_nvars(ctx);
    if (!screen || n == 0 || n > 16 || f->zpoly->length < 8 || !fmpq_mpoly_degrees_fit_si(f, ctx))
        return {};
    std::vector<slong> degrees(n);
    fmpq_mpoly_degrees_si(degrees.data(), f, ctx);
    // On x_j = T and other coordinates = 1, every factor x_j-r still vanishes
    // at r. Specializations and reduction modulo p may introduce false roots.
    // Only exact division can admit a proposed root.
    std::vector<Candidate> found;
    std::vector<int> flags(prime);
    for (slong axis = 0; axis < n; ++axis) {
        if (degrees[axis] < 4 || degrees[axis] > 512)
            continue;
        std::vector<int> coefficients(degrees[axis] + 1);
        for (slong i = 0; i < f->zpoly->length; ++i) {
            slong e = fmpq_mpoly_get_term_var_exp_si(f, i, axis, ctx);
            int& c = coefficients[e];
            c += fmpz_fdiv_ui(f->zpoly->coeffs + i, prime);
            if (c >= prime)
                c -= prime;
        }
        while (!coefficients.empty() && coefficients.back() == 0)
            coefficients.pop_back();
        if (coefficients.size() < 2)
            continue;
        std::fill(flags.begin(), flags.end(), 0);
        if (!screen(coefficients.data(), coefficients.size(), flags.data(), prime))
            continue;
        std::set<mpq_class> proposed;
        Z p, residue;
        fmpz_set_ui(&p.v, prime);
        Rational fraction;
        for (int i = 0; i < prime; ++i) {
            if (!flags[i])
                continue;
            proposed.emplace(i > prime / 2 ? i - prime : i);
            fmpz_set_ui(&residue.v, i);
            if (fmpq_reconstruct_fmpz(fraction.v, &residue.v, &p.v)) {
                mpq_class root;
                fmpq_get_mpq(root.get_mpq_t(), fraction.v);
                proposed.insert(std::move(root));
            }
        }
        for (const auto& root : proposed)
            found.push_back({axis, root, 0});
    }
    return found;
}

void linear_factor(fmpq_mpoly_t h, const Candidate& candidate, const fmpq_mpoly_ctx_t ctx) {
    Rational c;
    fmpq_set_mpq(c.v, candidate.root.get_mpq_t());
    fmpq_mpoly_gen(h, candidate.axis, ctx);
    fmpq_mpoly_sub_fmpq(h, h, c.v, ctx);
}

void add_term(Polynomial& f, const Powers& e, const mpq_class& c) {
    if (c != 0 && (f[e] += c) == 0)
        f.erase(e);
}
} // namespace

extern "C" void atlas_swap_factor_division_hooks(RootScreen* roots, BinomialSteps* binomials) {
    std::swap(screen, *roots);
    std::swap(steps, *binomials);
}

bool triton_roots_active() { return screen != nullptr; }

bool triton_linear_factors(fmpq_mpoly_factor_t out, const fmpq_mpoly_t f, const fmpq_mpoly_ctx_t ctx,
                           bool squarefree) {
    auto hits = candidates(f, ctx);
    if (hits.empty())
        return false;
    PolyValue residual(ctx), linear(ctx), q(ctx);
    fmpq_mpoly_set(residual.p, f, ctx);
    bool changed = false;
    for (auto& h : hits) {
        linear_factor(linear.p, h, ctx);
        while (fmpq_mpoly_divides(q.p, residual.p, linear.p, ctx)) {
            fmpq_mpoly_swap(q.p, residual.p, ctx);
            ++h.power;
            changed = true;
        }
    }
    if (!changed)
        return false;
    // All roots found by the full screen have been removed. Run the existing
    // general algorithm on the residual, including any factors the GPU missed.
    struct Restore {
        RootScreen old = screen;
        ~Restore() { screen = old; }
    } restore;
    screen = nullptr;
    if (squarefree)
        squarefree_factors(out, residual.p, ctx);
    else
        ordered_factors(out, residual.p, ctx);
    fmpq_mpoly_factor_fit_length(out, out->num + hits.size(), ctx);
    for (const auto& h : hits)
        if (h.power) {
            linear_factor(out->poly + out->num, h, ctx);
            fmpz_set_si(out->exp + out->num++, h.power);
        }
    return true;
}

bool triton_common_linear_factors(fmpq_mpoly_t common, fmpq_mpoly_t a, fmpq_mpoly_t b,
                                  const fmpq_mpoly_ctx_t ctx) {
    if (!screen || !a->zpoly->length || !b->zpoly->length)
        return false;
    auto hits = candidates(a->zpoly->length <= b->zpoly->length ? a : b, ctx);
    fmpq_mpoly_one(common, ctx);
    PolyValue linear(ctx), qa(ctx), qb(ctx);
    bool changed = false;
    for (const auto& h : hits) {
        linear_factor(linear.p, h, ctx);
        while (fmpq_mpoly_divides(qa.p, a, linear.p, ctx) && fmpq_mpoly_divides(qb.p, b, linear.p, ctx)) {
            fmpq_mpoly_swap(a, qa.p, ctx);
            fmpq_mpoly_swap(b, qb.p, ctx);
            fmpq_mpoly_mul(common, common, linear.p, ctx);
            changed = true;
        }
    }
    return changed;
}

bool triton_binomial_division(Polynomial* quotient, Polynomial& remainder, const Polynomial& f,
                              const Polynomial& divisor, bool lex) {
    if (!steps || divisor.size() != 2 || f.size() < 8 || f.size() > 65536)
        return false;
    auto leading = divisor.find(local::global_head(divisor, lex));
    const auto other = leading == divisor.begin() ? std::next(leading) : divisor.begin();
    const auto& u = leading->first;
    const auto& v = other->first;
    const size_t width = u.size();
    if (width == 0 || width > 32)
        return false;
    // One upload supplies the head, tail and term exponent matrix together.
    std::vector<int> packed(2 * width);
    packed.reserve((2 + f.size()) * width);
    for (size_t j = 0; j < width; ++j) {
        if (u[j] > INT_MAX / 2 || v[j] > INT_MAX / 2)
            return false;
        packed[j] = u[j].get_si();
        packed[width + j] = v[j].get_si();
    }
    for (const auto& [e, c] : f)
        for (const auto& power : e) {
            if (power > INT_MAX / 2)
                return false;
            packed.push_back(power.get_si());
        }
    std::vector<int> counts(f.size());
    if (!steps(packed.data(), f.size(), width, counts.data()))
        return false;
    Polynomial rem;
    const mpq_class scale = -other->second / leading->second;
    std::map<int, mpq_class> weights{{0, 1}};
    size_t term = 0;
    Powers r(width);
    for (const auto& [e, c] : f) {
        const auto offset = (2 + term) * width;
        const long k = counts[term++];
        if (k < 0)
            return false;
        bool irreducible = false;
        for (size_t j = 0; j < width; ++j) {
            // k is signed int32 and both exponents are at most INT_MAX/2.
            const int source = packed[offset + j], head = packed[j], tail = packed[width + j];
            const int64_t final = source + int64_t(k) * (tail - head);
            if (final < 0 || (k && (final < tail || source < head)))
                return false;
            irreducible |= final < head;
            r[j] = static_cast<unsigned long>(final);
        }
        // With a fixed binomial step, all preceding exponents lie on the
        // segment between the first and last legal rewrite.
        if (!irreducible)
            return false;
        auto [item, missing] = weights.try_emplace(k);
        if (missing) {
            auto& weight = item->second;
            if (scale == 1)
                weight = 1;
            else if (scale == -1)
                weight = (k % 2 ? -1 : 1);
            else {
                mpz_pow_ui(mpq_numref(weight.get_mpq_t()), mpq_numref(scale.get_mpq_t()), k);
                mpz_pow_ui(mpq_denref(weight.get_mpq_t()), mpq_denref(scale.get_mpq_t()), k);
            }
        }
        add_term(rem, r, c * item->second);
    }
    if (quotient) {
        // Recover all quotient terms together, avoiding overlapping geometric
        // progressions from individual inputs. This also certifies f = q*g+r.
        local::SparseFlintRing ring(width, lex ? ORD_LEX : ORD_DEGREVLEX);
        PolyValue a(ring.ctx), b(ring.ctx), rest(ring.ctx), q(ring.ctx);
        ring.write(a.p, f);
        ring.write(b.p, divisor);
        ring.write(rest.p, rem);
        fmpq_mpoly_sub(a.p, a.p, rest.p, ring.ctx);
        if (!fmpq_mpoly_divides(q.p, a.p, b.p, ring.ctx))
            return false;
        *quotient = ring.read(q.p);
    }
    remainder.swap(rem);
    return true;
}

namespace {
struct Dense {
    fmpq_poly_t p;
    Dense() { fmpq_poly_init(p); }
    ~Dense() { fmpq_poly_clear(p); }
};

void xgcd(PolyValue& g, PolyValue& s, PolyValue& t, PolyValue& a, PolyValue& b, slong axis,
          const fmpq_mpoly_ctx_t ctx) {
    PolyValue common(ctx);
    if (!triton_common_linear_factors(common.p, a.p, b.p, ctx))
        fmpq_mpoly_one(common.p, ctx);
    // With a constant second cofactor fix the Bezout representative explicitly,
    // including proportional inputs and the case of a zero first input.
    if (!fmpq_mpoly_is_zero(b.p, ctx) && fmpq_mpoly_is_fmpq(b.p, ctx)) {
        Rational c;
        fmpq_mpoly_get_fmpq(c.v, b.p, ctx);
        fmpq_inv(c.v, c.v);
        fmpq_mpoly_set_fmpq(t.p, c.v, ctx);
        fmpq_mpoly_set(g.p, common.p, ctx);
        return;
    }
    if (fmpq_mpoly_degrees_fit_si(a.p, ctx) && fmpq_mpoly_degrees_fit_si(b.p, ctx) &&
        std::max(fmpq_mpoly_degree_si(a.p, axis, ctx), fmpq_mpoly_degree_si(b.p, axis, ctx)) <= 100000) {
        Dense x, y, h, u, v;
        if (!fmpq_mpoly_get_fmpq_poly(x.p, a.p, axis, ctx) || !fmpq_mpoly_get_fmpq_poly(y.p, b.p, axis, ctx))
            throw std::logic_error("Unable to read the univariate extended GCD inputs.");
        fmpq_poly_xgcd(h.p, u.p, v.p, x.p, y.p);
        fmpq_mpoly_set_fmpq_poly(g.p, h.p, axis, ctx);
        fmpq_mpoly_set_fmpq_poly(s.p, u.p, axis, ctx);
        fmpq_mpoly_set_fmpq_poly(t.p, v.p, axis, ctx);
    } else {
        // Sparse Euclid avoids a dense array indexed by an arbitrary GMP exponent.
        PolyValue old_s(ctx), old_t(ctx), new_s(ctx), new_t(ctx), q(ctx), r(ctx), tmp(ctx);
        fmpq_mpoly_one(old_s.p, ctx);
        fmpq_mpoly_one(new_t.p, ctx);
        while (!fmpq_mpoly_is_zero(b.p, ctx)) {
            fmpq_mpoly_divrem(q.p, r.p, a.p, b.p, ctx);
            fmpq_mpoly_swap(a.p, b.p, ctx);
            fmpq_mpoly_swap(b.p, r.p, ctx);
            for (auto pair : {std::make_pair(old_s.p, new_s.p), std::make_pair(old_t.p, new_t.p)}) {
                fmpq_mpoly_mul(tmp.p, q.p, pair.second, ctx);
                fmpq_mpoly_sub(tmp.p, pair.first, tmp.p, ctx);
                fmpq_mpoly_swap(pair.first, pair.second, ctx);
                fmpq_mpoly_swap(pair.second, tmp.p, ctx);
            }
        }
        Rational lead;
        fmpq_mpoly_get_term_coeff_fmpq(lead.v, a.p, 0, ctx);
        fmpq_mpoly_scalar_div_fmpq(g.p, a.p, lead.v, ctx);
        fmpq_mpoly_scalar_div_fmpq(s.p, old_s.p, lead.v, ctx);
        fmpq_mpoly_scalar_div_fmpq(t.p, old_t.p, lead.v, ctx);
    }
    fmpq_mpoly_mul(g.p, g.p, common.p, ctx);
}

// A scalar has no active coordinate (-1). Mixed inputs have no common axis.
std::optional<slong> univariate_axis(const Polynomial& a, const Polynomial& b, size_t width) {
    slong axis = -1;
    for (const auto* source : {&a, &b}) {
        for (const auto& term : *source) {
            for (size_t j = 0; j < width; ++j) {
                if (term.first[j] == 0)
                    continue;
                if (axis >= 0 && axis != static_cast<slong>(j))
                    return std::nullopt;
                axis = static_cast<slong>(j);
            }
        }
    }
    return axis;
}

Json::Value polynomial_division_operation(const Json::Value& input) {
    auto op = input["operation"].asString();
    size_t width = input["width"].asUInt();
    const bool lex = input["order"].asString() == "lex";
    const auto f = local::decode_row(input["left"].asString(), width);
    const auto h = local::decode_row(input["right"].asString(), width);
    Json::Value result(Json::objectValue);
    const auto row = [](const Polynomial& p) { return local::encode_row(p); };
    if (op == "try-univariate-gcd") {
        if (!width || f.size() < 8 || h.size() < 8 || !univariate_axis(f, h, width))
            return Json::Value();
        local::SparseFlintRing ring(width, lex ? ORD_LEX : ORD_DEGREVLEX);
        PolyValue a(ring.ctx), b(ring.ctx), common(ring.ctx);
        ring.write(a.p, f);
        ring.write(b.p, h);
        monic_polynomial_gcd(common.p, a.p, b.p, ring.ctx);
        result["gcd"] = row(ring.read(common.p));
        return result;
    }
    if (op == "cancel-fraction") {
        if (h.empty())
            throw std::domain_error("A rational function requires a nonzero denominator.");
        const Polynomial one{{Powers(width), 1}};
        if (f.empty() || !width) {
            Polynomial value;
            if (!f.empty())
                value[{}] = f.begin()->second / h.begin()->second;
            result["numerator"] = row(value);
            result["denominator"] = row(one);
            return result;
        }
        local::SparseFlintRing ring(width, lex ? ORD_LEX : ORD_DEGREVLEX);
        PolyValue a(ring.ctx), b(ring.ctx), gcd(ring.ctx), num(ring.ctx), den(ring.ctx);
        ring.write(a.p, f);
        ring.write(b.p, h);
        monic_polynomial_gcd(gcd.p, a.p, b.p, ring.ctx);
        if (!fmpq_mpoly_divides(num.p, a.p, gcd.p, ring.ctx) ||
            !fmpq_mpoly_divides(den.p, b.p, gcd.p, ring.ctx))
            throw std::logic_error("The computed polynomial GCD did not divide both operands.");
        Rational lead;
        fmpq_mpoly_get_term_coeff_fmpq(lead.v, den.p, 0, ring.ctx);
        fmpq_mpoly_scalar_div_fmpq(num.p, num.p, lead.v, ring.ctx);
        fmpq_mpoly_scalar_div_fmpq(den.p, den.p, lead.v, ring.ctx);
        result["numerator"] = row(ring.read(num.p));
        result["denominator"] = row(ring.read(den.p));
        return result;
    }
    if (op == "polynomial-division" || op == "try-binomial-remainder") {
        if (h.empty())
            throw std::domain_error("Polynomial division requires a nonzero divisor.");
        Polynomial quotient, remainder;
        if (triton_binomial_division(op == "polynomial-division" ? &quotient : nullptr, remainder, f, h,
                                     lex)) {
            result["remainder"] = row(remainder);
            if (op == "polynomial-division")
                result["quotient"] = row(quotient);
            return result;
        }
        if (op == "try-binomial-remainder")
            return Json::Value();
        if (!width) {
            if (!f.empty())
                quotient[{}] = f.begin()->second / h.begin()->second;
            result["quotient"] = row(quotient);
            result["remainder"] = "";
            return result;
        }
        local::SparseFlintRing ring(width, lex ? ORD_LEX : ORD_DEGREVLEX);
        PolyValue a(ring.ctx), b(ring.ctx), q(ring.ctx), r(ring.ctx);
        ring.write(a.p, f);
        ring.write(b.p, h);
        fmpq_mpoly_divrem(q.p, r.p, a.p, b.p, ring.ctx);
        result["quotient"] = row(ring.read(q.p));
        result["remainder"] = row(ring.read(r.p));
        return result;
    }
    if (op != "extended-gcd")
        throw std::invalid_argument("Unknown polynomial division operation.");
    const auto axis = univariate_axis(f, h, width);
    if (!axis)
        throw std::invalid_argument("The extended GCD requires polynomials in one common variable over QQ.");
    if (*axis < 0) {
        const Polynomial one{{Powers(width), 1}};
        Polynomial s, t;
        if (!h.empty())
            t[Powers(width)] = 1 / h.begin()->second;
        else if (!f.empty())
            s[Powers(width)] = 1 / f.begin()->second;
        result["gcd"] = f.empty() && h.empty() ? "" : row(one);
        result["left_coefficient"] = row(s);
        result["right_coefficient"] = row(t);
        return result;
    }
    local::SparseFlintRing ring(width);
    PolyValue a(ring.ctx), b(ring.ctx), g(ring.ctx), s(ring.ctx), t(ring.ctx);
    ring.write(a.p, f);
    ring.write(b.p, h);
    xgcd(g, s, t, a, b, *axis, ring.ctx);
    result["gcd"] = row(ring.read(g.p));
    result["left_coefficient"] = row(ring.read(s.p));
    result["right_coefficient"] = row(ring.read(t.p));
    return result;
}
} // namespace
} // namespace atlas
extern "C" const char* atlas_polynomial_division(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, atlas::polynomial_division_operation);
}
