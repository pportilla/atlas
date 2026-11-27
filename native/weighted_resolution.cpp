// Embedded resolution of a hypersurface germ by weighted toroidal blowups.
//
// The centres are those of the logarithmic weighted resolution of Abramovich,
// Temkin and Wlodarczyk and of Abramovich and Quek. At a point p with
// exceptional (boundary) coordinates u, which stay fixed, and free coordinates
// x, the coefficient collections {(g, r)} start from {(f, 1)}:
//
//   * the logarithmic order of g is the order of g on {u = 0};
//   * a_i is the least logarithmic order of an element divided by its weight r;
//     x_i is a Tschirnhausen coordinate of an element that attains it;
//   * the next collection holds the coefficients g_j of g = sum_j g_j x_i^j
//     with the weights r - j/a_i that are still positive.
//
// When no free coordinate is left, or every element vanishes on {u = 0}, the
// remaining elements contribute the monomials of their u-parts, scaled by
// 1/r, as the monomial part M. The centre is the Q-ideal
// J = (x_1^{a_1}, ..., x_k^{a_k}) + M, monomial in these coordinates, so its
// normalized blowup is the toric modification given by the normal fan of its
// Newton polyhedron; each chart is the monomial map of a cone of that fan.
//
// Points over the blown-up point where the strict transform has logarithmic
// order at least two (it is singular there, or not transversal to the
// exceptional divisors) get a centre of their own. They are found as the prime
// components of a finite set over Q: a Galois orbit of points is one component,
// its residue field Q[g]/(f) comes from a separating linear form (shape lemma)
// and the orbit is resolved once, over that field. A chart over a number field
// carries its generator theta as one more variable, reduced modulo its minimal
// polynomial; orders and exponents concern the other variables only.
#include <gmp.h>
#include <gmpxx.h>

#include "groebner.hpp"
#include "primary_decomposition.hpp"
#include "separating_form.hpp"
#include "rational_sparse_json.hpp"

#include <flint/fmpq.h>
#include <flint/fmpq_mat.h>
#include <flint/fmpq_mpoly.h>
#include <flint/fmpq_poly.h>
#include <flint/fmpz_mat.h>
#include <flint/fmpz_poly.h>
#include <flint/fmpz_poly_factor.h>
#include <json/json.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <chrono>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

extern "C" char* atlas_ideal(const char*, size_t) noexcept;
extern "C" void atlas_groebner_release(char*) noexcept;

namespace atlas::weighted {
namespace {

using Q = mpq_class;
using Zi = mpz_class;
using Clock = std::chrono::steady_clock;

struct Declined : std::exception {
    std::string reason;
    explicit Declined(std::string message) : reason(std::move(message)) {}
    const char* what() const noexcept override { return reason.c_str(); }
};

long checked_order(const Zi& n) {
    if (n < 0 || !n.fits_slong_p()) throw Declined{"A weighted order exceeds the signed native word range."};
    return n.get_si();
}

thread_local Clock::time_point deadline_at = Clock::time_point::max();
thread_local Clock::time_point started_at;

void check_deadline() {
    if (Clock::now() > deadline_at) throw Declined{"The time allowed is over."};
}

// ---------------------------------------------------------------- rings over a number field

// n coordinates, and over a field of degree d > 1 the generator theta as
// variable n, reduced modulo its monic minimal polynomial.
struct Ctx {
    fmpq_mpoly_ctx_t c;
    fmpq_poly_t minimal;
    fmpq_mpoly_t modulus;  // the minimal polynomial in theta
    int n, nv, d;
    Ctx(int coordinates, const fmpq_poly_struct* field) : n(coordinates) {
        fmpq_poly_init(minimal);
        if (field && fmpq_poly_degree(field) > 1) fmpq_poly_set(minimal, field);
        else {
            fmpq_poly_zero(minimal);
            fmpq_poly_set_coeff_si(minimal, 1, 1);
        }
        fmpq_poly_make_monic(minimal, minimal);
        d = static_cast<int>(fmpq_poly_degree(minimal));
        nv = n + (d > 1 ? 1 : 0);
        fmpq_mpoly_ctx_init(c, nv, ORD_DEGREVLEX);
        fmpq_mpoly_init(modulus, c);
        if (d > 1) {
            fmpq_t q;
            fmpq_init(q);
            std::vector<ulong> e(nv, 0);
            for (slong k = 0; k <= d; ++k) {
                fmpq_poly_get_coeff_fmpq(q, minimal, k);
                if (fmpq_is_zero(q)) continue;
                e[n] = static_cast<ulong>(k);
                fmpq_mpoly_push_term_fmpq_ui(modulus, q, e.data(), c);
            }
            fmpq_clear(q);
            fmpq_mpoly_sort_terms(modulus, c);
            fmpq_mpoly_combine_like_terms(modulus, c);
        }
    }
    ~Ctx() {
        fmpq_mpoly_clear(modulus, c);
        fmpq_mpoly_ctx_clear(c);
        fmpq_poly_clear(minimal);
    }
    Ctx(const Ctx&) = delete;
    Ctx& operator=(const Ctx&) = delete;
};

using CtxPtr = std::shared_ptr<Ctx>;

class P {
public:
    explicit P(const Ctx& x) : x_(&x) { fmpq_mpoly_init(p, x_->c); }
    P(const P& o) : x_(o.x_) {
        fmpq_mpoly_init(p, x_->c);
        fmpq_mpoly_set(p, o.p, x_->c);
    }
    P(P&& o) noexcept : x_(o.x_) {
        fmpq_mpoly_init(p, x_->c);
        fmpq_mpoly_swap(p, o.p, x_->c);
    }
    P& operator=(const P& o) {
        if (this == &o) return *this;
        if (x_ != o.x_) {
            fmpq_mpoly_clear(p, x_->c);
            x_ = o.x_;
            fmpq_mpoly_init(p, x_->c);
        }
        fmpq_mpoly_set(p, o.p, x_->c);
        return *this;
    }
    P& operator=(P&& o) noexcept {
        if (x_ != o.x_) {
            fmpq_mpoly_clear(p, x_->c);
            x_ = o.x_;
            fmpq_mpoly_init(p, x_->c);
        }
        fmpq_mpoly_swap(p, o.p, x_->c);
        return *this;
    }
    ~P() { fmpq_mpoly_clear(p, x_->c); }
    const Ctx& ctx() const { return *x_; }
    bool zero() const { return fmpq_mpoly_is_zero(p, x_->c); }
    slong length() const { return fmpq_mpoly_length(p, x_->c); }
    std::vector<ulong> exponent(slong k) const {
        if (!fmpq_mpoly_term_exp_fits_si(p, k, x_->c))
            throw Declined{"A weighted chart exponent exceeds the signed native word range."};
        std::vector<ulong> e(x_->nv);
        fmpq_mpoly_get_term_exp_ui(e.data(), p, k, x_->c);
        // Orders, including orders on subsets of coordinates, are signed longs.
        ulong remaining = std::numeric_limits<long>::max();
        for (int i = 0; i < x_->n; ++i) {
            if (e[i] > remaining) throw Declined{"A weighted total degree exceeds the signed native word range."};
            remaining -= e[i];
        }
        return e;
    }
    Q coefficient(slong k) const {
        fmpq_t c;
        fmpq_init(c);
        fmpq_mpoly_get_term_coeff_fmpq(c, p, k, x_->c);
        Q out;
        fmpq_get_mpq(out.get_mpq_t(), c);
        fmpq_clear(c);
        return out;
    }
    fmpq_mpoly_t p;

private:
    const Ctx* x_;
};

// The total degree in the coordinates (theta is part of the coefficient).
ulong degree(const std::vector<ulong>& e, int n) {
    ulong s = 0;
    for (int i = 0; i < n; ++i) s += e[i];
    return s;
}

void reduce(P& f) {
    const Ctx& x = f.ctx();
    if (x.d <= 1 || f.zero()) return;
    P q(x), r(x);
    fmpq_mpoly_divrem(q.p, r.p, f.p, x.modulus, x.c);
    f = std::move(r);
}

P gen(const Ctx& x, int i) {
    P g(x);
    fmpq_mpoly_gen(g.p, i, x.c);
    return g;
}

P constant(const Ctx& x, const Q& q) {
    P g(x);
    fmpq_t c;
    fmpq_init(c);
    fmpq_set_mpq(c, q.get_mpq_t());
    fmpq_mpoly_set_fmpq(g.p, c, x.c);
    fmpq_clear(c);
    return g;
}

P add(const P& a, const P& b) {
    P out(a.ctx());
    fmpq_mpoly_add(out.p, a.p, b.p, a.ctx().c);
    return out;
}

P sub(const P& a, const P& b) {
    P out(a.ctx());
    fmpq_mpoly_sub(out.p, a.p, b.p, a.ctx().c);
    return out;
}

P mul(const P& a, const P& b) {
    P out(a.ctx());
    fmpq_mpoly_mul(out.p, a.p, b.p, a.ctx().c);
    reduce(out);
    return out;
}

P scale(const P& a, const Q& q) {
    P out(a.ctx());
    fmpq_t c;
    fmpq_init(c);
    fmpq_set_mpq(c, q.get_mpq_t());
    fmpq_mpoly_scalar_mul_fmpq(out.p, a.p, c, a.ctx().c);
    fmpq_clear(c);
    return out;
}

// Terms given by exponent vectors (with theta last), in any order.
P from_terms(const Ctx& x, const std::vector<std::pair<std::vector<ulong>, Q>>& terms) {
    P out(x);
    fmpq_t c;
    fmpq_init(c);
    for (const auto& [e, q] : terms) {
        if (q == 0) continue;
        fmpq_set_mpq(c, q.get_mpq_t());
        fmpq_mpoly_push_term_fmpq_ui(out.p, c, e.data(), x.c);
    }
    fmpq_clear(c);
    fmpq_mpoly_sort_terms(out.p, x.c);
    fmpq_mpoly_combine_like_terms(out.p, x.c);
    reduce(out);
    return out;
}

// The least total degree of a term; -1 for zero.
long order(const P& f) {
    if (f.zero()) return -1;
    long best = -1;
    const int n = f.ctx().n;
    for (slong k = 0; k < f.length(); ++k) {
        const long d = static_cast<long>(degree(f.exponent(k), n));
        if (best < 0 || d < best) best = d;
    }
    return best;
}

// f with the coordinates marked in `zero` set to zero.
P restrict_zero(const P& f, const std::vector<bool>& zero) {
    std::vector<std::pair<std::vector<ulong>, Q>> terms;
    for (slong k = 0; k < f.length(); ++k) {
        auto e = f.exponent(k);
        bool keep = true;
        for (size_t i = 0; i < zero.size(); ++i)
            if (zero[i] && e[i]) keep = false;
        if (keep) terms.emplace_back(std::move(e), f.coefficient(k));
    }
    return from_terms(f.ctx(), terms);
}

P homogeneous(const P& f, ulong d) {
    std::vector<std::pair<std::vector<ulong>, Q>> terms;
    for (slong k = 0; k < f.length(); ++k) {
        auto e = f.exponent(k);
        if (degree(e, f.ctx().n) == d) terms.emplace_back(std::move(e), f.coefficient(k));
    }
    return from_terms(f.ctx(), terms);
}


// f = sum_j f_j x_var^j with f_j free of x_var.
std::map<ulong, P> expand(const P& f, int var) {
    std::map<ulong, std::vector<std::pair<std::vector<ulong>, Q>>> parts;
    for (slong k = 0; k < f.length(); ++k) {
        auto e = f.exponent(k);
        const ulong j = e[var];
        e[var] = 0;
        parts[j].emplace_back(std::move(e), f.coefficient(k));
    }
    std::map<ulong, P> out;
    for (auto& [j, terms] : parts) out.emplace(j, from_terms(f.ctx(), terms));
    return out;
}

// The coefficient, in the field, of the monomial with coordinate exponents e.
P field_coefficient(const P& f, const std::vector<ulong>& e) {
    const int n = f.ctx().n;
    std::vector<std::pair<std::vector<ulong>, Q>> terms;
    for (slong k = 0; k < f.length(); ++k) {
        auto ek = f.exponent(k);
        bool same = true;
        for (int i = 0; i < n; ++i)
            if (ek[i] != e[i]) same = false;
        if (same) terms.emplace_back(std::move(ek), f.coefficient(k));
    }
    return from_terms(f.ctx(), terms);
}

// The inverse of a nonzero field element (a polynomial in theta alone).
P field_inverse(const P& a) {
    const Ctx& x = a.ctx();
    if (a.zero()) throw Declined{"Division by zero in the field."};
    if (x.d <= 1) {
        const Q c = a.coefficient(0);
        return constant(x, 1 / c);
    }
    fmpq_poly_t A, G, S, T;
    fmpq_poly_init(A);
    fmpq_poly_init(G);
    fmpq_poly_init(S);
    fmpq_poly_init(T);
    for (slong k = 0; k < a.length(); ++k) {
        const auto e = a.exponent(k);
        const Q c = a.coefficient(k);
        fmpq_t q;
        fmpq_init(q);
        fmpq_set_mpq(q, c.get_mpq_t());
        fmpq_poly_set_coeff_fmpq(A, static_cast<slong>(e[x.n]), q);
        fmpq_clear(q);
    }
    fmpq_poly_xgcd(G, S, T, A, x.minimal);
    // S*A + T*m = G, a nonzero constant since m is irreducible.
    if (fmpq_poly_degree(G) != 0) throw Declined{"A field element is not invertible."};
    fmpq_t g;
    fmpq_init(g);
    fmpq_poly_get_coeff_fmpq(g, G, 0);
    fmpq_poly_scalar_div_fmpq(S, S, g);
    fmpq_clear(g);
    std::vector<std::pair<std::vector<ulong>, Q>> terms;
    for (slong k = 0; k <= fmpq_poly_degree(S); ++k) {
        fmpq_t q;
        fmpq_init(q);
        fmpq_poly_get_coeff_fmpq(q, S, k);
        Q c;
        fmpq_get_mpq(c.get_mpq_t(), q);
        fmpq_clear(q);
        std::vector<ulong> e(x.nv, 0);
        e[x.n] = static_cast<ulong>(k);
        terms.emplace_back(e, c);
    }
    fmpq_poly_clear(A);
    fmpq_poly_clear(G);
    fmpq_poly_clear(S);
    fmpq_poly_clear(T);
    return from_terms(x, terms);
}

// f with every variable (theta included) replaced by an image in the ring `to`.
P map_into(const P& f, const Ctx& to, const std::vector<P>& images) {
    std::vector<fmpq_mpoly_struct*> at;
    std::vector<P> copies(images);
    for (auto& g : copies) at.push_back(g.p);
    P out(to);
    if (!fmpq_mpoly_compose_fmpq_mpoly(out.p, f.p, at.data(), f.ctx().c, to.c))
        throw Declined{"A substitution exceeded the exponent range."};
    reduce(out);
    return out;
}

// The identity on theta, as the image list of map_into within one ring.
std::vector<P> identity_images(const Ctx& x) {
    std::vector<P> images;
    for (int i = 0; i < x.nv; ++i) images.push_back(gen(x, i));
    return images;
}

// f with x_i replaced by images[i] for the coordinates i (theta stays).
P compose(const P& f, const std::vector<P>& images) {
    auto all = images;
    while (static_cast<int>(all.size()) < f.ctx().nv) all.push_back(gen(f.ctx(), static_cast<int>(all.size())));
    return map_into(f, f.ctx(), all);
}

P substitute(const P& f, int var, const P& image) {
    auto images = identity_images(f.ctx());
    images[var] = image;
    return map_into(f, f.ctx(), images);
}

// A jet: terms with at most `free_cap` in the free coordinates and at most
// `fixed_cap` in the fixed (boundary) coordinates.
struct Jet {
    ulong free_cap, fixed_cap;
    std::vector<bool> fixed;
    bool keeps(const std::vector<ulong>& e) const {
        ulong a = 0, b = 0;
        for (size_t i = 0; i < fixed.size(); ++i) (fixed[i] ? b : a) += e[i];
        return a <= free_cap && b <= fixed_cap;
    }
};

P truncate(const P& f, const Jet& jet) {
    std::vector<std::pair<std::vector<ulong>, Q>> terms;
    for (slong k = 0; k < f.length(); ++k) {
        auto e = f.exponent(k);
        if (jet.keeps(e)) terms.emplace_back(std::move(e), f.coefficient(k));
    }
    return from_terms(f.ctx(), terms);
}

P substitute_truncated(const P& f, int var, const P& image, const Jet& jet) {
    auto parts = expand(f, var);
    P result(f.ctx());
    P power = constant(f.ctx(), Q(1));
    ulong j = 0;
    for (const auto& [e, part] : parts) {
        while (j < e) {
            power = truncate(mul(power, image), jet);
            ++j;
        }
        result = add(result, truncate(mul(part, power), jet));
    }
    return result;
}

// f with x_i replaced by x_i + point[i] (field elements).
P translate(const P& f, const std::vector<P>& point) {
    auto images = identity_images(f.ctx());
    for (int i = 0; i < f.ctx().n; ++i)
        if (!point[i].zero()) images[i] = add(images[i], point[i]);
    return map_into(f, f.ctx(), images);
}

P derivative(const P& f, int var) {
    P out(f.ctx());
    fmpq_mpoly_derivative(out.p, f.p, var, f.ctx().c);
    return out;
}

std::string row(const P& f) {
    std::string out;
    for (slong k = 0; k < f.length(); ++k) {
        if (k) out += ';';
        out += f.coefficient(k).get_str();
        for (ulong e : f.exponent(k)) out += ' ' + std::to_string(e);
    }
    return out;
}

P from_row(const Ctx& x, const std::string& text) {
    std::vector<std::pair<std::vector<ulong>, Q>> terms;
    std::istringstream in(text);
    std::string term;
    while (std::getline(in, term, ';')) {
        std::istringstream t(term);
        std::string c;
        if (!(t >> c)) continue;
        std::vector<ulong> e(x.nv);
        for (auto& v : e)
            if (!(t >> v)) throw std::invalid_argument("Bad polynomial row.");
        Q q(c);
        q.canonicalize();
        terms.emplace_back(e, q);
    }
    return from_terms(x, terms);
}

Json::Value minimal_json(const Ctx& x) {
    Json::Value out(Json::arrayValue);
    for (slong k = 0; k <= fmpq_poly_degree(x.minimal); ++k) {
        fmpq_t q;
        fmpq_init(q);
        fmpq_poly_get_coeff_fmpq(q, x.minimal, k);
        Q c;
        fmpq_get_mpq(c.get_mpq_t(), q);
        fmpq_clear(q);
        out.append(c.get_str());
    }
    return out;
}

// ---------------------------------------------------------------- orbits of points

// A Galois orbit of points: its field, the old theta and the coordinates in
// that field, and generators over the old ring of its prime ideal.
struct Orbit {
    CtxPtr field;
    P theta;                 // the old generator, in the new field (when the old field is not Q)
    std::vector<P> point;    // coordinates, in the new field
    std::vector<P> ideal;    // in the old ring
};

std::vector<CtxPtr>* context_registry = nullptr;

CtxPtr make_field(int n, const fmpq_poly_struct* minimal) {
    auto x = std::make_shared<Ctx>(n, minimal);
    if (context_registry) context_registry->push_back(x);
    return x;
}

// Rows of the native Groebner engine over Q in the variables `vars` of x (theta
// included when listed) and, last, one extra variable.
std::string engine_row(const P& e, const std::vector<int>& vars, size_t extra_power) {
    std::string text;
    for (slong t = 0; t < e.length(); ++t) {
        if (t) text += ';';
        text += e.coefficient(t).get_str();
        const auto exps = e.exponent(t);
        for (int v : vars) text += ' ' + std::to_string(exps[v]);
        text += ' ' + std::to_string(extra_power);
    }
    return text;
}

Poly engine_poly(const Ring& R, const std::string& text) {
    const char* at = text.data();
    Rational scale;
    return parse(R, at, text.data() + text.size(), &scale);
}

// The orbit with field Q[g]/(f), with the coordinates of the variables `all`
// (theta is variable n) as polynomials phi in g, where g = L in the old ring.
Orbit make_orbit(const fmpz_poly_struct* f, const std::vector<std::vector<Q>>& phi, const std::vector<int>& all,
                 const P& L, const std::vector<P>& fixed_zero, const Ctx& x) {
    fmpq_poly_t minimal;
    fmpq_poly_init(minimal);
    fmpq_poly_set_fmpz_poly(minimal, f);
    CtxPtr field = make_field(x.n, minimal);
    fmpq_poly_clear(minimal);
    const Ctx& y = *field;
    Q root = 0;
    if (y.d <= 1) {
        Zi f0, f1;
        fmpz_get_mpz(f0.get_mpz_t(), f->coeffs + 0);
        fmpz_get_mpz(f1.get_mpz_t(), f->coeffs + 1);
        root = Q(-f0, f1);
        root.canonicalize();
    }
    auto in_field = [&](const std::vector<Q>& coeffs) {
        if (y.d <= 1) {
            Q v = 0, power = 1;
            for (const auto& q : coeffs) v += q * power, power *= root;
            return constant(y, v);
        }
        std::vector<std::pair<std::vector<ulong>, Q>> terms;
        for (size_t t = 0; t < coeffs.size(); ++t) {
            std::vector<ulong> ex(y.nv, 0);
            ex[y.n] = static_cast<ulong>(t);
            terms.emplace_back(ex, coeffs[t]);
        }
        return from_terms(y, terms);
    };
    Orbit o{field, constant(y, Q(0)), {}, {}};
    for (int i = 0; i < x.n; ++i) o.point.push_back(constant(y, Q(0)));
    for (size_t j = 0; j < all.size(); ++j) {
        if (all[j] == x.n) o.theta = in_field(phi[j]);
        else o.point[all[j]] = in_field(phi[j]);
    }
    auto at_L = [&](const std::vector<Q>& coeffs) {
        P out = constant(x, Q(0)), power = constant(x, Q(1));
        for (const auto& q : coeffs) {
            out = add(out, scale(power, q));
            power = mul(power, L);
        }
        return out;
    };
    std::vector<Q> fq;
    for (slong t = 0; t <= fmpz_poly_degree(f); ++t) {
        Zi z;
        fmpz_get_mpz(z.get_mpz_t(), f->coeffs + t);
        fq.push_back(Q(z));
    }
    o.ideal = fixed_zero;
    o.ideal.push_back(at_L(fq));
    for (size_t j = 0; j < all.size(); ++j) o.ideal.push_back(sub(gen(x, all[j]), at_L(phi[j])));
    return o;
}

std::vector<Orbit> orbits_by_form(const std::vector<P>& equations, const std::vector<int>& vars, const Ctx& x);

// The Galois orbits of V(equations) in the coordinates `vars` (the other
// coordinates are zero already). The lexicographic basis has theta first and
// the last coordinate last; each factor of its last eliminant is one orbit
// when that coordinate separates the points over it, the usual case, and is
// split by a separating linear form otherwise. Declines on a set of positive
// dimension.
std::vector<Orbit> orbits(const std::vector<P>& equations, const std::vector<int>& vars, const Ctx& x) {
    check_deadline();
    std::vector<P> eqs;
    for (const auto& e : equations)
        if (!e.zero()) eqs.push_back(e);
    for (const auto& e : eqs)
        if (order(e) == 0) {
            bool only_field = true;
            for (slong k = 0; k < e.length(); ++k)
                if (degree(e.exponent(k), x.n) != 0) only_field = false;
            if (only_field) return {};
        }
    if (vars.empty()) return orbits_by_form(equations, vars, x);
    std::vector<P> fixed_zero;
    for (int i = 0; i < x.n; ++i)
        if (std::find(vars.begin(), vars.end(), i) == vars.end()) fixed_zero.push_back(gen(x, i));
    std::vector<int> all;
    if (x.d > 1) all.push_back(x.n);
    for (int v : vars) all.push_back(v);
    const int k = static_cast<int>(all.size());
    if (k > MAX_VARIABLES) throw Declined{"Too many variables for the point search."};
    if (eqs.empty()) throw Declined{"A locus expected to be finite has positive dimension."};
    Ring R;
    R.init(k, 1);
    auto plain_row = [&](const P& e) {
        std::string text;
        for (slong t = 0; t < e.length(); ++t) {
            if (t) text += ';';
            text += e.coefficient(t).get_str();
            const auto exps = e.exponent(t);
            for (int v : all) text += ' ' + std::to_string(exps[v]);
        }
        return text;
    };
    std::vector<Poly> input;
    for (const auto& e : eqs) input.push_back(engine_poly(R, plain_row(e)));
    if (x.d > 1) {
        P m(x);
        fmpq_mpoly_set(m.p, x.modulus, x.c);
        input.push_back(engine_poly(R, plain_row(m)));
    }
    Engine engine;
    engine.R = R;
    engine.compute(input, 0);
    if (engine.unit) return {};
    std::vector<long> e(k);
    std::vector<bool> pure(k, false);
    for (const Element& g : engine.G) {
        R.unpack(g.p.m.data(), e.data());
        int nonzero = 0, last = -1;
        for (int j = 0; j < k; ++j)
            if (e[j]) ++nonzero, last = j;
        if (nonzero == 1) pure[last] = true;
    }
    if (!std::all_of(pure.begin(), pure.end(), [](bool b) { return b; }))
        throw Declined{"A locus expected to be finite has positive dimension."};
    const Poly* eliminant = nullptr;
    for (const Element& g : engine.G) {
        bool only = true;
        for (size_t t = 0; t < g.p.size() && only; ++t) {
            R.unpack(g.p.m.data() + t * R.stride, e.data());
            for (int j = 0; j < k - 1; ++j)
                if (e[j]) only = false;
        }
        if (only) eliminant = &g.p;
    }
    if (!eliminant) throw Declined{"No eliminant in a finite basis."};
    fmpz_poly_t u;
    fmpz_poly_init(u);
    for (size_t t = 0; t < eliminant->size(); ++t) {
        R.unpack(eliminant->m.data() + t * R.stride, e.data());
        fmpz_poly_set_coeff_fmpz(u, e[k - 1], &eliminant->c[t].v);
    }
    fmpz_poly_factor_t factors;
    fmpz_poly_factor_init(factors);
    fmpz_poly_factor(factors, u);
    std::vector<Orbit> out;
    for (slong i = 0; i < factors->num; ++i) {
        const fmpz_poly_struct* f = factors->p + i;
        std::vector<Poly> more;
        for (const Element& g : engine.G) more.push_back(g.p);
        P fp(x);
        {
            std::vector<std::pair<std::vector<ulong>, Q>> terms;
            for (slong t = 0; t <= fmpz_poly_degree(f); ++t) {
                if (fmpz_is_zero(f->coeffs + t)) continue;
                Zi z;
                fmpz_get_mpz(z.get_mpz_t(), f->coeffs + t);
                std::vector<ulong> ex(x.nv, 0);
                ex[all[k - 1]] = static_cast<ulong>(t);
                terms.emplace_back(ex, Q(z));
            }
            fp = from_terms(x, terms);
        }
        more.push_back(engine_poly(R, plain_row(fp)));
        Engine part;
        part.R = R;
        part.compute(more, 0);
        if (part.unit) continue;
        std::vector<std::vector<Q>> phi(k);
        std::vector<bool> found(k, false);
        phi[k - 1] = {Q(0), Q(1)};
        found[k - 1] = true;
        for (const Element& g : part.G) {
            R.unpack(g.p.m.data(), e.data());
            int lead = -1, count = 0;
            for (int j = 0; j < k - 1; ++j)
                if (e[j]) lead = j, count += static_cast<int>(e[j]);
            if (lead < 0 || count != 1) continue;
            Zi a;
            fmpz_get_mpz(a.get_mpz_t(), &g.p.c[0].v);
            std::vector<Q> poly;
            bool linear = true;
            for (size_t t = 1; t < g.p.size(); ++t) {
                R.unpack(g.p.m.data() + t * R.stride, e.data());
                for (int j = 0; j < k - 1; ++j)
                    if (e[j]) linear = false;
                Zi b;
                fmpz_get_mpz(b.get_mpz_t(), &g.p.c[t].v);
                if (static_cast<long>(poly.size()) <= e[k - 1]) poly.resize(e[k - 1] + 1, Q(0));
                poly[e[k - 1]] = Q(-b, a);
                poly[e[k - 1]].canonicalize();
            }
            if (!linear) continue;
            phi[lead] = poly;
            found[lead] = true;
        }
        if (std::all_of(found.begin(), found.end(), [](bool b) { return b; })) {
            out.push_back(make_orbit(f, phi, all, gen(x, all[k - 1]), fixed_zero, x));
            continue;
        }
        // The last coordinate does not separate these points: fix its value and
        // solve for the other coordinates, over Q[v]/(f) when f is not linear.
        const int v = all[k - 1];
        std::vector<int> rest(vars.begin(), vars.end());
        rest.erase(std::find(rest.begin(), rest.end(), v));
        if (fmpz_poly_degree(f) == 1) {
            Zi f0, f1;
            fmpz_get_mpz(f0.get_mpz_t(), f->coeffs + 0);
            fmpz_get_mpz(f1.get_mpz_t(), f->coeffs + 1);
            Q r(-f0, f1);
            r.canonicalize();
            auto images = identity_images(x);
            images[v] = constant(x, r);
            std::vector<P> fixed;
            for (const auto& eq : eqs) fixed.push_back(map_into(eq, x, images));
            for (auto& o : orbits(fixed, rest, x)) {
                o.point[v] = constant(*o.field, r);
                std::vector<P> ideal{sub(gen(x, v), constant(x, r))};
                for (auto& g : o.ideal) {
                    bool is_v = g.length() == 1 && degree(g.exponent(0), x.n) == 1 && g.exponent(0)[v] == 1;
                    if (!is_v) ideal.push_back(g);
                }
                o.ideal = std::move(ideal);
                out.push_back(std::move(o));
            }
        } else if (x.d <= 1) {
            fmpq_poly_t minimal;
            fmpq_poly_init(minimal);
            fmpq_poly_set_fmpz_poly(minimal, f);
            CtxPtr inner = make_field(x.n, minimal);
            fmpq_poly_clear(minimal);
            std::vector<P> images;
            for (int i = 0; i < x.n; ++i) images.push_back(i == v ? gen(*inner, inner->n) : gen(*inner, i));
            std::vector<P> moved;
            for (const auto& eq : eqs) moved.push_back(map_into(eq, *inner, images));
            // Back to the old ring: the generator of the inner field is v there.
            std::vector<P> back;
            for (int i = 0; i < inner->nv; ++i) back.push_back(i == inner->n ? gen(x, v) : gen(x, i));
            for (auto& o : orbits(moved, rest, *inner)) {
                o.point[v] = o.theta;
                std::vector<P> ideal{fp};
                for (const auto& g : o.ideal) {
                    P h = map_into(g, x, back);
                    bool is_v = h.length() == 1 && degree(h.exponent(0), x.n) == 1 && h.exponent(0)[v] == 1;
                    if (!h.zero() && !is_v) ideal.push_back(std::move(h));
                }
                o.ideal = std::move(ideal);
                o.theta = constant(*o.field, Q(0));
                out.push_back(std::move(o));
            }
        } else {
            auto split = equations;
            split.push_back(fp);
            for (auto& o : orbits_by_form(split, vars, x)) out.push_back(std::move(o));
        }
    }
    fmpz_poly_factor_clear(factors);
    fmpz_poly_clear(u);
    return out;
}

// The Galois orbits of V(equations) by a separating linear form g = L.
std::vector<Orbit> orbits_by_form(const std::vector<P>& equations, const std::vector<int>& vars, const Ctx& x) {
    check_deadline();
    std::vector<P> eqs;
    for (const auto& e : equations)
        if (!e.zero()) eqs.push_back(e);
    for (const auto& e : eqs)
        if (order(e) == 0) {
            // A nonzero element of the field among the equations: no point.
            bool only_field = true;
            for (slong k = 0; k < e.length(); ++k)
                if (degree(e.exponent(k), x.n) != 0) only_field = false;
            if (only_field) return {};
        }
    std::vector<int> all = vars;  // the variables solved for, theta last
    if (x.d > 1) all.push_back(x.n);
    const int k = static_cast<int>(all.size()) + 1;  // and the separating form g
    if (k > MAX_VARIABLES) throw Declined{"Too many variables for the point search."};
    // The coordinates fixed at zero belong to the ideal of every orbit.
    std::vector<P> fixed_zero;
    for (int i = 0; i < x.n; ++i)
        if (std::find(vars.begin(), vars.end(), i) == vars.end()) fixed_zero.push_back(gen(x, i));
    if (vars.empty() && eqs.empty() && x.d <= 1) {
        Orbit o{make_field(x.n, nullptr), constant(x, Q(0)), {}, fixed_zero};
        for (int i = 0; i < x.n; ++i) o.point.push_back(constant(*o.field, Q(0)));
        o.theta = constant(*o.field, Q(0));
        return {o};
    }
    for (long attempt = 0; attempt < 12; ++attempt) {
        // A separating form L = theta + sum c_j v_j with small coefficients.
        auto c = atlas::separating_weights(all.size(), attempt);
        Ring R;
        R.init(k, 1);
        std::vector<Poly> input;
        for (const auto& e : eqs) input.push_back(engine_poly(R, engine_row(e, all, 0)));
        if (x.d > 1) {
            P m(x);
            fmpq_mpoly_set(m.p, x.modulus, x.c);
            input.push_back(engine_poly(R, engine_row(m, all, 0)));
        }
        {
            // g - L
            std::string text = "1";
            for (size_t j = 0; j < all.size(); ++j) text += " 0";
            text += " 1";
            for (size_t j = 0; j < all.size(); ++j) {
                text += ";" + std::to_string(-c[j]);
                for (size_t i = 0; i < all.size(); ++i) text += i == j ? " 1" : " 0";
                text += " 0";
            }
            input.push_back(engine_poly(R, text));
        }
        Engine engine;
        engine.R = R;
        engine.compute(input, 0);
        if (engine.unit) return {};
        std::vector<long> e(k);
        std::vector<bool> pure(k, false);
        for (const Element& g : engine.G) {
            R.unpack(g.p.m.data(), e.data());
            int nonzero = 0, last = -1;
            for (int j = 0; j < k; ++j)
                if (e[j]) ++nonzero, last = j;
            if (nonzero == 1) pure[last] = true;
        }
        if (!std::all_of(pure.begin(), pure.end(), [](bool b) { return b; }))
            throw Declined{"A locus expected to be finite has positive dimension."};
        const Poly* eliminant = nullptr;
        for (const Element& g : engine.G) {
            bool only = true;
            for (size_t t = 0; t < g.p.size() && only; ++t) {
                R.unpack(g.p.m.data() + t * R.stride, e.data());
                for (int j = 0; j < k - 1; ++j)
                    if (e[j]) only = false;
            }
            if (only) eliminant = &g.p;
        }
        if (!eliminant) throw Declined{"No eliminant in a finite basis."};
        fmpz_poly_t u;
        fmpz_poly_init(u);
        for (size_t t = 0; t < eliminant->size(); ++t) {
            R.unpack(eliminant->m.data() + t * R.stride, e.data());
            fmpz_poly_set_coeff_fmpz(u, e[k - 1], &eliminant->c[t].v);
        }
        fmpz_poly_factor_t factors;
        fmpz_poly_factor_init(factors);
        fmpz_poly_factor(factors, u);
        std::vector<Orbit> out;
        bool separated = true;
        for (slong i = 0; i < factors->num && separated; ++i) {
            const fmpz_poly_struct* f = factors->p + i;
            // The basis with f(g): a separating form gives the shape v - phi_v(g).
            std::vector<Poly> more;
            for (const Element& g : engine.G) more.push_back(g.p);
            {
                std::string text;
                for (slong t = 0; t <= fmpz_poly_degree(f); ++t) {
                    if (fmpz_is_zero(f->coeffs + t)) continue;
                    if (!text.empty()) text += ';';
                    Zi z;
                    fmpz_get_mpz(z.get_mpz_t(), f->coeffs + t);
                    text += z.get_str();
                    for (size_t j = 0; j < all.size(); ++j) text += " 0";
                    text += ' ' + std::to_string(t);
                }
                more.push_back(engine_poly(R, text));
            }
            Engine part;
            part.R = R;
            part.compute(more, 0);
            if (part.unit) continue;
            // phi_v for each solved variable, as rationals in powers of g.
            std::vector<std::vector<Q>> phi(all.size());
            std::vector<bool> found(all.size(), false);
            for (const Element& g : part.G) {
                R.unpack(g.p.m.data(), e.data());
                int lead = -1, count = 0;
                for (int j = 0; j < k - 1; ++j)
                    if (e[j]) lead = j, count += static_cast<int>(e[j]);
                if (lead < 0 || count != 1) continue;
                // g.p = a*v + sum b_t g^t, with a constant.
                Zi a;
                fmpz_get_mpz(a.get_mpz_t(), &g.p.c[0].v);
                std::vector<Q> poly;
                bool linear = true;
                for (size_t t = 1; t < g.p.size(); ++t) {
                    R.unpack(g.p.m.data() + t * R.stride, e.data());
                    for (int j = 0; j < k - 1; ++j)
                        if (e[j]) linear = false;
                    Zi b;
                    fmpz_get_mpz(b.get_mpz_t(), &g.p.c[t].v);
                    if (static_cast<long>(poly.size()) <= e[k - 1]) poly.resize(e[k - 1] + 1, Q(0));
                    poly[e[k - 1]] = Q(-b, a);
                    poly[e[k - 1]].canonicalize();
                }
                if (!linear) continue;
                phi[lead] = poly;
                found[lead] = true;
            }
            if (!std::all_of(found.begin(), found.end(), [](bool b) { return b; })) {
                separated = false;
                break;
            }
            // The orbit's field: Q[g]/(f).
            fmpq_poly_t minimal;
            fmpq_poly_init(minimal);
            fmpq_poly_set_fmpz_poly(minimal, f);
            CtxPtr field = make_field(x.n, minimal);
            const Ctx& y = *field;
            Q root = 0;
            if (y.d <= 1) {
                Zi f0, f1;
                fmpz_get_mpz(f0.get_mpz_t(), f->coeffs + 0);
                fmpz_get_mpz(f1.get_mpz_t(), f->coeffs + 1);
                root = Q(-f0, f1);
                root.canonicalize();
            }
            fmpq_poly_clear(minimal);
            auto in_field = [&](const std::vector<Q>& coeffs) {
                if (y.d <= 1) {
                    Q v = 0, power = 1;
                    for (const auto& q : coeffs) v += q * power, power *= root;
                    return constant(y, v);
                }
                std::vector<std::pair<std::vector<ulong>, Q>> terms;
                for (size_t t = 0; t < coeffs.size(); ++t) {
                    std::vector<ulong> ex(y.nv, 0);
                    ex[y.n] = static_cast<ulong>(t);
                    terms.emplace_back(ex, coeffs[t]);
                }
                return from_terms(y, terms);
            };
            Orbit o{field, constant(y, Q(0)), {}, {}};
            for (int i = 0; i < x.n; ++i) o.point.push_back(constant(y, Q(0)));
            for (size_t j = 0; j < vars.size(); ++j) o.point[vars[j]] = in_field(phi[j]);
            if (x.d > 1) o.theta = in_field(phi[all.size() - 1]);
            // The prime ideal in the old ring: f(L) and v - phi_v(L).
            P L = constant(x, Q(0));
            for (size_t j = 0; j < all.size(); ++j) L = add(L, scale(gen(x, all[j]), Q(c[j])));
            auto at_L = [&](const std::vector<Q>& coeffs) {
                P out = constant(x, Q(0)), power = constant(x, Q(1));
                for (const auto& q : coeffs) {
                    out = add(out, scale(power, q));
                    power = mul(power, L);
                }
                return out;
            };
            std::vector<Q> fq;
            for (slong t = 0; t <= fmpz_poly_degree(f); ++t) {
                Zi z;
                fmpz_get_mpz(z.get_mpz_t(), f->coeffs + t);
                fq.push_back(Q(z));
            }
            o.ideal = fixed_zero;
            o.ideal.push_back(at_L(fq));
            for (size_t j = 0; j < all.size(); ++j) o.ideal.push_back(sub(gen(x, all[j]), at_L(phi[j])));
            out.push_back(std::move(o));
        }
        fmpz_poly_factor_clear(factors);
        fmpz_poly_clear(u);
        if (separated) return out;
    }
    throw Declined{"No separating linear form was found for a finite set."};
}

// Whether the orbit b lies on the orbit given by the ideal of a (both in the
// same old ring): its generators vanish at b's point.
bool on_orbit(const std::vector<P>& ideal_a, const Orbit& b) {
    std::vector<P> images = b.point;
    if (ideal_a.empty()) return false;
    const Ctx& old = ideal_a.front().ctx();
    if (old.d > 1) images.push_back(b.theta);
    for (const auto& g : ideal_a)
        if (!map_into(g, *b.field, images).zero()) return false;
    return true;
}

bool nonzero_in_orbit(const Orbit& o, int i) { return !o.point[i].zero(); }

// ---------------------------------------------------------------- centres

struct Centre {
    std::vector<Q> a;                     // the entries a_1 <= ... <= a_k
    std::vector<int> axes;                // their coordinates
    std::vector<std::vector<Q>> monomial; // generators of M, exponent vectors over the coordinates
    std::vector<P> images;                // old coordinates in the new ones
    int ambient = 0;                      // the first axes are those of the ambient equations
};

// The centre of f at the origin, with the boundary coordinates fixed,
// computed on the jet of f of degree `cap`: the coordinate changes have
// degree at most `cap`, and the caller applies them to f itself and checks
// that the centre is admissible for it.
Centre centre(const P& whole, const std::vector<P>& ambient, const std::vector<bool>& boundary, bool logarithmic,
              long cap, long fixed_cap) {
    const Ctx& x = whole.ctx();
    const int n = x.n;
    Centre out;
    for (int i = 0; i < n; ++i) out.images.push_back(gen(x, i));
    std::vector<bool> fixed = logarithmic ? boundary : std::vector<bool>(n, false);
    std::vector<int> remaining;
    for (int i = 0; i < n; ++i)
        if (!fixed[i]) remaining.push_back(i);
    const Jet jet{static_cast<ulong>(cap), static_cast<ulong>(fixed_cap), fixed};
    P f = truncate(whole, jet);
    // The equations of a smooth ambient variety come first: each has order
    // one, so each gives an entry 1 whose axis is its Tschirnhausen
    // coordinate, and what follows is the restriction to the ambient.
    std::vector<std::pair<P, Q>> collection;
    std::vector<bool> amb;
    for (const auto& A : ambient) {
        collection.emplace_back(truncate(A, jet), Q(1));
        amb.push_back(true);
    }
    collection.emplace_back(f, Q(1));
    amb.push_back(false);

    auto change = [&](int var, const P& image) {
        check_deadline();
        for (auto& [g, r] : collection) g = substitute_truncated(g, var, image, jet);
        for (auto& img : out.images) img = substitute_truncated(img, var, image, jet);
    };
    auto log_order = [&](const P& g) { return order(restrict_zero(g, fixed)); };

    while (!remaining.empty()) {
        check_deadline();
        long best_index = -1;
        Q best;
        for (size_t k = 0; k < collection.size(); ++k) {
            const long d = log_order(collection[k].first);
            if (d < 0) continue;
            const Q value = Q(d) / collection[k].second;
            if (best_index < 0 || value < best) best = value, best_index = static_cast<long>(k);
        }
        if (best_index < 0) break;
        const Q ai = best;
        const ulong d = static_cast<ulong>(log_order(collection[best_index].first));
        if (d == 0) throw Declined{"A coefficient is a unit at the point."};
        if (amb[best_index] && d != 1)
            throw Declined{"The ambient variety is not smooth and transversal to the boundary at a point."};
        if (amb[best_index]) ++out.ambient;
        auto initial = [&]() { return homogeneous(restrict_zero(collection[best_index].first, fixed), d); };
        auto pure = [&](const P& h, int i) {
            std::vector<ulong> e(x.nv, 0);
            e[i] = d;
            return !field_coefficient(h, e).zero();
        };
        P h = initial();
        int axis = -1;
        for (int i : remaining)
            if (pure(h, i)) {
                axis = i;
                break;
            }
        if (axis < 0) {
            for (long size = 1; size <= 5 && axis < 0; ++size) {
                for (int i : remaining) {
                    std::vector<int> others;
                    for (int j : remaining)
                        if (j != i) others.push_back(j);
                    std::vector<long> c(others.size(), -size);
                    while (true) {
                        check_deadline();
                        if (std::any_of(c.begin(), c.end(), [&](long v) { return std::labs(v) == size; })) {
                            P test = h;
                            for (size_t t = 0; t < others.size(); ++t)
                                if (c[t]) test = substitute(test, others[t],
                                                            add(gen(x, others[t]), scale(gen(x, i), Q(c[t]))));
                            if (pure(test, i)) {
                                for (size_t t = 0; t < others.size(); ++t)
                                    if (c[t]) change(others[t], add(gen(x, others[t]), scale(gen(x, i), Q(c[t]))));
                                axis = i;
                                break;
                            }
                        }
                        size_t t = 0;
                        while (t < c.size() && c[t] == size) c[t++] = -size;
                        if (t == c.size()) break;
                        ++c[t];
                    }
                    if (axis >= 0) break;
                }
            }
            if (axis < 0) throw Declined{"No direction found for an initial form."};
        }
        // Tschirnhausen: remove the coefficient of x^(d-1) up to the order that
        // matters. Each shift x -> x + phi has phi free of x, so the shifts add
        // up: they are found on the element alone and applied to the rest once.
        P g = collection[best_index].first;
        P total(x);
        for (int step = 0; step < 64; ++step) {
            check_deadline();
            auto parts = expand(g, axis);
            auto lead = parts.find(d), below = parts.find(d - 1);
            if (below == parts.end() || below->second.zero()) break;
            std::vector<ulong> origin(x.nv, 0);
            if (lead == parts.end()) throw Declined{"The leading coefficient vanished."};
            const P c0 = field_coefficient(lead->second, origin);
            if (c0.zero()) throw Declined{"The leading coefficient vanished."};
            const Q need = collection[best_index].second - Q(long(d - 1)) / ai;
            if (need > 0) {
                const long lo = log_order(below->second);
                if (lo < 0 || Q(lo) / need > Q(cap)) break;
            }
            const P factor = scale(field_inverse(c0), Q(-1, long(d)));
            P phi = truncate(mul(below->second, factor), jet);
            g = substitute_truncated(g, axis, add(gen(x, axis), phi), jet);
            total = add(total, phi);
        }
        if (!total.zero()) change(axis, add(gen(x, axis), total));
        std::vector<std::pair<P, Q>> next;
        std::vector<bool> next_amb;
        for (size_t k = 0; k < collection.size(); ++k) {
            // The chosen ambient equation leaves only a residue beyond the jet.
            if (amb[k] && static_cast<long>(k) == best_index) continue;
            auto& [g, r] = collection[k];
            for (auto& [j, gj] : expand(g, axis)) {
                const Q req = r - Q(long(j)) / ai;
                if (req > 0 && !gj.zero()) {
                    next.emplace_back(std::move(gj), req);
                    next_amb.push_back(amb[k]);
                }
            }
        }
        collection = std::move(next);
        amb = std::move(next_amb);
        out.a.push_back(ai);
        out.axes.push_back(axis);
        remaining.erase(std::find(remaining.begin(), remaining.end(), axis));
    }
    if (logarithmic) {
        // Monomial part: the u-parts of the terms of what is left, scaled by 1/r.
        for (const auto& [g, r] : collection) {
            for (slong k = 0; k < g.length(); ++k) {
                const auto e = g.exponent(k);
                std::vector<Q> v(n, Q(0));
                bool any = false, free_term = true;
                for (int i = 0; i < n; ++i)
                    if (fixed[i] && e[i]) v[i] = Q(long(e[i])) / r, any = true;
                for (int i = 0; i < n; ++i)
                    if (!fixed[i] && e[i]) free_term = false;
                if (!any) {
                    if (free_term) throw Declined{"A remaining coefficient is a unit."};
                    continue;  // a term of a free coordinate outside the centre
                }
                out.monomial.push_back(std::move(v));
            }
        }
        std::vector<std::vector<Q>> minimal;
        for (size_t i = 0; i < out.monomial.size(); ++i) {
            bool dominated = false;
            for (size_t j = 0; j < out.monomial.size() && !dominated; ++j) {
                if (i == j) continue;
                bool le = true, eq = true;
                for (int c = 0; c < n; ++c) {
                    if (out.monomial[j][c] > out.monomial[i][c]) le = false;
                    if (out.monomial[j][c] != out.monomial[i][c]) eq = false;
                }
                if (le && (!eq || j < i)) dominated = true;
            }
            if (!dominated) minimal.push_back(out.monomial[i]);
        }
        out.monomial = std::move(minimal);
    }
    return out;
}

// ---------------------------------------------------------------- fans

using Vec = std::vector<Zi>;

Zi dot(const Vec& a, const Vec& b) {
    Zi s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

Vec primitive(Vec v) {
    Zi g = 0;
    for (auto& c : v) g = gcd(g, c);
    if (g > 1)
        for (auto& c : v) c /= g;
    return v;
}

// A basis vector of the kernel of the rows (each of length m), when the kernel is a line.
bool kernel_line(const std::vector<Vec>& rows, size_t m, Vec& out) {
    fmpz_mat_t A, N;
    fmpz_mat_init(A, static_cast<slong>(std::max<size_t>(rows.size(), 1)), static_cast<slong>(m));
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < m; ++j) fmpz_set_mpz(fmpz_mat_entry(A, i, j), rows[i][j].get_mpz_t());
    fmpz_mat_init(N, static_cast<slong>(m), static_cast<slong>(m));
    const slong dim = fmpz_mat_nullspace(N, A);
    bool ok = dim == 1;
    if (ok) {
        out.assign(m, 0);
        for (size_t j = 0; j < m; ++j) fmpz_get_mpz(out[j].get_mpz_t(), fmpz_mat_entry(N, j, 0));
        out = primitive(out);
    }
    fmpz_mat_clear(A);
    fmpz_mat_clear(N);
    return ok;
}

long rank(const std::vector<Vec>& rows, size_t m) {
    if (rows.empty()) return 0;
    fmpz_mat_t A;
    fmpz_mat_init(A, static_cast<slong>(rows.size()), static_cast<slong>(m));
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < m; ++j) fmpz_set_mpz(fmpz_mat_entry(A, i, j), rows[i][j].get_mpz_t());
    const long r = fmpz_mat_rank(A);
    fmpz_mat_clear(A);
    return r;
}

Zi determinant(const std::vector<Vec>& cols) {
    const size_t m = cols.size();
    fmpz_mat_t A;
    fmpz_mat_init(A, static_cast<slong>(m), static_cast<slong>(m));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < m; ++j) fmpz_set_mpz(fmpz_mat_entry(A, j, i), cols[i][j].get_mpz_t());
    fmpz_t d;
    fmpz_init(d);
    fmpz_mat_det(d, A);
    Zi out;
    fmpz_get_mpz(out.get_mpz_t(), d);
    fmpz_clear(d);
    fmpz_mat_clear(A);
    return out;
}

// Coordinates of v in the basis given by the columns `cols` (square, invertible).
std::vector<Q> coordinates(const std::vector<Vec>& cols, const Vec& v) {
    const size_t m = cols.size();
    fmpq_mat_t A, B, X;
    fmpq_mat_init(A, m, m);
    fmpq_mat_init(B, m, 1);
    fmpq_mat_init(X, m, 1);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) fmpz_set_mpz(fmpq_mat_entry_num(A, j, i), cols[i][j].get_mpz_t());
        fmpz_set_mpz(fmpq_mat_entry_num(B, i, 0), v[i].get_mpz_t());
    }
    fmpq_mat_solve(X, A, B);
    std::vector<Q> out(m);
    for (size_t i = 0; i < m; ++i) fmpq_get_mpq(out[i].get_mpq_t(), fmpq_mat_entry(X, i, 0));
    fmpq_mat_clear(A);
    fmpq_mat_clear(B);
    fmpq_mat_clear(X);
    return out;
}

struct Fan {
    size_t m = 0;
    std::vector<Vec> rays;
    std::vector<std::vector<size_t>> cones;  // maximal simplicial cones, ray indices
};

// The inner normals of the facets of conv(points) + R^m_{>=0}.
std::vector<Vec> facet_normals(const std::vector<Vec>& points, size_t m) {
    std::vector<Vec> normals;
    const size_t N = points.size();
    std::vector<size_t> pick;
    // Subsets S of points (by bitmask) and Z of coordinates with |S|-1+|Z| = m-1.
    if (N > 20) throw Declined{"Too many generators for the centre's Newton polyhedron."};
    for (unsigned long s = 1; s < (1UL << N); ++s) {
        const int size_s = __builtin_popcountl(s);
        if (size_s > static_cast<int>(m)) continue;
        const int size_z = static_cast<int>(m) - size_s;
        std::vector<size_t> S;
        for (size_t i = 0; i < N; ++i)
            if (s >> i & 1) S.push_back(i);
        for (unsigned long z = 0; z < (1UL << m); ++z) {
            if (__builtin_popcountl(z) != size_z) continue;
            std::vector<Vec> rows;
            for (size_t i = 1; i < S.size(); ++i) {
                Vec r(m);
                for (size_t j = 0; j < m; ++j) r[j] = points[S[i]][j] - points[S[0]][j];
                rows.push_back(r);
            }
            for (size_t j = 0; j < m; ++j)
                if (z >> j & 1) {
                    Vec r(m, 0);
                    r[j] = 1;
                    rows.push_back(r);
                }
            Vec w;
            if (!kernel_line(rows, m, w)) continue;
            bool positive = false, negative = false;
            for (const auto& c : w) {
                if (c > 0) positive = true;
                if (c < 0) negative = true;
            }
            if (positive && negative) continue;
            if (negative)
                for (auto& c : w) c = -c;
            const Zi at = dot(w, points[S[0]]);
            bool supporting = true;
            for (size_t i = 0; i < N && supporting; ++i)
                if (dot(w, points[i]) < at) supporting = false;
            for (size_t i : S)
                if (dot(w, points[i]) != at) supporting = false;
            if (!supporting) continue;
            if (std::find(normals.begin(), normals.end(), w) == normals.end()) normals.push_back(w);
        }
    }
    return normals;
}

// The facets of the cone spanned by `rays` (indices into all), inside its span.
std::vector<std::vector<size_t>> cone_facets(const std::vector<Vec>& all, const std::vector<size_t>& rays, size_t m) {
    std::vector<Vec> cols;
    for (size_t r : rays) cols.push_back(all[r]);
    const long d = rank(cols, m);
    // A basis B of the span, from the rays.
    std::vector<Vec> basis;
    for (size_t r : rays) {
        auto test = basis;
        test.push_back(all[r]);
        if (rank(test, m) > static_cast<long>(basis.size())) basis.push_back(all[r]);
        if (static_cast<long>(basis.size()) == d) break;
    }
    std::vector<std::vector<size_t>> facets;
    std::vector<size_t> idx(rays.size());
    std::iota(idx.begin(), idx.end(), 0);
    // Every (d-1)-subset of independent rays spans a hyperplane of the span.
    std::vector<bool> choose(rays.size(), false);
    std::fill(choose.begin(), choose.begin() + std::min<size_t>(rays.size(), static_cast<size_t>(d - 1)), true);
    std::sort(choose.begin(), choose.end(), std::greater<bool>());
    do {
        std::vector<Vec> sub;
        for (size_t i = 0; i < rays.size(); ++i)
            if (choose[i]) sub.push_back(all[rays[i]]);
        if (rank(sub, m) != d - 1) continue;
        // n = B y with (g . B) y = 0 for the chosen rays g.
        std::vector<Vec> rows;
        for (const auto& g : sub) {
            Vec r(basis.size());
            for (size_t b = 0; b < basis.size(); ++b) r[b] = dot(g, basis[b]);
            rows.push_back(r);
        }
        Vec y;
        if (!kernel_line(rows, basis.size(), y)) continue;
        Vec normal(m, 0);
        for (size_t b = 0; b < basis.size(); ++b)
            for (size_t j = 0; j < m; ++j) normal[j] += y[b] * basis[b][j];
        int sign = 0;
        bool ok = true;
        std::vector<size_t> on;
        for (size_t r : rays) {
            const Zi s = dot(normal, all[r]);
            if (s == 0) {
                on.push_back(r);
                continue;
            }
            const int sg = s > 0 ? 1 : -1;
            if (sign == 0) sign = sg;
            else if (sign != sg) ok = false;
        }
        if (!ok) continue;
        std::sort(on.begin(), on.end());
        if (std::find(facets.begin(), facets.end(), on) == facets.end()) facets.push_back(on);
    } while (std::prev_permutation(choose.begin(), choose.end()));
    return facets;
}

// A pulling triangulation, with the rays pulled in increasing index order, so
// that cones meeting in a face triangulate it the same way.
std::vector<std::vector<size_t>> triangulate(const std::vector<Vec>& all, std::vector<size_t> rays, size_t m) {
    std::sort(rays.begin(), rays.end());
    std::vector<Vec> cols;
    for (size_t r : rays) cols.push_back(all[r]);
    const long d = rank(cols, m);
    if (static_cast<long>(rays.size()) == d) return {rays};
    const size_t apex = rays.front();
    std::vector<std::vector<size_t>> out;
    for (const auto& facet : cone_facets(all, rays, m)) {
        if (std::find(facet.begin(), facet.end(), apex) != facet.end()) continue;
        for (auto simplex : triangulate(all, facet, m)) {
            simplex.push_back(apex);
            std::sort(simplex.begin(), simplex.end());
            out.push_back(simplex);
        }
    }
    return out;
}

// The normal fan of conv(points) + R^m_{>=0}, simplicial.
Fan normal_fan(const std::vector<Vec>& points, size_t m) {
    Fan fan;
    fan.m = m;
    fan.rays = facet_normals(points, m);
    // Standard rays first, so that pulling prefers them.
    std::stable_sort(fan.rays.begin(), fan.rays.end(), [](const Vec& a, const Vec& b) {
        auto ones = [](const Vec& v) {
            int nz = 0;
            for (const auto& c : v) nz += c != 0;
            return nz;
        };
        return ones(a) < ones(b);
    });
    std::set<std::vector<size_t>> cones;
    for (const auto& p : points) {
        Zi best;
        std::vector<size_t> on;
        for (size_t r = 0; r < fan.rays.size(); ++r) {
            Zi lowest = dot(fan.rays[r], points[0]);
            for (const auto& q : points) lowest = std::min(lowest, dot(fan.rays[r], q));
            if (dot(fan.rays[r], p) == lowest) on.push_back(r);
        }
        std::vector<Vec> cols;
        for (size_t r : on) cols.push_back(fan.rays[r]);
        if (rank(cols, m) != static_cast<long>(m)) continue;  // not a vertex
        for (auto& s : triangulate(fan.rays, on, m)) cones.insert(s);
    }
    fan.cones.assign(cones.begin(), cones.end());
    return fan;
}

// Star subdivisions at lattice points of fundamental parallelepipeds until
// every cone is regular.
void regularize(Fan& fan) {
    const size_t m = fan.m;
    for (int guard = 0; guard < 2000; ++guard) {
        long bad = -1;
        Zi index;
        for (size_t c = 0; c < fan.cones.size(); ++c) {
            std::vector<Vec> cols;
            for (size_t r : fan.cones[c]) cols.push_back(fan.rays[r]);
            const Zi d = abs(determinant(cols));
            if (d > 1) {
                bad = static_cast<long>(c);
                index = d;
                break;
            }
        }
        if (bad < 0) return;
        if (index > 100000) throw Declined{"A cone of the centre's fan has too large an index."};
        std::vector<Vec> cols;
        for (size_t r : fan.cones[bad]) cols.push_back(fan.rays[r]);
        // The group Z^m / (cone lattice): fractional parts of coordinates of unit vectors.
        std::vector<std::vector<Q>> gens;
        for (size_t j = 0; j < m; ++j) {
            Vec e(m, 0);
            e[j] = 1;
            auto lam = coordinates(cols, e);
            for (auto& q : lam) {
                Zi fl;
                mpz_fdiv_q(fl.get_mpz_t(), q.get_num_mpz_t(), q.get_den_mpz_t());
                q -= fl;
            }
            gens.push_back(lam);
        }
        std::set<std::vector<Q>> seen;
        std::vector<std::vector<Q>> queue{std::vector<Q>(m, Q(0))};
        seen.insert(queue[0]);
        for (size_t at = 0; at < queue.size(); ++at)
            for (const auto& g : gens) {
                std::vector<Q> s(m);
                for (size_t i = 0; i < m; ++i) {
                    s[i] = queue[at][i] + g[i];
                    if (s[i] >= 1) s[i] -= 1;
                }
                if (seen.insert(s).second) queue.push_back(s);
            }
        std::vector<Q> chosen;
        Q height = -1;
        for (const auto& lam : queue) {
            Q h = 0;
            for (const auto& q : lam) h += q;
            if (h == 0) continue;
            if (height < 0 || h < height) height = h, chosen = lam;
        }
        Vec p(m, 0);
        for (size_t i = 0; i < m; ++i) {
            Q s = 0;
            for (size_t r = 0; r < m; ++r) s += chosen[r] * Q(cols[r][i]);
            if (s.get_den() != 1) throw Declined{"A parallelepiped point is not integral."};
            p[i] = s.get_num();
        }
        p = primitive(p);
        const size_t pi = fan.rays.size();
        fan.rays.push_back(p);
        std::vector<std::vector<size_t>> next;
        for (const auto& cone : fan.cones) {
            std::vector<Vec> cc;
            for (size_t r : cone) cc.push_back(fan.rays[r]);
            const auto lam = coordinates(cc, p);
            if (std::any_of(lam.begin(), lam.end(), [](const Q& q) { return q < 0; })) {
                next.push_back(cone);
                continue;
            }
            for (size_t i = 0; i < cone.size(); ++i) {
                if (lam[i] == 0) continue;
                auto s = cone;
                s[i] = pi;
                std::sort(s.begin(), s.end());
                next.push_back(s);
            }
        }
        fan.cones = std::move(next);
    }
    throw Declined{"The regular refinement did not finish."};
}

// ---------------------------------------------------------------- loci of positive dimension

// The coordinates of x, and theta last over a number field.
std::vector<int> ring_variables(const Ctx& x) {
    std::vector<int> all(x.n);
    std::iota(all.begin(), all.end(), 0);
    if (x.d > 1) all.push_back(x.n);
    return all;
}

// f in the variables `all` (theta is x.n), as a polynomial of the primary decomposition.
local::Polynomial to_local(const P& f, const std::vector<int>& all) {
    local::Polynomial out;
    for (slong k = 0; k < f.length(); ++k) {
        const auto e = f.exponent(k);
        local::Powers p(all.size());
        for (size_t j = 0; j < all.size(); ++j) p[j] = Zi(static_cast<unsigned long>(e[all[j]]));
        const Q c = f.coefficient(k);
        auto it = out.emplace(std::move(p), c);
        if (!it.second && (it.first->second += c) == 0) out.erase(it.first);
    }
    return out;
}

P from_local(const local::Polynomial& f, const std::vector<int>& all, const Ctx& x) {
    std::vector<std::pair<std::vector<ulong>, Q>> terms;
    for (const auto& [p, c] : f) {
        std::vector<ulong> e(x.nv, 0);
        for (size_t j = 0; j < all.size(); ++j) e[all[j]] = checked_order(p[j]);
        terms.emplace_back(std::move(e), c);
    }
    return from_terms(x, terms);
}

// Rows over the variables `all`, with the minimal polynomial of theta.
local::Basis local_rows(const std::vector<P>& gens, const std::vector<int>& all, const Ctx& x) {
    local::Basis rows;
    for (const auto& g : gens)
        if (!g.zero()) rows.push_back(to_local(g, all));
    if (x.d > 1) {
        P m(x);
        fmpq_mpoly_set(m.p, x.modulus, x.c);
        rows.push_back(to_local(m, all));
    }
    return rows;
}

// Whether V(prime) lies in V(ideal): every generator of `ideal` is in `prime`.
bool contains(const std::vector<P>& prime, const std::vector<P>& ideal) {
    if (prime.empty()) return std::all_of(ideal.begin(), ideal.end(), [](const P& g) { return g.zero(); });
    const Ctx& x = prime.front().ctx();
    const auto all = ring_variables(x);
    const auto gb = primary::reduced(local_rows(prime, all, x), all.size(), true);
    for (const auto& g : ideal)
        if (!g.zero() && !primary::member_lex(to_local(g, all), gb)) return false;
    return true;
}

struct Locus {
    std::vector<Orbit> points;
    std::vector<std::vector<P>> curves;  // prime ideals of positive dimension in the ring of x
};

// The Galois orbits of V(equations) in the coordinates `vars` (the others are
// zero), or, when it has positive dimension, its prime components over the
// field of x: the finite ones as orbits, the others as prime ideals.
Locus locus(const std::vector<P>& equations, const std::vector<int>& vars, const Ctx& x) {
    try {
        return {orbits(equations, vars, x), {}};
    } catch (const Declined& d) {
        if (d.reason.rfind("A locus expected to be finite", 0) != 0) throw;
    }
    std::vector<int> all = vars;
    if (x.d > 1) all.push_back(x.n);
    std::vector<P> fixed_zero;
    for (int i = 0; i < x.n; ++i)
        if (std::find(vars.begin(), vars.end(), i) == vars.end()) fixed_zero.push_back(gen(x, i));
    primary::Components parts;
    try {
        const auto rows = local_rows(equations, all, x);
        parts = primary::decompose(primary::radical_lex(rows, all.size()), all.size(), true);
    } catch (const std::exception& e) {
        throw Declined{std::string("The decomposition of a bad locus failed: ") + e.what()};
    }
    Locus out;
    for (const auto& part : parts) {
        check_deadline();
        std::vector<P> gens;
        for (const auto& f : part.prime) gens.push_back(from_local(f, all, x));
        if (primary::dimension_lex(part.prime, all.size()) <= 0) {
            for (auto& o : orbits(gens, vars, x)) out.points.push_back(std::move(o));
            continue;
        }
        for (const auto& z : fixed_zero) gens.push_back(z);
        out.curves.push_back(std::move(gens));
    }
    return out;
}

P determinant(const std::vector<std::vector<P>>& M) {
    const size_t k = M.size();
    if (k == 1) return M[0][0];
    P out = constant(M[0][0].ctx(), Q(0));
    for (size_t c = 0; c < k; ++c) {
        if (M[0][c].zero()) continue;
        std::vector<std::vector<P>> minor;
        for (size_t r = 1; r < k; ++r) {
            std::vector<P> line;
            for (size_t j = 0; j < k; ++j)
                if (j != c) line.push_back(M[r][j]);
            minor.push_back(std::move(line));
        }
        const P term = mul(M[0][c], determinant(minor));
        out = c % 2 ? sub(out, term) : add(out, term);
    }
    return out;
}

// ---------------------------------------------------------------- the resolution

struct Divisor {
    std::string label;
    Zi N, nu;  // order of f and log discrepancy plus one
};

struct Task {
    CtxPtr ctx;
    std::vector<P> state;               // [F, map_0, ..., map_{n-1}] over the original coordinates
    std::vector<P> ambient;             // equations of the smooth ambient variety
    std::vector<long> divisor;          // per coordinate: index into divisors, or -1
    std::vector<std::vector<P>> avoid;  // closed sets whose points other charts resolve
    std::string name;
    int depth = 0;
};

// A chart is the ambient variety V(ambient) in affine space (all of it when
// there are no ambient equations). It owns the points over the blown-up
// locus (the zeros of `fibre`) on its owned faces, except those blown up
// further and those in `avoid`.
struct Chart {
    CtxPtr ctx;
    std::string name;
    std::vector<P> state;
    std::vector<P> ambient;
    std::vector<long> divisor;
    std::vector<P> fibre;
    std::vector<int> toric;               // coordinates of the cone
    std::vector<std::vector<int>> owned;  // faces (sets of coordinates) searched here
    std::vector<Orbit> excluded;          // orbits of points blown up further
    std::vector<std::vector<P>> curves;   // loci of positive dimension blown up further
    std::vector<std::vector<P>> avoid;
    int depth = 0;
    Zi orbifold_order = 1;  // the order of the finite group of a cone that is not regular
};

struct Found {
    Centre use;
    P G;
    std::vector<P> GA;
    std::string attempts;
    long cap;
};

// The data of the blowup of a bad locus of positive dimension.
struct CurvePlan {
    std::vector<P> ideal;  // its prime ideal in the chart's ring
    Task task;             // the chart, in a ring with one more coordinate for a graph
    Centre centre;
    std::vector<P> G;      // F in the centre's coordinates
    std::vector<P> GA;     // the ambient equations in the centre's coordinates
    std::vector<bool> boundary;
    std::vector<P> region;
    std::vector<P> lift;   // the chart's variables, theta last, in the task's ring
    std::vector<Q> invariant;
    std::string route;
};

std::vector<bool> boundary_of(const std::vector<long>& divisor) {
    std::vector<bool> b(divisor.size());
    for (size_t i = 0; i < divisor.size(); ++i) b[i] = divisor[i] >= 0;
    return b;
}

struct Run {
    int n;  // the original coordinates
    bool logarithmic;
    bool regular;
    Clock::time_point deadline;
    std::vector<CtxPtr> contexts;
    std::vector<Divisor> divisors;
    std::vector<Chart> charts;
    Json::Value blowups = Json::Value(Json::arrayValue);
    long extensions = 0;  // orbits of more than one point
    int labels = 0;
    std::string last_route;  // how the centre being blown up was found
    unsigned long long seed = 0x9e3779b97f4a7c15ULL;

    long small_random() {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<long>((seed >> 33) % 7) - 3;
    }

    // The bad locus of a chart over the blown-up locus, among the points whose
    // zero coordinates on the toric part are exactly `face` and which lie on
    // the boundary coordinates `extra`: points where F has logarithmic order
    // at least two on the ambient variety.
    Locus bad_locus(const Chart& chart, const std::vector<int>& face, const std::vector<int>& extra) {
        const Ctx& x = *chart.ctx;
        const int n = x.n;
        std::vector<bool> zero(n, false);
        for (int i : face) zero[i] = true;
        for (int i : extra) zero[i] = true;
        std::vector<P> images;
        for (int i = 0; i < n; ++i) images.push_back(zero[i] ? constant(x, Q(0)) : gen(x, i));
        std::vector<int> cols;
        for (int m = 0; m < n; ++m)
            if (!(logarithmic && zero[m] && chart.divisor[m] >= 0)) cols.push_back(m);
        std::vector<P> generators = chart.ambient;
        generators.push_back(chart.state[0]);
        std::vector<std::vector<P>> jacobian;
        for (const auto& g : generators) {
            std::vector<P> line;
            for (int m : cols) line.push_back(compose(derivative(g, m), images));
            jacobian.push_back(std::move(line));
        }
        std::vector<P> eqs;
        for (const auto& g : generators) eqs.push_back(compose(g, images));
        // Logarithmic order at least two on V(ambient): the logarithmic
        // differential of F lies in the span of those of the ambient equations,
        // so the minors of size (equations + 1) vanish.
        const size_t k = generators.size();
        if (cols.size() >= k) {
            std::vector<size_t> pick(k);
            std::iota(pick.begin(), pick.end(), 0);
            while (true) {
                std::vector<std::vector<P>> M;
                for (size_t r = 0; r < k; ++r) {
                    std::vector<P> line;
                    for (size_t c : pick) line.push_back(jacobian[r][c]);
                    M.push_back(std::move(line));
                }
                eqs.push_back(determinant(M));
                long i = static_cast<long>(k) - 1;
                while (i >= 0 && pick[i] == cols.size() - k + static_cast<size_t>(i)) --i;
                if (i < 0) break;
                ++pick[i];
                for (size_t j = static_cast<size_t>(i) + 1; j < k; ++j) pick[j] = pick[j - 1] + 1;
            }
        }
        for (const auto& g : chart.fibre) eqs.push_back(compose(g, images));
        std::vector<int> vars;
        for (int i = 0; i < n; ++i)
            if (!zero[i]) vars.push_back(i);
        Locus found;
        try {
            found = locus(eqs, vars, x);
        } catch (const Declined& d) {
            std::string detail = d.reason + " In chart " + chart.name + " over a field of degree " +
                                 std::to_string(x.d) + ", zero coordinates {";
            for (int i = 0; i < n; ++i)
                if (zero[i]) detail += std::to_string(i) + " ";
            detail += "}, boundary {";
            for (int i = 0; i < n; ++i)
                if (chart.divisor[i] >= 0) detail += std::to_string(i) + " ";
            detail += "}, " + std::to_string(chart.ambient.size()) + " ambient equations; strict " +
                      row(chart.state[0]).substr(0, 400);
            throw Declined{detail};
        }
        Locus out;
        for (auto& o : found.points) {
            bool exact = true;
            for (int t : chart.toric)
                if (!zero[t] && !nonzero_in_orbit(o, t)) exact = false;
            if (exact) out.points.push_back(std::move(o));
        }
        // A locus of positive dimension belongs here when no other toric
        // coordinate and no other boundary coordinate vanishes on all of it.
        for (auto& c : found.curves) {
            bool exact = true;
            for (int i = 0; i < n && exact; ++i) {
                if (zero[i]) continue;
                const bool toric = std::find(chart.toric.begin(), chart.toric.end(), i) != chart.toric.end();
                if ((toric || chart.divisor[i] >= 0) && contains(c, {gen(x, i)})) exact = false;
            }
            if (exact) out.curves.push_back(std::move(c));
        }
        return out;
    }

    // The chart moved to an orbit's field and translated to its point.
    Task point_task(const Chart& chart, const Orbit& o, const std::string& name) const {
        const Ctx& x = *chart.ctx;
        std::vector<P> into;
        for (int i = 0; i < x.n; ++i) into.push_back(gen(*o.field, i));
        if (x.d > 1) into.push_back(o.theta);
        auto move = [&](const P& s) { return translate(map_into(s, *o.field, into), o.point); };
        Task t;
        t.ctx = o.field;
        for (const auto& s : chart.state) t.state.push_back(move(s));
        for (const auto& a : chart.ambient) t.ambient.push_back(move(a));
        for (const auto& av : chart.avoid) {
            std::vector<P> g;
            for (const auto& a : av) g.push_back(move(a));
            t.avoid.push_back(std::move(g));
        }
        t.divisor = chart.divisor;
        for (int i = 0; i < x.n; ++i)
            if (!o.point[i].zero()) t.divisor[i] = -1;  // not through the new point
        t.name = name;
        t.depth = chart.depth + 1;
        return t;
    }

    // Bad points and bad loci of positive dimension of a new chart: each
    // point becomes a task, each locus of positive dimension is blown up.
    void examine(Chart chart, std::vector<Task>& stack) {
        const Ctx& x = *chart.ctx;
        const int n = x.n;
        std::vector<int> outside;  // boundary coordinates outside the cone
        for (int i = 0; i < n; ++i)
            if (chart.divisor[i] >= 0 && std::find(chart.toric.begin(), chart.toric.end(), i) == chart.toric.end())
                outside.push_back(i);
        std::vector<Orbit> found;
        std::vector<std::vector<P>> curves;
        for (const auto& face : chart.owned) {
            for (unsigned long s = 0; s < (1UL << outside.size()); ++s) {
                std::vector<int> extra;
                for (size_t i = 0; i < outside.size(); ++i)
                    if (s >> i & 1) extra.push_back(outside[i]);
                Locus here = bad_locus(chart, face, extra);
                for (auto& o : here.points) {
                    bool skip = false;
                    for (const auto& av : chart.avoid)
                        if (on_orbit(av, o)) skip = true;
                    for (const auto& f : found)
                        if (f.field->d == o.field->d && on_orbit(f.ideal, o)) skip = true;
                    if (!skip) found.push_back(std::move(o));
                }
                for (auto& c : here.curves) {
                    bool skip = false;
                    for (const auto& av : chart.avoid)
                        if (contains(c, av)) skip = true;
                    for (const auto& d : curves)
                        if (contains(c, d) && contains(d, c)) skip = true;
                    if (!skip) curves.push_back(std::move(c));
                }
            }
        }
        // A locus inside another is blown up with it.
        auto maximal = [](std::vector<std::vector<P>>& loci) {
            std::vector<std::vector<P>> out;
            for (size_t i = 0; i < loci.size(); ++i) {
                bool inside = false;
                for (size_t j = 0; j < loci.size() && !inside; ++j)
                    if (j != i && contains(loci[i], loci[j]) && !(j > i && contains(loci[j], loci[i]))) inside = true;
                if (!inside) out.push_back(loci[i]);
            }
            loci = std::move(out);
        };
        maximal(curves);
        // A point on a bad locus of positive dimension is resolved with it.
        std::vector<Orbit> points;
        for (auto& o : found) {
            bool on = false;
            for (const auto& c : curves)
                if (on_orbit(c, o)) on = true;
            if (!on) points.push_back(std::move(o));
        }
        // A point may lie on a bad locus of positive dimension whose general
        // points lie on faces that other charts own; its centre is that of the
        // locus, blown up here for the points over this point only.
        std::vector<std::vector<P>> elsewhere;
        if (!points.empty() || !curves.empty()) {
            // Toric coordinates that vanish on the whole fibre: a fibre equation
            // is a power of one of them.
            std::vector<int> forced;
            for (int t : chart.toric)
                for (const auto& g : chart.fibre) {
                    if (g.length() != 1) continue;
                    const auto e = g.exponent(0);
                    bool pure = e[t] > 0;
                    for (int i = 0; i < x.nv; ++i)
                        if (i != t && e[i]) pure = false;
                    if (pure && std::find(forced.begin(), forced.end(), t) == forced.end()) forced.push_back(t);
                }
            const size_t m = chart.toric.size();
            for (unsigned long s = 0; s < (1UL << m); ++s) {
                std::vector<int> face;
                for (size_t i = 0; i < m; ++i)
                    if (s >> i & 1) face.push_back(chart.toric[i]);
                std::sort(face.begin(), face.end());
                if (face.empty()) continue;
                bool holds = true;  // a face without the forced coordinates has no point of the fibre
                for (int t : forced)
                    if (std::find(face.begin(), face.end(), t) == face.end()) holds = false;
                if (!holds) continue;
                if (std::find(chart.owned.begin(), chart.owned.end(), face) != chart.owned.end()) continue;
                for (unsigned long e = 0; e < (1UL << outside.size()); ++e) {
                    std::vector<int> extra;
                    for (size_t i = 0; i < outside.size(); ++i)
                        if (e >> i & 1) extra.push_back(outside[i]);
                    Locus here;
                    try {
                        here = bad_locus(chart, face, extra);
                    } catch (const Declined&) {
                        continue;  // a face without a usable search has no locus for these points
                    }
                    for (auto& c : here.curves) {
                        bool seen = false;
                        for (const auto& d : elsewhere)
                            if (contains(c, d) && contains(d, c)) seen = true;
                        if (!seen) elsewhere.push_back(std::move(c));
                    }
                }
            }
        }
        maximal(elsewhere);
        // Loci found here that lie in a larger one found elsewhere.
        std::vector<std::vector<std::vector<P>>> inside_elsewhere(elsewhere.size());
        {
            std::vector<std::vector<P>> own;
            for (auto& c : curves) {
                long at = -1;
                for (size_t e = 0; e < elsewhere.size() && at < 0; ++e)
                    if (contains(c, elsewhere[e]) && !contains(elsewhere[e], c)) at = static_cast<long>(e);
                if (at < 0) own.push_back(std::move(c));
                else inside_elsewhere[static_cast<size_t>(at)].push_back(std::move(c));
            }
            curves = std::move(own);
        }
        std::vector<Orbit> isolated;
        std::vector<std::vector<Orbit>> on_elsewhere(elsewhere.size());
        for (auto& o : points) {
            long at = -1;
            for (size_t c = 0; c < elsewhere.size() && at < 0; ++c)
                if (on_orbit(elsewhere[c], o)) at = static_cast<long>(c);
            if (at < 0) isolated.push_back(std::move(o));
            else on_elsewhere[static_cast<size_t>(at)].push_back(std::move(o));
        }
        chart.excluded = isolated;
        for (const auto& list : on_elsewhere)
            for (const auto& o : list) chart.excluded.push_back(o);
        chart.curves = curves;
        for (const auto& list : inside_elsewhere)
            for (const auto& c : list) chart.curves.push_back(c);
        charts.push_back(chart);
        for (size_t q = 0; q < isolated.size(); ++q) {
            if (isolated[q].field->d > 1) ++extensions;
            Task t = point_task(chart, isolated[q], chart.name + "#" + std::to_string(q));
            if (t.depth > 60) throw Declined{"The resolution is deeper than allowed."};
            stack.push_back(std::move(t));
        }
        if (!curves.empty()) blow_up_curves(chart, curves, stack);
        for (size_t c = 0; c < elsewhere.size(); ++c) {
            if (on_elsewhere[c].empty() && inside_elsewhere[c].empty()) continue;
            if (chart.depth + 1 > 60) throw Declined{"The resolution is deeper than allowed."};
            CurvePlan plan = plan_curve(chart, elsewhere[c], 100 + c);
            for (size_t q = 0; q < inside_elsewhere[c].size(); ++q) {
                std::vector<P> region;
                for (const auto& g : inside_elsewhere[c][q]) region.push_back(map_into(g, *plan.task.ctx, plan.lift));
                Task t = plan.task;
                t.name = chart.name + "~" + std::to_string(q) + " in a larger locus";
                last_route = plan.route;
                finish(t, plan.centre, plan.G.front(), plan.GA, plan.boundary, region, true, stack);
            }
            for (size_t q = 0; q < on_elsewhere[c].size(); ++q) {
                const Orbit& o = on_elsewhere[c][q];
                if (o.field->d > 1) ++extensions;
                std::vector<P> region;
                for (const auto& g : o.ideal) region.push_back(map_into(g, *plan.task.ctx, plan.lift));
                Task t = plan.task;
                t.name = chart.name + "#" + std::to_string(isolated.size() + q) + " on a locus";
                last_route = plan.route;
                finish(t, plan.centre, plan.G.front(), plan.GA, plan.boundary, region, true, stack);
            }
        }
    }

    // The centre at the task's point, read from a jet that grows until the
    // centre is admissible for F and the ambient equations themselves.
    Found admissible_centre(const Task& task, int tries, Centre* last) {
        const auto boundary = boundary_of(task.divisor);
        long cap = std::max<long>(8, checked_order(2 * Zi(order(task.state[0])) + 2)), fixed_cap = cap;
        std::string attempts;
        auto ceiling = [](const Q& q) {
            Zi c;
            mpz_cdiv_q(c.get_mpz_t(), q.get_num_mpz_t(), q.get_den_mpz_t());
            return checked_order(c);
        };
        for (int attempt = 0;; ++attempt) {
            if (attempt == tries) throw Declined{"No admissible centre within the jets tried: " + attempts};
            Centre trial = centre(task.state[0], task.ambient, boundary, logarithmic, cap, fixed_cap);
            if (last) *last = trial;
            Q largest = 0, largest_fixed = 0;
            for (const auto& q : trial.a) largest = std::max(largest, q);
            for (const auto& g : trial.monomial) {
                Q total = 0;
                for (const auto& q : g) total += q;
                largest_fixed = std::max(largest_fixed, total);
            }
            attempts += "[jet " + std::to_string(cap) + "/" + std::to_string(fixed_cap) + " largest " +
                        largest.get_str() + "/" + largest_fixed.get_str();
            const bool small_free = Q(cap) < 2 * largest + 2, small_fixed = Q(fixed_cap) < 2 * largest_fixed + 2;
            if (small_free || small_fixed) {
                attempts += " small]";
                if (small_free) cap = ceiling(2 * largest + 2);
                if (small_fixed) fixed_cap = ceiling(2 * largest_fixed + 2);
                continue;
            }
            long outside_degree = -1;
            // The trimmed coordinates give the same centre up to integral closure;
            // the untrimmed ones are the fallback.
            for (int version = 0; version < 2; ++version) {
                Centre use = trial;
                if (version == 0) use.images = trimmed(trial);
                P G = compose(task.state[0], use.images);
                std::vector<P> GA;
                for (const auto& A : task.ambient) GA.push_back(compose(A, use.images));
                long outside = admissible(G, use, boundary);
                for (const auto& A : GA) {
                    const long o = admissible(A, use, boundary);
                    if (o >= 0) outside = outside < 0 ? o : std::min(outside, o);
                }
                const bool ok = outside < 0;
                if (!ok) outside_degree = outside_degree < 0 ? outside : std::min(outside_degree, outside);
                attempts += std::string(version == 0 ? " trimmed" : " full") + (ok ? " ok" : " inadmissible");
                if (!ok) continue;
                attempts += "]";
                return Found{std::move(use), std::move(G), std::move(GA), attempts, cap};
            }
            attempts += "]";
            // A jet that holds the monomials the centre missed.
            const Zi needed = 2 * Zi(outside_degree) + 2;
            cap = checked_order(std::max<Zi>(2 * Zi(cap), needed));
            fixed_cap = checked_order(std::max<Zi>(2 * Zi(fixed_cap), needed));
        }
    }

    void process(Task task, std::vector<Task>& stack) {
        check_deadline();
        Found f = admissible_centre(task, 16, nullptr);
        last_route = "jet of degree " + std::to_string(f.cap);
        std::vector<P> region;
        for (int i = 0; i < task.ctx->n; ++i) region.push_back(gen(*task.ctx, i));
        finish(task, f.use, f.G, f.GA, boundary_of(task.divisor), region, false, stack);
    }

    // F written in new coordinates w_k standing for the polynomials X_k:
    // F = r + sum_k q_k X_k by division, recursively in the quotients.
    static P adic(const P& F, const std::vector<P>& X, const std::vector<int>& w, int depth) {
        if (F.zero()) return F;
        if (depth > 512) throw Declined{"The expansion along a bad locus does not end."};
        const Ctx& y = F.ctx();
        std::vector<P> quotients;
        for (size_t k = 0; k < X.size(); ++k) quotients.emplace_back(y);
        std::vector<fmpq_mpoly_struct*> qs;
        for (auto& q : quotients) qs.push_back(q.p);
        std::vector<P> copies = X;
        std::vector<fmpq_mpoly_struct*> bs;
        for (auto& b : copies) bs.push_back(b.p);
        P r(y);
        fmpq_mpoly_divrem_ideal(qs.data(), r.p, F.p, bs.data(), static_cast<slong>(bs.size()), y.c);
        reduce(r);
        P out = r;
        for (size_t k = 0; k < X.size(); ++k) {
            reduce(quotients[k]);
            if (quotients[k].zero()) continue;
            out = add(out, mul(gen(y, w[k]), adic(quotients[k], X, w, depth + 1)));
        }
        return out;
    }

    // The centre at the generic point of {N = 0, u = 0} (u the fixed
    // coordinates) of a polynomial whose coefficients in the other
    // coordinates, the parameters, are units there: the algorithm of `centre`
    // with orders and initial forms in N alone. It makes no change of
    // coordinates, so it fails when an initial form has no pure power of a
    // coordinate of N.
    struct Shift {
        P c;  // the coefficient of x^(d-1) in the element that gave the axis x
        Q a;
    };
    static bool generic_centre(const P& F, const std::vector<int>& N, const std::vector<bool>& fixed, Centre& out,
                               std::vector<Shift>& shifts) {
        const Ctx& x = F.ctx();
        const int n = x.n;
        auto degree_in = [&](const std::vector<ulong>& e) {
            ulong s = 0;
            for (int i : N) s += e[i];
            return s;
        };
        auto log_order = [&](const P& g) {
            long best = -1;
            const P h = restrict_zero(g, fixed);
            for (slong k = 0; k < h.length(); ++k) {
                const long d = static_cast<long>(degree_in(h.exponent(k)));
                if (best < 0 || d < best) best = d;
            }
            return best;
        };
        std::vector<int> remaining = N;
        std::vector<std::pair<P, Q>> collection;
        collection.emplace_back(F, Q(1));
        while (!remaining.empty()) {
            check_deadline();
            long best_index = -1;
            Q best;
            for (size_t k = 0; k < collection.size(); ++k) {
                const long d = log_order(collection[k].first);
                if (d < 0) continue;
                const Q value = Q(d) / collection[k].second;
                if (best_index < 0 || value < best) best = value, best_index = static_cast<long>(k);
            }
            if (best_index < 0) break;
            const long d = log_order(collection[best_index].first);
            if (d == 0) return false;
            const P h = restrict_zero(collection[best_index].first, fixed);
            int axis = -1;
            for (int i : remaining) {
                for (slong k = 0; k < h.length() && axis < 0; ++k) {
                    const auto e = h.exponent(k);
                    if (static_cast<long>(e[i]) == d && static_cast<long>(degree_in(e)) == d) axis = i;
                }
                if (axis >= 0) break;
            }
            if (axis < 0) return false;
            {
                auto parts = expand(collection[best_index].first, axis);
                if (parts.count(static_cast<ulong>(d - 1))) shifts.push_back({parts.at(static_cast<ulong>(d - 1)), best});
            }
            std::vector<std::pair<P, Q>> next;
            for (auto& [g, r] : collection)
                for (auto& [j, gj] : expand(g, axis)) {
                    const Q req = r - Q(long(j)) / best;
                    if (req > 0 && !gj.zero()) next.emplace_back(std::move(gj), req);
                }
            collection = std::move(next);
            out.a.push_back(best);
            out.axes.push_back(axis);
            remaining.erase(std::find(remaining.begin(), remaining.end(), axis));
        }
        for (const auto& [g, r] : collection) {
            for (slong k = 0; k < g.length(); ++k) {
                const auto e = g.exponent(k);
                std::vector<Q> v(n, Q(0));
                bool any = false;
                for (int i = 0; i < n; ++i)
                    if (fixed[i] && e[i]) v[i] = Q(long(e[i])) / r, any = true;
                if (!any) {
                    if (degree_in(e) == 0) return false;  // a unit at the generic point
                    continue;
                }
                out.monomial.push_back(std::move(v));
            }
        }
        std::vector<std::vector<Q>> minimal;
        for (size_t i = 0; i < out.monomial.size(); ++i) {
            bool dominated = false;
            for (size_t j = 0; j < out.monomial.size() && !dominated; ++j) {
                if (i == j) continue;
                bool le = true, eq = true;
                for (int c = 0; c < n; ++c) {
                    if (out.monomial[j][c] > out.monomial[i][c]) le = false;
                    if (out.monomial[j][c] != out.monomial[i][c]) eq = false;
                }
                if (le && (!eq || j < i)) dominated = true;
            }
            if (!dominated) minimal.push_back(out.monomial[i]);
        }
        out.monomial = std::move(minimal);
        return true;
    }

    // The centre along a bad locus Z of positive dimension, lying on the
    // boundary coordinates u. On its stratum {u = 0}, Z is cut out by a reduced
    // Groebner basis X_1, ..., X_c, c its codimension there. An X_k linear in a
    // coordinate of its own becomes that coordinate by a triangular change;
    // any other becomes a new coordinate w_k with the ambient equation
    // w_k = X_k, and F is written in powers of the w_k. Its coefficients are
    // then units at the generic point of Z, where the centre is read off in
    // the coordinates X with the parameters along Z as constants. The jet
    // centre at a general point of Z checks its entries a_i: those that do not
    // change with the jet are the invariant there (the others come from what
    // the jet leaves of coordinates along Z). An admissible centre with that
    // invariant is the centre of Abramovich, Temkin and Wlodarczyk along Z.
    CurvePlan plan_curve(const Chart& chart, const std::vector<P>& Z, size_t index) {
        const Ctx& x = *chart.ctx;
        const int n = x.n;
        if (!chart.ambient.empty())
            throw Declined{"A bad locus of positive dimension on a chart with ambient equations."};
        const auto all = ring_variables(x);
        const auto gbZ = primary::reduced(local_rows(Z, all, x), all.size(), true);
        const long dimZ = primary::dimension_lex(gbZ, all.size());
        auto vanishes = [&](const P& g) { return g.zero() || primary::member_lex(to_local(g, all), gbZ); };
        std::vector<int> free;
        std::vector<bool> u(n, false);
        const auto boundary = boundary_of(chart.divisor);
        for (int i = 0; i < n; ++i) {
            const bool v = vanishes(gen(x, i));
            if (!v) free.push_back(i);
            if (v && boundary[i] && logarithmic) u[i] = true;
        }
        if (free.empty() || dimZ < 1) throw Declined{"A bad locus of positive dimension without free coordinates."};
        auto special = [&](int i) {
            return chart.divisor[i] >= 0 || std::find(chart.toric.begin(), chart.toric.end(), i) != chart.toric.end();
        };
        // The normal coordinates of Z on its stratum.
        std::vector<P> zero_images;
        for (int i = 0; i < n; ++i) zero_images.push_back(u[i] ? constant(x, Q(0)) : gen(x, i));
        std::vector<P> on_stratum;
        for (const auto& z : Z) {
            P r = compose(z, zero_images);
            if (!r.zero()) on_stratum.push_back(std::move(r));
        }
        std::vector<P> X;
        for (const auto& f : primary::reduced(local_rows(on_stratum, all, x), all.size(), true)) {
            P h = from_local(f, all, x);
            bool only_theta = true;
            for (slong t = 0; t < h.length(); ++t)
                if (degree(h.exponent(t), n) != 0) only_theta = false;
            if (!only_theta) X.push_back(std::move(h));
        }
        long stratum = n;
        for (int i = 0; i < n; ++i) stratum -= u[i];
        if (static_cast<long>(X.size()) != stratum - dimZ)
            throw Declined{"A bad locus of positive dimension that is not a complete intersection on its stratum."};
        // Straighten what is linear in a coordinate of its own.
        auto images = identity_images(x);
        images.resize(n, constant(x, Q(0)));
        std::vector<int> N;
        std::vector<P> graphs;
        std::vector<bool> pivot(n, false);
        for (const auto& g : X) {
            int chosen = -1;
            for (int j = 0; j < n && chosen < 0; ++j) {
                if (pivot[j]) continue;
                auto parts = expand(g, j);
                if (parts.size() > 2 || !parts.count(1)) continue;
                // A boundary coordinate stays itself: it is an axis only when g is it.
                if (boundary[j] && (parts.size() != 1 || parts.at(1).length() != 1)) continue;
                const P& c1 = parts.at(1);
                if (c1.length() != 1) continue;
                const auto e = c1.exponent(0);
                if (std::any_of(e.begin(), e.end(), [](ulong v) { return v != 0; })) continue;
                bool alone = true;  // no other generator involves x_j
                for (const auto& h : X)
                    if (&h != &g) {
                        auto ph = expand(h, j);
                        if (ph.size() > 1 || (ph.size() == 1 && ph.begin()->first != 0)) alone = false;
                    }
                const P r = parts.count(0) ? parts.at(0) : constant(x, Q(0));
                for (int k = 0; k < n; ++k)
                    if (pivot[k] && expand(r, k).size() > 1) alone = false;
                if (!alone) continue;
                chosen = j;
                images[j] = scale(sub(gen(x, j), r), 1 / c1.coefficient(0));
            }
            if (chosen >= 0) {
                pivot[chosen] = true;
                N.push_back(chosen);
            } else {
                graphs.push_back(g);
            }
        }
        // New coordinates for the rest.
        const int added = static_cast<int>(graphs.size());
        CtxPtr y = added ? make_field(n + added, x.d > 1 ? x.minimal : nullptr) : chart.ctx;
        std::vector<P> lift;
        for (int i = 0; i < n; ++i) lift.push_back(gen(*y, i));
        if (x.d > 1) lift.push_back(gen(*y, y->n));
        auto up = [&](const P& f) { return added ? map_into(f, *y, lift) : f; };
        std::vector<int> w;
        std::vector<P> X_up;
        for (int k = 0; k < added; ++k) {
            w.push_back(n + k);
            N.push_back(n + k);
            X_up.push_back(up(graphs[static_cast<size_t>(k)]));
        }
        Centre gc;
        for (int i = 0; i < n; ++i) gc.images.push_back(up(images[i]));
        for (int k = 0; k < added; ++k) gc.images.push_back(gen(*y, n + k));
        P F = compose(up(chart.state[0]), gc.images);
        if (added) F = adic(F, X_up, w, 0);
        std::vector<bool> fixed = u, boundary_y = boundary;
        fixed.resize(n + added, false);
        boundary_y.resize(n + added, false);
        std::vector<Shift> shifts;
        if (!generic_centre(F, N, fixed, gc, shifts))
            throw Declined{"No centre at the generic point of a bad locus in its normal coordinates."};
        std::vector<Q> got = gc.a;
        // Without the Tschirnhausen changes the centre is the same when each
        // change x_i -> x_i - s, where s has the monomials of the coefficient of
        // x_i^(d-1), has J-order above the weight 1/a_i of x_i: the terms it
        // adds then lie strictly inside the Newton polyhedron of J and cannot
        // cancel the terms on its boundary, which give the later entries and M.
        bool safe = true;
        {
            const auto toric = centre_coordinates(gc, boundary_y);
            const size_t m = toric.size();
            Zi common;
            const auto pts = centre_points(gc, toric, common);
            const auto normals = facet_normals(pts, m);
            for (const auto& sh : shifts) {
                for (slong k = 0; k < sh.c.length() && safe; ++k) {
                    const auto e = sh.c.exponent(k);
                    Q order = -1;
                    for (const auto& wv : normals) {
                        Zi lowest = dot(wv, pts[0]);
                        for (const auto& p : pts) lowest = std::min(lowest, dot(wv, p));
                        if (lowest <= 0) continue;
                        Zi sv = 0;
                        for (size_t j = 0; j < m; ++j) sv += wv[j] * Zi(static_cast<unsigned long>(e[toric[j]]));
                        const Q ratio = Q(sv * common, lowest);
                        if (order < 0 || ratio < order) order = ratio;
                    }
                    if (order >= 0 && !(order > 1 / sh.a)) {
                        safe = false;
                    }
                }
            }
        }
        std::vector<Q> invariant = got;
        long cap = 0;
        bool residue_free = true;
        if (!safe) {
            // The check at a general point, with jets just large enough for these entries.
            Q largest = 0;
            for (const auto& q : got) largest = std::max(largest, q);
            for (const auto& v : gc.monomial) {
                Q total = 0;
                for (const auto& q : v) total += q;
                largest = std::max(largest, total);
            }
            // A general point: Z cut by random hyperplanes.
            std::vector<int> every(n);
            std::iota(every.begin(), every.end(), 0);
            std::optional<Orbit> point;
            for (int attempt = 0; attempt < 12 && !point; ++attempt) {
                check_deadline();
                std::vector<P> eqs = Z;
                for (long j = 0; j < dimZ; ++j) {
                    P L = constant(x, Q(small_random()));
                    bool any = false;
                    for (int i : free) {
                        const long c = small_random();
                        if (c) L = add(L, scale(gen(x, i), Q(c))), any = true;
                    }
                    if (!any) L = add(L, gen(x, free[static_cast<size_t>(j) % free.size()]));
                    eqs.push_back(L);
                }
                std::vector<Orbit> pts;
                try {
                    pts = orbits(eqs, every, x);
                } catch (const Declined&) {
                    continue;
                }
                for (auto& o : pts) {
                    bool general = true;
                    for (int i : free)
                        if (special(i) && !nonzero_in_orbit(o, i)) general = false;
                    if (general && (!point || o.field->d < point->field->d)) point = std::move(o);
                }
            }
            if (!point) throw Declined{"No general point found on a bad locus of positive dimension."};
            if (point->field->d > 1) ++extensions;
            Task tp = point_task(chart, *point, chart.name + "~" + std::to_string(index));
            const auto boundary_p = boundary_of(tp.divisor);
            Zi top;
            mpz_cdiv_q(top.get_mpz_t(), largest.get_num_mpz_t(), largest.get_den_mpz_t());
            cap = std::max<long>(4, checked_order(2 * top + 2));
            const Centre near = centre(tp.state[0], tp.ambient, boundary_p, logarithmic, cap, cap);
            const Centre far = centre(tp.state[0], tp.ambient, boundary_p, logarithmic, cap + 4, cap + 4);
            invariant.clear();
            for (size_t k = 0; k < std::min(near.a.size(), far.a.size()) && near.a[k] == far.a[k]; ++k)
                invariant.push_back(near.a[k]);
            residue_free = near.a == far.a && near.monomial.size() == far.monomial.size();
            if (got != invariant)
                throw Declined{"The centre in the normal coordinates of a bad locus does not have its invariant."};
            if (residue_free) {
                // The monomial parts agree too.
                auto key = [&](const std::vector<std::vector<Q>>& gens) {
                    std::set<std::vector<Q>> out;
                    for (const auto& v : gens) out.emplace(v.begin(), v.begin() + n);
                    return out;
                };
                if (key(near.monomial) != key(gc.monomial))
                    throw Declined{"The centre in the normal coordinates of a bad locus has another monomial part."};
            } else if (N.size() == 1) {
                throw Declined{"A degenerate initial edge along a bad locus of positive dimension."};
            }
        }
        if (admissible(F, gc, boundary_y) >= 0)
            throw Declined{"The centre along a bad locus is not admissible."};
        CurvePlan plan;
        plan.ideal = Z;
        plan.invariant = invariant;
        Task t;
        t.ctx = y;
        t.state.push_back(F);
        for (size_t i = 1; i < chart.state.size(); ++i) t.state.push_back(up(chart.state[i]));
        for (int k = 0; k < added; ++k) t.ambient.push_back(sub(gen(*y, w[k]), X_up[k]));
        t.divisor = chart.divisor;
        t.divisor.resize(n + added, -1);
        for (const auto& av : chart.avoid) {
            std::vector<P> h;
            for (const auto& a : av) h.push_back(up(a));
            t.avoid.push_back(std::move(h));
        }
        t.name = chart.name + "~" + std::to_string(index);
        t.depth = chart.depth + 1;
        plan.task = t;
        plan.centre = gc;
        plan.G = {F};
        // The ambient equations in the centre's coordinates: w_k = X_k(images).
        for (int k = 0; k < added; ++k) plan.GA.push_back(compose(t.ambient[k], gc.images));
        plan.boundary = boundary_y;
        for (const auto& z : Z) plan.region.push_back(up(z));
        plan.lift = added ? lift : identity_images(x);
        plan.route = std::to_string(N.size() - static_cast<size_t>(added)) + " straightened and " +
                     std::to_string(added) + " new normal coordinates";
        return plan;
    }

    // Loci with larger invariants first; a point where a locus meets one blown
    // up before belongs to that one.
    void blow_up_curves(const Chart& chart, const std::vector<std::vector<P>>& curves, std::vector<Task>& stack) {
        if (chart.depth + 1 > 60) throw Declined{"The resolution is deeper than allowed."};
        std::vector<CurvePlan> plans;
        for (size_t i = 0; i < curves.size(); ++i) plans.push_back(plan_curve(chart, curves[i], i));
        std::stable_sort(plans.begin(), plans.end(), [](const CurvePlan& a, const CurvePlan& b) {
            const auto& u = a.invariant;
            const auto& v = b.invariant;
            for (size_t k = 0; k < std::min(u.size(), v.size()); ++k)
                if (u[k] != v[k]) return u[k] > v[k];
            return u.size() < v.size();
        });
        std::vector<int> every(chart.ctx->n);
        std::iota(every.begin(), every.end(), 0);
        for (size_t i = 0; i < plans.size(); ++i) {
            auto& plan = plans[i];
            for (size_t j = 0; j < i; ++j) {
                std::vector<P> both = plan.ideal;
                both.insert(both.end(), plans[j].ideal.begin(), plans[j].ideal.end());
                Locus meet = locus(both, every, *chart.ctx);
                auto up = [&](const std::vector<P>& ideal) {
                    std::vector<P> h;
                    for (const auto& g : ideal) h.push_back(map_into(g, *plan.task.ctx, plan.lift));
                    return h;
                };
                for (const auto& o : meet.points) plan.task.avoid.push_back(up(o.ideal));
                for (const auto& c : meet.curves) plan.task.avoid.push_back(up(c));
            }
            last_route = plan.route;
            finish(plan.task, plan.centre, plan.G.front(), plan.GA, plan.boundary, plan.region, true, stack);
        }
    }

    // A coordinate of weight w (1/a on an axis of the centre) may change by terms
    // of weighted order at least w without changing the integral closure of J.
    // Each image keeps its linear part and the terms of weighted order below the
    // largest weight among the coordinates of its linear part; coordinates
    // outside the axes weigh nothing, so terms in them alone are kept.
    std::vector<P> trimmed(const Centre& c) const {
        const int n = static_cast<int>(c.images.size());
        std::vector<Q> weight(n, Q(0));
        for (size_t k = 0; k < c.axes.size(); ++k) weight[c.axes[k]] = 1 / c.a[k];
        std::vector<P> out;
        for (const auto& img : c.images) {
            Q top = 0;
            for (slong k = 0; k < img.length(); ++k) {
                const auto e = img.exponent(k);
                if (degree(e, n) != 1) continue;
                for (int i = 0; i < n; ++i)
                    if (e[i]) top = std::max(top, weight[i]);
            }
            std::vector<std::pair<std::vector<ulong>, Q>> keep;
            for (slong k = 0; k < img.length(); ++k) {
                auto e = img.exponent(k);
                Q w = 0;
                for (int i = 0; i < n; ++i) w += weight[i] * Q(long(e[i]));
                if (degree(e, n) <= 1 || w < top) keep.emplace_back(std::move(e), img.coefficient(k));
            }
            out.push_back(from_terms(img.ctx(), keep));
        }
        return out;
    }

    std::vector<int> centre_coordinates(const Centre& c, const std::vector<bool>& boundary) const {
        const int n = static_cast<int>(boundary.size());
        std::vector<int> toric = c.axes;
        for (int i = 0; i < n; ++i) {
            if (!boundary[i]) continue;
            for (const auto& g : c.monomial)
                if (g[i] != 0 && std::find(toric.begin(), toric.end(), i) == toric.end()) toric.push_back(i);
        }
        std::sort(toric.begin(), toric.end());
        return toric;
    }

    // Generators of J in the coordinates `toric`, made integral by `common`.
    std::vector<Vec> centre_points(const Centre& c, const std::vector<int>& toric, Zi& common) const {
        const size_t m = toric.size();
        std::vector<std::vector<Q>> gens;
        for (size_t k = 0; k < c.axes.size(); ++k) {
            std::vector<Q> v(m, Q(0));
            v[std::find(toric.begin(), toric.end(), c.axes[k]) - toric.begin()] = c.a[k];
            gens.push_back(v);
        }
        for (const auto& g : c.monomial) {
            std::vector<Q> v(m, Q(0));
            for (size_t j = 0; j < m; ++j) v[j] = g[toric[j]];
            gens.push_back(v);
        }
        common = 1;
        for (const auto& v : gens)
            for (const auto& q : v) common = lcm(common, Zi(q.get_den()));
        std::vector<Vec> points;
        for (const auto& v : gens) {
            Vec p(m);
            for (size_t j = 0; j < m; ++j) p[j] = Q(v[j] * Q(common)).get_num();
            points.push_back(p);
        }
        return points;
    }

    // The least degree of a monomial of F outside the Newton polyhedron of the
    // centre, or -1 when F is in the integral closure of J.
    long admissible(const P& F, const Centre& c, const std::vector<bool>& boundary) const {
        const int n = static_cast<int>(boundary.size());
        const auto toric = centre_coordinates(c, boundary);
        const size_t m = toric.size();
        if (m == 0) return 0;
        Zi common;
        const auto points = centre_points(c, toric, common);
        long outside = -1;
        for (const auto& w : facet_normals(points, m)) {
            Zi lowest = dot(w, points[0]);
            for (const auto& p : points) lowest = std::min(lowest, dot(w, p));
            for (slong k = 0; k < F.length(); ++k) {
                const auto e = F.exponent(k);
                Zi s = 0;
                for (size_t j = 0; j < m; ++j) s += w[j] * Zi(static_cast<unsigned long>(e[toric[j]]));
                if (s * common < lowest) {
                    const long d = static_cast<long>(degree(e, n));
                    if (outside < 0 || d < outside) outside = d;
                }
            }
        }
        return outside;
    }

    // The weighted blowup of the centre c, monomial in its coordinates, of the
    // task's chart: F (and the ambient equations GA) are already in the
    // centre's coordinates. The new charts own the points over V(region).
    void finish(const Task& task, const Centre& c, const P& F, const std::vector<P>& GA,
                const std::vector<bool>& boundary, const std::vector<P>& region,
                bool along_locus, std::vector<Task>& stack) {
        const Ctx& x = *task.ctx;
        const int n = x.n;
        const auto toric = centre_coordinates(c, boundary);
        const size_t m = toric.size();
        if (m == 0) throw Declined{"An empty centre at a point that needs one."};
        Zi common;
        const auto points = centre_points(c, toric, common);
        Fan fan = normal_fan(points, m);
        if (regular) regularize(fan);
        auto ray_order = [&](const P& g, const Vec& r) {
            Zi best = -1;
            for (slong k = 0; k < g.length(); ++k) {
                const auto e = g.exponent(k);
                Zi s = 0;
                for (size_t j = 0; j < m; ++j) s += r[j] * Zi(static_cast<unsigned long>(e[toric[j]]));
                if (best < 0 || s < best) best = s;
            }
            return best < 0 ? Zi(0) : best;
        };
        std::vector<P> maps;
        for (size_t i = 1; i < task.state.size(); ++i) maps.push_back(compose(task.state[i], c.images));
        const std::string label = "W" + std::to_string(++labels);
        Json::Value record;
        record["label"] = label;
        record["at"] = task.name;
        record["depth"] = task.depth;
        record["field_degree"] = x.d;
        record["locus"] = along_locus ? "positive dimension" : "point";
        record["variables"] = n;
        record["ambient_equations"] = Json::UInt(GA.size());
        record["centre_from"] = last_route;
        record["at_seconds"] = std::chrono::duration<double>(Clock::now() - started_at).count();
        Json::Value inv(Json::arrayValue);
        for (size_t k = static_cast<size_t>(c.ambient); k < c.a.size(); ++k) inv.append(c.a[k].get_str());
        record["invariant"] = inv;
        Json::Value mono(Json::arrayValue);
        for (const auto& g : c.monomial) {
            Json::Value v(Json::arrayValue);
            for (const auto& q : g) v.append(q.get_str());
            mono.append(v);
        }
        record["monomial_part"] = mono;
        record["rays"] = Json::UInt(fan.rays.size());
        record["charts"] = Json::UInt(fan.cones.size());
        blowups.append(record);
        // Faces owned by the first cone that contains them.
        std::map<std::vector<size_t>, size_t> owner;
        for (size_t k = 0; k < fan.cones.size(); ++k) {
            const auto& cone = fan.cones[k];
            for (unsigned long s = 0; s < (1UL << cone.size()); ++s) {
                std::vector<size_t> face;
                for (size_t i = 0; i < cone.size(); ++i)
                    if (s >> i & 1) face.push_back(cone[i]);
                owner.emplace(face, k);
            }
        }
        for (size_t k = 0; k < fan.cones.size(); ++k) {
            const auto& cone = fan.cones[k];
            // Slots: a standard ray keeps its coordinate, new rays take the freed ones.
            std::vector<int> slot(cone.size(), -1);
            std::vector<bool> taken(m, false);
            for (size_t i = 0; i < cone.size(); ++i) {
                const Vec& r = fan.rays[cone[i]];
                int nz = 0, at = -1;
                for (size_t j = 0; j < m; ++j)
                    if (r[j] != 0) ++nz, at = static_cast<int>(j);
                if (nz == 1 && r[at] == 1) slot[i] = at, taken[at] = true;
            }
            for (size_t i = 0; i < cone.size(); ++i)
                if (slot[i] < 0)
                    for (size_t j = 0; j < m; ++j)
                        if (!taken[j]) {
                            slot[i] = static_cast<int>(j);
                            taken[j] = true;
                            break;
                        }
            // x_{toric[j]} = prod_i t_{toric[slot[i]]}^{r_i[j]}.
            std::vector<P> images;
            for (int i = 0; i < n; ++i) images.push_back(gen(x, i));
            for (size_t j = 0; j < m; ++j) {
                std::vector<ulong> e(x.nv, 0);
                for (size_t i = 0; i < cone.size(); ++i) {
                    auto& exponent = e[toric[slot[i]]];
                    exponent = checked_order(Zi(exponent) + fan.rays[cone[i]][j]);
                }
                images[toric[j]] = from_terms(x, {{e, Q(1)}});
            }
            // Strict transforms: divide by the order along each new ray.
            std::vector<long> divisor = task.divisor;
            for (size_t j = 0; j < m; ++j) divisor[toric[j]] = -1;
            auto strict = [&](const P& before, bool record_divisors) {
                std::vector<ulong> divide(n, 0);
                for (size_t i = 0; i < cone.size(); ++i) {
                    const Vec& r = fan.rays[cone[i]];
                    const int s = toric[slot[i]];
                    int nz = 0;
                    for (const auto& v : r) nz += v != 0;
                    const bool standard = nz == 1 && r[slot[i]] == 1;
                    if (standard) {
                        if (record_divisors) divisor[s] = task.divisor[s];  // the old coordinate, boundary or not
                        continue;
                    }
                    const Zi o = ray_order(before, r);
                    divide[s] = checked_order(o);
                    if (!record_divisors) continue;
                    Divisor d;
                    d.label = label + "." + std::to_string(cone[i]);
                    d.N = o;
                    d.nu = 0;
                    for (size_t j = 0; j < m; ++j) {
                        const long old = task.divisor[toric[j]];
                        d.N += r[j] * (old >= 0 ? divisors[old].N : Zi(0));
                        d.nu += r[j] * (old >= 0 ? divisors[old].nu : Zi(1));
                    }
                    long index = -1;
                    for (size_t q = 0; q < divisors.size(); ++q)
                        if (divisors[q].label == d.label) index = static_cast<long>(q);
                    if (index < 0) {
                        divisors.push_back(d);
                        index = static_cast<long>(divisors.size()) - 1;
                    }
                    divisor[s] = index;
                }
                const P G = compose(before, images);
                std::vector<std::pair<std::vector<ulong>, Q>> terms;
                for (slong t = 0; t < G.length(); ++t) {
                    auto e = G.exponent(t);
                    for (int i = 0; i < n; ++i) {
                        if (e[i] < divide[i]) throw Declined{"A strict transform is not divisible."};
                        e[i] -= divide[i];
                    }
                    terms.emplace_back(std::move(e), G.coefficient(t));
                }
                return from_terms(x, terms);
            };
            std::vector<P> next;
            next.push_back(strict(F, true));
            for (const auto& g : maps) next.push_back(compose(g, images));
            std::vector<P> ambient;
            for (const auto& A : GA) ambient.push_back(strict(A, false));
            // Points over the blown-up locus.
            std::vector<P> fibre;
            for (const auto& g : region) fibre.push_back(compose(compose(g, c.images), images));
            Chart chart;
            chart.ctx = task.ctx;
            chart.fibre = fibre;
            chart.name = label + ".c" + std::to_string(k);
            chart.state = next;
            chart.ambient = ambient;
            chart.divisor = divisor;
            chart.depth = task.depth;
            {
                std::vector<Vec> rays;
                for (size_t i : cone) rays.push_back(fan.rays[i]);
                chart.orbifold_order = abs(determinant(rays));
            }
            for (const auto& av : task.avoid) {
                std::vector<P> h;
                for (const auto& g : av) h.push_back(compose(compose(g, c.images), images));
                chart.avoid.push_back(std::move(h));
            }
            for (size_t j = 0; j < m; ++j) chart.toric.push_back(toric[j]);
            for (unsigned long s = 0; s < (1UL << cone.size()); ++s) {
                std::vector<size_t> face;
                std::vector<int> coords;
                for (size_t i = 0; i < cone.size(); ++i)
                    if (s >> i & 1) face.push_back(cone[i]), coords.push_back(toric[slot[i]]);
                if (owner.at(face) != k) continue;
                std::sort(coords.begin(), coords.end());
                chart.owned.push_back(coords);
            }
            examine(std::move(chart), stack);
        }
    }
};

Json::Value orbit_json(const Orbit& o);

// The kernel's normal-crossings test ("crossings") on every face a chart owns:
// the loci over the blown-up locus where some set of components among the
// strict transform and the exceptional divisors fails to meet smoothly in the
// expected codimension must be exactly those blown up further or resolved by
// other charts. The chart lies on its ambient equations, and over a number
// field on m(theta) = 0.
Json::Value certify(const Run& run) {
    Json::Value failures(Json::arrayValue);
    long faces = 0;
    for (const auto& chart : run.charts) {
        const Ctx& x = *chart.ctx;
        const int n = x.n;
        std::vector<int> boundary;
        for (int i = 0; i < n; ++i)
            if (chart.divisor[i] >= 0) boundary.push_back(i);
        for (const auto& face : chart.owned) {
            check_deadline();
            ++faces;
            std::vector<std::string> ambient, rows, fibre;
            if (x.d > 1) {
                P m(x);
                fmpq_mpoly_set(m.p, x.modulus, x.c);
                ambient.push_back(row(m));
            }
            for (const auto& a : chart.ambient) ambient.push_back(row(a));
            rows.push_back(row(chart.state[0]));
            for (int b : boundary) rows.push_back(row(gen(x, b)));
            for (const auto& g : chart.fibre)
                if (!g.zero()) fibre.push_back(row(g));
            for (int i : face) fibre.push_back(row(gen(x, i)));
            std::string request = "crossings " + std::to_string(x.nv) + " " +
                                  std::to_string(n - static_cast<int>(chart.ambient.size())) + " " +
                                  std::to_string(ambient.size()) + " " + std::to_string(rows.size());
            for (size_t i = 0; i < rows.size(); ++i) request += " 1";
            request += " " + std::to_string(fibre.size()) + " 0\n";
            for (const auto& r : ambient) request += r + "\n";
            for (const auto& r : rows) request += r + "\n";
            for (const auto& r : fibre) request += r + "\n";
            char* raw = atlas_ideal(request.data(), request.size());
            if (!raw) {
                Json::Value v;
                v["chart"] = chart.name;
                v["problem"] = "the kernel could not check the chart";
                failures.append(v);
                continue;
            }
            std::string answer(raw);
            atlas_groebner_release(raw);
            std::istringstream lines(answer);
            std::string line;
            std::vector<int> all(n);
            std::iota(all.begin(), all.end(), 0);
            auto toric_outside = [&](int t) { return std::find(face.begin(), face.end(), t) == face.end(); };
            while (std::getline(lines, line)) {
                if (line.empty() || line == "none" || line == "unknown") continue;
                std::istringstream head(line);
                std::string word;
                long size, inside, count;
                head >> word >> size;
                std::vector<long> indices(size);
                for (auto& i : indices) head >> i;
                head >> inside >> count;
                std::vector<P> gens;
                for (long c = 0; c < count && std::getline(lines, line); ++c) gens.push_back(from_row(x, line));
                Json::Value idx(Json::arrayValue);
                for (long i : indices) idx.append(Json::Int64(i));
                Locus found;
                try {
                    found = locus(gens, all, x);
                } catch (const Declined& d) {
                    Json::Value v;
                    v["chart"] = chart.name;
                    v["problem"] = "a failure locus could not be decomposed: " + d.reason;
                    v["components"] = idx;
                    failures.append(v);
                    continue;
                }
                for (const auto& o : found.points) {
                    bool owned = true;
                    for (int t : chart.toric)
                        if (toric_outside(t) && !nonzero_in_orbit(o, t)) owned = false;
                    if (!owned) continue;
                    bool excluded = false;
                    for (const auto& e : chart.excluded)
                        if (on_orbit(e.ideal, o)) excluded = true;
                    for (const auto& c : chart.curves)
                        if (on_orbit(c, o)) excluded = true;
                    for (const auto& av : chart.avoid)
                        if (on_orbit(av, o)) excluded = true;
                    if (excluded) continue;
                    Json::Value v;
                    v["chart"] = chart.name;
                    v["problem"] = "an orbit that is not normal crossings and was not blown up";
                    v["field_degree"] = o.field->d;
                    v["orbit"] = orbit_json(o);
                    v["components"] = idx;
                    failures.append(v);
                }
                for (const auto& c : found.curves) {
                    bool owned = true;
                    for (int t : chart.toric)
                        if (toric_outside(t) && contains(c, {gen(x, t)})) owned = false;
                    if (!owned) continue;
                    bool excluded = false;
                    for (const auto& e : chart.curves)
                        if (contains(c, e)) excluded = true;
                    for (const auto& av : chart.avoid)
                        if (contains(c, av)) excluded = true;
                    if (excluded) continue;
                    Json::Value v;
                    v["chart"] = chart.name;
                    v["problem"] = "a locus of positive dimension that is not normal crossings and was not blown up";
                    Json::Value ideal(Json::arrayValue);
                    for (const auto& g : c) ideal.append(row(g));
                    v["ideal"] = ideal;
                    v["components"] = idx;
                    failures.append(v);
                }
            }
        }
    }
    Json::Value out;
    out["charts"] = Json::UInt(run.charts.size());
    out["faces"] = Json::Int64(faces);
    out["failures"] = failures;
    out["certified"] = failures.empty();
    return out;
}

Json::Value orbit_json(const Orbit& o) {
    Json::Value out;
    out["field"] = minimal_json(*o.field);
    Json::Value pt(Json::arrayValue);
    for (const auto& c : o.point) pt.append(row(c));
    out["point"] = pt;
    return out;
}

std::vector<std::pair<std::vector<ulong>, Q>> read_terms(const std::string& text, int n) {
    std::vector<std::pair<std::vector<ulong>, Q>> terms;
    std::istringstream in(text);
    std::string term;
    while (std::getline(in, term, ';')) {
        std::istringstream t(term);
        std::string c;
        if (!(t >> c)) continue;
        std::vector<ulong> e(n);
        for (auto& v : e) {
            Zi value;
            if (!(t >> value) || value < 0) throw std::invalid_argument("Bad polynomial exponent.");
            v = checked_order(value);
        }
        std::string extra;
        if (t >> extra) throw std::invalid_argument("Bad polynomial row width.");
        Q q(c);
        q.canonicalize();
        terms.emplace_back(e, q);
    }
    return terms;
}

Json::Value resolve(const Json::Value& request) {
    const int n = request["variables"].asInt();
    if (n < 1 || n > 16) throw std::invalid_argument("Expected between 1 and 16 variables.");
    const bool logarithmic = request.get("logarithmic", true).asBool();
    const bool regular = request.get("regular", false).asBool();
    const auto seconds = request.get("seconds", 40.0);
    const auto deadline = [&seconds]() {
        return seconds.isNull() ? Clock::time_point::max()
                                : Clock::now() + std::chrono::milliseconds(static_cast<long>(seconds.asDouble() * 1000));
    };
    Run run{};
    run.n = n;
    run.logarithmic = logarithmic;
    run.regular = regular;
    run.deadline = deadline();
    context_registry = &run.contexts;
    CtxPtr base = make_field(n, nullptr);
    Json::Value out;
    const auto start = Clock::now();
    started_at = start;
    deadline_at = run.deadline;
    try {
        const auto terms = read_terms(request["polynomial"].asString(), n);
        Task first;
        first.ctx = base;
        first.state.push_back(from_terms(*base, terms));
        for (int i = 0; i < n; ++i) first.state.push_back(gen(*base, i));
        first.divisor.assign(n, -1);
        first.name = "origin";
        if (first.state[0].zero()) throw Declined{"The zero polynomial."};
        std::vector<Task> stack;
        if (!field_coefficient(first.state[0], std::vector<ulong>(n, 0)).zero())
            throw Declined{"The origin is not on the hypersurface."};
        if (order(first.state[0]) >= 2) stack.push_back(std::move(first));
        while (!stack.empty()) {
            Task t = std::move(stack.back());
            stack.pop_back();
            run.process(std::move(t), stack);
        }
        out["status"] = "resolved";
    } catch (const Declined& d) {
        out["status"] = "declined";
        out["reason"] = d.reason;
    } catch (const Limit& e) {
        out["status"] = "declined";
        out["reason"] = e.what();
    }
    out["seconds"] = std::chrono::duration<double>(Clock::now() - start).count();
    out["algebraic_orbits"] = Json::Int64(run.extensions);
    out["blowups"] = run.blowups;
    if (request.get("certify", false).asBool() && out["status"].asString() == "resolved") {
        const auto t0 = Clock::now();
        deadline_at = deadline();
        try {
            out["certificate"] = certify(run);
        } catch (const Declined& d) {
            out["certificate"]["certified"] = false;
            out["certificate"]["problem"] = d.reason;
        } catch (const Limit& e) {
            out["certificate"]["certified"] = false;
            out["certificate"]["problem"] = e.what();
        }
        out["certificate"]["seconds"] = std::chrono::duration<double>(Clock::now() - t0).count();
    }
    Json::Value divisors(Json::arrayValue);
    for (const auto& d : run.divisors) {
        Json::Value v;
        v["label"] = d.label;
        v["N"] = d.N.get_str();
        v["nu"] = d.nu.get_str();
        divisors.append(v);
    }
    out["divisors"] = divisors;
    Json::Value charts(Json::arrayValue);
    for (const auto& c : run.charts) {
        Json::Value v;
        v["name"] = c.name;
        if (c.ctx->d > 1) v["field"] = minimal_json(*c.ctx);
        v["strict"] = row(c.state[0]);
        Json::Value map(Json::arrayValue), fibre(Json::arrayValue);
        for (int i = 1; i <= n; ++i) map.append(row(c.state[i]));
        for (const auto& g : c.fibre) fibre.append(row(g));
        v["map"] = map;
        v["fibre"] = fibre;
        Json::Value ambient(Json::arrayValue);
        for (const auto& a : c.ambient) ambient.append(row(a));
        v["ambient"] = ambient;
        v["variables"] = c.ctx->n;
        v["orbifold_order"] = c.orbifold_order.get_str();
        Json::Value curves(Json::arrayValue);
        for (const auto& ideal : c.curves) {
            Json::Value w(Json::arrayValue);
            for (const auto& g : ideal) w.append(row(g));
            curves.append(w);
        }
        v["blown_up_loci"] = curves;
        Json::Value boundary(Json::arrayValue);
        for (size_t i = 0; i < c.divisor.size(); ++i)
            if (c.divisor[i] >= 0) {
                Json::Value b;
                b["coordinate"] = Json::UInt(i);
                b["divisor"] = run.divisors[c.divisor[i]].label;
                boundary.append(b);
            }
        v["boundary"] = boundary;
        Json::Value toric(Json::arrayValue);
        for (int i : c.toric) toric.append(i);
        v["toric"] = toric;
        Json::Value owned(Json::arrayValue);
        for (const auto& f : c.owned) {
            Json::Value w(Json::arrayValue);
            for (int i : f) w.append(i);
            owned.append(w);
        }
        v["owned_faces"] = owned;
        Json::Value excluded(Json::arrayValue);
        for (const auto& o : c.excluded) excluded.append(orbit_json(o));
        v["excluded"] = excluded;
        charts.append(v);
    }
    out["charts"] = charts;
    return out;
}

// The centre at the origin of a polynomial with the given boundary coordinates.
Json::Value centre_at_origin(const Json::Value& request) {
    const int n = request["variables"].asInt();
    if (n < 1 || n > 16) throw std::invalid_argument("Expected between 1 and 16 variables.");
    Ctx x(n, nullptr);
    P f = from_terms(x, read_terms(request["polynomial"].asString(), n));
    std::vector<bool> boundary(n, false);
    for (const auto& b : request["boundary"]) boundary[b.asInt()] = true;
    const long cap = request.get("cap", 16).asInt64();
    Centre c = centre(f, {}, boundary, request.get("logarithmic", true).asBool(), cap, cap);
    Json::Value out;
    Json::Value a(Json::arrayValue), axes(Json::arrayValue), mono(Json::arrayValue), images(Json::arrayValue);
    for (const auto& q : c.a) a.append(q.get_str());
    for (int i : c.axes) axes.append(i);
    for (const auto& g : c.monomial) {
        Json::Value v(Json::arrayValue);
        for (const auto& q : g) v.append(q.get_str());
        mono.append(v);
    }
    for (const auto& img : c.images) images.append(row(img));
    out["invariant"] = a;
    out["axes"] = axes;
    out["monomial_part"] = mono;
    out["images"] = images;
    return out;
}

Json::Value execute(const Json::Value& request) {
    const std::string op = request["operation"].asString();
    if (op == "resolve") {
        struct RestoreContexts {
            std::vector<CtxPtr>* saved;
            ~RestoreContexts() { context_registry = saved; }
        } restore{context_registry};
        try {
            return resolve(request);
        } catch (const Declined& decline) {
            // Converting chart exponents to the weighted record may exceed
            // the word range even after the blowup calculations succeeded.
            Json::Value result;
            result["status"] = "declined";
            result["reason"] = decline.reason;
            return result;
        }
    }
    if (op == "centre") return centre_at_origin(request);
    throw std::invalid_argument("Unknown weighted resolution operation.");
}

}  // namespace
}  // namespace atlas::weighted

extern "C" const char* atlas_weighted_resolution(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, atlas::weighted::execute);
}

// The weighted resolution as a helper program: the request on standard input,
// the answer ({"result": ...} or {"error": ...}) on standard output. Atlas runs
// it in a forked process, which a deadline or a memory bound can stop.
extern "C" int atlas_weighted_main(int, char**) noexcept {
    int status = 1;
    try {
        const std::string input(std::istreambuf_iterator<char>(std::cin), {});
        std::cout << atlas_weighted_resolution(input.c_str());
        std::cout.flush();
        status = std::cout ? 0 : 1;
    } catch (...) {
        status = 1;
    }
    std::fflush(nullptr);
    return status;
}
