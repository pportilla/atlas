// Exact Euler characteristic with compact supports of a complex affine
// constructible set: the algorithm of engine/constructible_euler.py on FLINT
// polynomials.
//
// A closed set V(E) in the affine space of the variables that E involves is
// projected along one of them, v. Over a point of the base the fibre is the
// set of common roots of the equations as polynomials in v: the whole line
// (chi = 1), or the roots of their greatest common divisor g (chi = the number
// of distinct roots of g). The base is cut into pieces, each given by
// polynomials that vanish and polynomials that do not, on which the degrees of
// the polynomials in the Euclidean algorithm are constant; on such a piece a
// pseudo-remainder is a remainder up to a function without zeros, so g and
// gcd(g, g') have constant degrees there and chi of the part of V(E) over the
// piece is that number times chi of the piece. Additivity of chi gives the
// total; the pieces live in an affine space with one variable less.
//
// A condition is kept as the primitive squarefree polynomial with the same
// zeros, so that equal conditions are equal polynomials.
#include "groebner.hpp"

#include <flint/fmpz_mpoly.h>

#include <map>
#include <memory>
#include <utility>

namespace atlas {
namespace {

struct MP {
    fmpz_mpoly_t p;
    const fmpz_mpoly_ctx_struct* ctx;
    explicit MP(const fmpz_mpoly_ctx_struct* c) : ctx(c) { fmpz_mpoly_init(p, ctx); }
    MP(const MP& other) : ctx(other.ctx) {
        fmpz_mpoly_init(p, ctx);
        fmpz_mpoly_set(p, other.p, ctx);
    }
    MP(MP&& other) noexcept : ctx(other.ctx) {
        fmpz_mpoly_init(p, ctx);
        fmpz_mpoly_swap(p, other.p, ctx);
    }
    MP& operator=(MP other) noexcept {
        fmpz_mpoly_swap(p, other.p, ctx);
        return *this;
    }
    ~MP() { fmpz_mpoly_clear(p, ctx); }
    bool zero() const { return fmpz_mpoly_is_zero(p, ctx); }
    bool constant() const { return fmpz_mpoly_is_fmpz(p, ctx); }
};

// Sorted numbers of conditions.
using Set = std::vector<int>;

bool has(const Set& s, int id) { return std::binary_search(s.begin(), s.end(), id); }
Set with(const Set& s, int id) {
    Set out(s);
    const auto at = std::lower_bound(out.begin(), out.end(), id);
    if (at == out.end() || *at != id) out.insert(at, id);
    return out;
}
Set united(const Set& a, const Set& b) {
    Set out;
    std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return out;
}
bool disjoint(const Set& a, const Set& b) {
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) return false;
        if (a[i] < b[j]) ++i;
        else ++j;
    }
    return true;
}

// A polynomial in one variable over the polynomials in the others: the
// coefficient of v^k is (*c)[k] for k < len, and the last one is not zero.
struct Uni {
    std::shared_ptr<const std::vector<MP>> c;
    int len = 0;
    int degree() const { return len - 1; }
    const MP& lead() const { return (*c)[len - 1]; }
    const MP& at(int k) const { return (*c)[k]; }
};

// A piece of the base, and what a polynomial is on it; `some` false for zero.
struct Case {
    Uni g;
    bool some;
    Set z, u;
};

}  // namespace

struct Euler::State {
    fmpz_mpoly_ctx_t ctx;
    Ring R;
    int n;
    std::vector<MP> conditions;          // [0] is the constant 1
    std::vector<u64> support;            // the variables of each condition
    std::map<std::string, int> numbers;  // a condition -> its number
    std::map<std::string, int> seen;     // a polynomial -> the number of its condition
    std::map<Set, int> products;
    std::map<std::pair<u64, Set>, long> values;

    explicit State(const Ring& ring) : R(ring), n(ring.n) {
        fmpz_mpoly_ctx_init(ctx, n, ORD_LEX);
        MP one(ctx);
        fmpz_mpoly_one(one.p, ctx);
        conditions.push_back(std::move(one));
        support.push_back(0);
    }
    ~State() {
        conditions.clear();
        fmpz_mpoly_ctx_clear(ctx);
    }

    MP make() const { return MP(ctx); }
    std::string key(const MP& f) const {
        char* text = fmpz_mpoly_get_str_pretty(f.p, nullptr, ctx);
        std::string out(text);
        flint_free(text);
        return out;
    }
    u64 variables(const MP& f) const {
        slong degrees[MAX_VARIABLES];
        fmpz_mpoly_degrees_si(degrees, f.p, ctx);
        u64 mask = 0;
        for (int v = 0; v < n; ++v)
            if (degrees[v] > 0) mask |= 1ULL << v;
        return mask;
    }
    MP gcd(const MP& a, const MP& b) const {
        MP g(ctx);
        if (!fmpz_mpoly_gcd(g.p, a.p, b.p, ctx)) throw Limit("A greatest common divisor was not found.");
        return g;
    }
    MP quotient(const MP& a, const MP& b) const {
        MP q(ctx);
        if (!fmpz_mpoly_divides(q.p, a.p, b.p, ctx)) throw Limit("A division was not exact.");
        return q;
    }
    MP product(const MP& a, const MP& b) const {
        MP out(ctx);
        fmpz_mpoly_mul(out.p, a.p, b.p, ctx);
        return out;
    }
    // Primitive, with positive leading coefficient.
    void normalize(MP& f) const {
        if (f.zero()) return;
        fmpz_mpoly_primitive_part(f.p, f.p, ctx);
        Z lead;
        fmpz_mpoly_get_term_coeff_fmpz(&lead.v, f.p, 0, ctx);
        if (fmpz_sgn(&lead.v) < 0) fmpz_mpoly_neg(f.p, f.p, ctx);
    }

    // The number of the condition "f = 0": -1 when f is zero (always true),
    // 0 when f is a nonzero constant (never true).
    int condition(const MP& f) {
        if (f.zero()) return -1;
        if (f.constant()) return 0;
        const std::string raw = key(f);
        const auto known = seen.find(raw);
        if (known != seen.end()) return known->second;
        // The squarefree part: f divided by the gcd of f and its derivatives.
        MP g(f), d(ctx);
        for (int v = 0; v < n && !g.constant(); ++v) {
            if (fmpz_mpoly_degree_si(f.p, v, ctx) <= 0) continue;
            fmpz_mpoly_derivative(d.p, f.p, v, ctx);
            g = gcd(g, d);
        }
        MP s = g.constant() ? f : quotient(f, g);
        normalize(s);
        const std::string canonical = key(s);
        const auto found = numbers.find(canonical);
        int id;
        if (found != numbers.end()) {
            id = found->second;
        } else {
            id = static_cast<int>(conditions.size());
            support.push_back(variables(s));
            conditions.push_back(std::move(s));
            numbers.emplace(canonical, id);
        }
        seen.emplace(raw, id);
        return id;
    }
    // The condition that some polynomial of u vanishes.
    int forbidden(const Set& u) {
        const auto known = products.find(u);
        if (known != products.end()) return known->second;
        MP all(conditions[u[0]]);
        for (size_t i = 1; i < u.size(); ++i) all = product(all, conditions[u[i]]);
        const int id = condition(all);
        products.emplace(u, id);
        return id;
    }

    MP converted(const Poly& f) const {
        MP out(ctx);
        long e[MAX_VARIABLES];
        ulong exponents[MAX_VARIABLES];
        for (size_t i = 0; i < f.size(); ++i) {
            R.unpack(f.m.data() + i * R.stride, e);
            for (int v = 0; v < n; ++v) exponents[v] = static_cast<ulong>(e[v]);
            fmpz_mpoly_push_term_fmpz_ui(out.p, &f.c[i].v, exponents, ctx);
        }
        fmpz_mpoly_sort_terms(out.p, ctx);
        fmpz_mpoly_combine_like_terms(out.p, ctx);
        return out;
    }
    Set set_of(const std::vector<Poly>& ideal) {
        Set out;
        for (const Poly& f : ideal) {
            const int id = condition(converted(f));
            if (id >= 0) out = with(out, id);
        }
        return out;
    }

    // f as a polynomial in the variable v.
    Uni split(const MP& f, int v) const {
        const slong degree = fmpz_mpoly_degree_si(f.p, v, ctx);
        auto coefficients = std::make_shared<std::vector<MP>>();
        Uni out;
        if (degree < 0) return out;
        coefficients->assign(static_cast<size_t>(degree) + 1, MP(ctx));
        fmpz_mpoly_univar_t u;
        fmpz_mpoly_univar_init(u, ctx);
        fmpz_mpoly_to_univar(u, f.p, v, ctx);
        const slong length = fmpz_mpoly_univar_length(u, ctx);
        for (slong i = 0; i < length; ++i) {
            const slong e = fmpz_mpoly_univar_get_term_exp_si(u, i, ctx);
            fmpz_mpoly_univar_get_term_coeff((*coefficients)[static_cast<size_t>(e)].p, u, i, ctx);
        }
        fmpz_mpoly_univar_clear(u, ctx);
        out.len = static_cast<int>(degree) + 1;
        out.c = std::move(coefficients);
        return out;
    }
    // c^d * q(v = -p0/c), for d the degree of q in v and c a nonzero integer.
    MP substituted(const MP& q, int v, const MP& p0, const fmpz_t c) const {
        const Uni parts = split(q, v);
        if (parts.len <= 1) return q;
        MP minus(ctx), result(parts.lead()), term(ctx);
        fmpz_mpoly_neg(minus.p, p0.p, ctx);
        Z power;
        fmpz_one(&power.v);
        for (int k = parts.degree() - 1; k >= 0; --k) {
            fmpz_mul(&power.v, &power.v, c);
            fmpz_mpoly_mul(result.p, result.p, minus.p, ctx);
            fmpz_mpoly_scalar_mul_fmpz(term.p, parts.at(k).p, &power.v, ctx);
            fmpz_mpoly_add(result.p, result.p, term.p, ctx);
        }
        return result;
    }

    static void trim(std::vector<MP>& r) {
        while (!r.empty() && r.back().zero()) r.pop_back();
    }
    static Uni wrap(std::vector<MP>&& r) {
        Uni out;
        out.len = static_cast<int>(r.size());
        out.c = std::make_shared<const std::vector<MP>>(std::move(r));
        return out;
    }
    Uni derivative(const Uni& g) const {
        std::vector<MP> d;
        for (int k = 1; k < g.len; ++k) {
            MP term(ctx);
            fmpz_mpoly_scalar_mul_si(term.p, g.at(k).p, k, ctx);
            d.push_back(std::move(term));
        }
        return wrap(std::move(d));
    }
    // A pseudo-remainder of a by b: lc(b)^k * a = q*b + r with deg r < deg b,
    // then divided by integers and by factors of lc(b) common to its
    // coefficients. Where lc(b) is not zero it is the remainder up to a
    // function without zeros.
    Uni remainder(const Uni& a, const Uni& b) const {
        std::vector<MP> r(a.c->begin(), a.c->begin() + a.len);
        const int db = b.degree();
        const MP& lb = b.lead();
        MP term(ctx);
        while (static_cast<int>(r.size()) - 1 >= db) {
            const int dr = static_cast<int>(r.size()) - 1;
            const MP lr = std::move(r.back());
            r.pop_back();
            if (!lb.constant() || !fmpz_mpoly_equal_si(lb.p, 1, ctx))
                for (MP& coefficient : r) fmpz_mpoly_mul(coefficient.p, coefficient.p, lb.p, ctx);
            for (int k = 0; k < db; ++k) {
                if (b.at(k).zero()) continue;
                fmpz_mpoly_mul(term.p, lr.p, b.at(k).p, ctx);
                MP& target = r[static_cast<size_t>(k + dr - db)];
                fmpz_mpoly_sub(target.p, target.p, term.p, ctx);
            }
            trim(r);
        }
        if (r.empty()) return wrap(std::move(r));
        MP content(r[0]);
        for (size_t k = 1; k < r.size() && !(content.constant() && fmpz_mpoly_equal_si(content.p, 1, ctx)); ++k)
            if (!r[k].zero()) content = gcd(content, r[k]);
        Z integer;
        for (slong i = 0; i < fmpz_mpoly_length(content.p, ctx); ++i) {
            Z coefficient;
            fmpz_mpoly_get_term_coeff_fmpz(&coefficient.v, content.p, i, ctx);
            fmpz_gcd(&integer.v, &integer.v, &coefficient.v);
        }
        if (!fmpz_is_one(&integer.v))
            for (MP& coefficient : r) fmpz_mpoly_scalar_divexact_fmpz(coefficient.p, coefficient.p, &integer.v, ctx);
        normalize(content);
        while (!content.constant() && !lb.constant()) {
            MP common = gcd(content, lb);
            normalize(common);
            if (common.constant()) break;
            for (MP& coefficient : r) coefficient = quotient(coefficient, common);
            content = quotient(content, common);
        }
        return wrap(std::move(r));
    }

    // The pieces of the piece (z, u) on which p has a constant degree.
    std::vector<Case> degrees(Uni p, Set z, Set u) {
        std::vector<Case> out;
        while (p.len > 0) {
            const int c = condition(p.lead());
            if (!has(z, c)) {
                if (c == 0 || has(u, c)) {
                    out.push_back({p, true, z, u});
                    return out;
                }
                out.push_back({p, true, z, with(u, c)});
                z = with(z, c);
            }
            // On what is left the leading coefficient vanishes.
            --p.len;
            while (p.len > 0 && p.lead().zero()) --p.len;
        }
        out.push_back({p, false, z, u});
        return out;
    }
    // The pieces of (z, u) with the greatest common divisor of the
    // polynomials on each, up to a function without zeros.
    std::vector<Case> gcd_cases(std::vector<Uni> polynomials, Set z, Set u) {
        struct Todo {
            Uni g;
            bool some;
            std::vector<Uni> pending;
            Set z, u;
        };
        std::vector<Case> out;
        std::vector<Todo> todo;
        todo.push_back({Uni(), false, std::move(polynomials), std::move(z), std::move(u)});
        while (!todo.empty()) {
            Todo t = std::move(todo.back());
            todo.pop_back();
            if (t.pending.empty()) {
                out.push_back({t.g, t.some, t.z, t.u});
                continue;
            }
            const Uni p = t.pending[0];
            const std::vector<Uni> rest(t.pending.begin() + 1, t.pending.end());
            for (Case& h : degrees(p, t.z, t.u)) {
                if (!h.some) {
                    todo.push_back({t.g, t.some, rest, h.z, h.u});
                } else if (h.g.degree() == 0) {
                    out.push_back(h);
                } else if (!t.some) {
                    todo.push_back({h.g, true, rest, h.z, h.u});
                } else {
                    const bool first = t.g.degree() >= h.g.degree();
                    const Uni& a = first ? t.g : h.g;
                    const Uni& b = first ? h.g : t.g;
                    std::vector<Uni> pending;
                    pending.push_back(remainder(a, b));
                    pending.insert(pending.end(), rest.begin(), rest.end());
                    todo.push_back({b, true, std::move(pending), h.z, h.u});
                }
            }
        }
        return out;
    }

    // chi of the points of the affine space of `variables` where the
    // polynomials of z vanish and those of u do not.
    long open_part(u64 variables, const Set& z, const Set& u) {
        if (has(z, 0) || !disjoint(z, u)) return 0;
        const long value = closed(variables, z);
        if (u.empty()) return value;
        return value - closed(variables, with(z, forbidden(u)));
    }
    // chi of V(equations) in the affine space of the variables they involve:
    // a variable that they do not involve contributes the factor chi(C) = 1.
    long closed(u64, const Set& equations) {
        if (has(equations, 0)) return 0;
        if (equations.empty()) return 1;
        u64 variables = 0;
        for (int id : equations) variables |= support[static_cast<size_t>(id)];
        const auto key = std::make_pair(variables, equations);
        const auto known = values.find(key);
        if (known != values.end()) return known->second;
        const long value = projected(variables, equations);
        values.emplace(key, value);
        return value;
    }
    long projected(u64 variables, const Set& equations) {
        std::vector<int> names;
        for (int v = 0; v < n; ++v)
            if (variables >> v & 1) names.push_back(v);
        if (names.size() == 1) {
            MP g(conditions[static_cast<size_t>(equations[0])]);
            for (size_t i = 1; i < equations.size(); ++i) g = gcd(g, conditions[static_cast<size_t>(equations[i])]);
            return static_cast<long>(fmpz_mpoly_degree_si(g.p, names[0], ctx));
        }
        // An equation c*v + p0 with c a nonzero constant removes v: first those
        // that give v a value, then any.
        for (int pass = 0; pass < 2; ++pass) {
            for (int id : equations) {
                const MP& p = conditions[static_cast<size_t>(id)];
                const u64 here = support[static_cast<size_t>(id)];
                if (pass == 0 && (fmpz_mpoly_total_degree_si(p.p, ctx) != 1 || (here & (here - 1)))) continue;
                for (int v : names) {
                    if (fmpz_mpoly_degree_si(p.p, v, ctx) != 1) continue;
                    const Uni parts = split(p, v);
                    if (!parts.at(1).constant()) continue;
                    Z c;
                    fmpz_mpoly_get_fmpz(&c.v, parts.at(1).p, ctx);
                    Set remaining;
                    for (int other : equations) {
                        if (other == id) continue;
                        const int next = condition(substituted(conditions[static_cast<size_t>(other)], v, parts.at(0), &c.v));
                        if (next >= 0) remaining = with(remaining, next);
                    }
                    return closed(variables & ~(1ULL << v), remaining);
                }
            }
        }
        // The projection along the variable of least degree.
        int variable = names[0];
        std::pair<slong, slong> least(-1, -1);
        for (int v : names) {
            std::pair<slong, slong> cost(0, 0);
            for (int id : equations) {
                const slong d = fmpz_mpoly_degree_si(conditions[static_cast<size_t>(id)].p, v, ctx);
                cost.first = std::max(cost.first, d);
                cost.second += d;
            }
            if (least.first < 0 || cost < least) {
                least = cost;
                variable = v;
            }
        }
        const u64 base = variables & ~(1ULL << variable);
        std::vector<Uni> fibre;
        Set below;
        for (int id : equations) {
            if (support[static_cast<size_t>(id)] >> variable & 1) fibre.push_back(split(conditions[static_cast<size_t>(id)], variable));
            else below = with(below, id);
        }
        long value = 0;
        for (const Case& piece : gcd_cases(fibre, below, Set())) {
            if (!piece.some || piece.g.degree() == 1) {
                value += open_part(base, piece.z, piece.u);
            } else if (piece.g.degree() > 1) {
                std::vector<Uni> pair = {piece.g, derivative(piece.g)};
                for (const Case& repeated : gcd_cases(pair, piece.z, piece.u)) {
                    if (!repeated.some) throw Limit("A polynomial of known degree became zero.");
                    value += static_cast<long>(piece.g.degree() - repeated.g.degree()) * open_part(base, repeated.z, repeated.u);
                }
            }
        }
        return value;
    }
};

Euler::Euler(const Ring& R) : state_(new State(R)) {}
Euler::~Euler() { delete state_; }

long Euler::characteristic(const std::vector<Poly>& equations, const std::vector<std::vector<Poly>>& excluded) {
    State& s = *state_;
    const u64 all = s.n >= 64 ? ~0ULL : (1ULL << s.n) - 1;
    std::map<Set, long> pieces;
    pieces[s.set_of(equations)] = 1;
    Set inequations;
    for (const std::vector<Poly>& ideal : excluded) {
        const Set forbidden = s.set_of(ideal);
        if (forbidden.empty()) return 0;
        if (has(forbidden, 0)) continue;
        if (forbidden.size() == 1) {
            inequations = with(inequations, forbidden[0]);
            continue;
        }
        // The indicator of X minus V(I) is that of X minus that of X and V(I).
        std::map<Set, long> next(pieces);
        for (const auto& piece : pieces) next[united(piece.first, forbidden)] -= piece.second;
        pieces.clear();
        for (const auto& piece : next)
            if (piece.second) pieces.insert(piece);
    }
    long total = 0;
    for (const auto& piece : pieces) total += piece.second * s.open_part(all, piece.first, inequations);
    return total;
}

}  // namespace atlas
