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
#include <flint/fmpz_mpoly_factor.h>
#include <flint/fmpz_mat.h>
#include <flint/fmpq_mpoly.h>
#include <sstream>

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

// Only store the nonzero coefficients. Specializing a leading coefficient
// to zero removes one term from the shared prefix.
using FibreTerm = std::pair<Z, MP>;
using FibreTerms = std::vector<FibreTerm>;
struct Uni {
    std::shared_ptr<const FibreTerms> terms;
    size_t len = 0;
    const FibreTerm& at(size_t i) const { return (*terms)[i]; }
    const Z& degree() const { return at(len - 1).first; }
    const MP& lead() const { return at(len - 1).second; }
};

struct PowerLess {
    bool operator()(const Z& a, const Z& b) const { return fmpz_cmp(&a.v, &b.v) < 0; }
};

Uni fibre_polynomial(FibreTerms terms) {
    const size_t count = terms.size();
    return {std::make_shared<const FibreTerms>(std::move(terms)), count};
}

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
    std::vector<Set> support;            // the variables of each condition
    std::map<std::string, int> numbers;  // a condition -> its number
    std::map<std::string, int> seen;     // a polynomial -> the number of its condition
    std::map<Set, int> products;
    std::map<Set, Z> values;

    explicit State(const Ring& ring) : State(ring.n) { R = ring; }
    explicit State(int dimensions) : n(dimensions) {
        fmpz_mpoly_ctx_init(ctx, n, ORD_LEX);
        MP one(ctx);
        fmpz_mpoly_one(one.p, ctx);
        conditions.push_back(std::move(one));
        support.emplace_back();
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
    Set variables(const MP& f) const {
        Set names;
        Z degree;
        for (int v = 0; v < n; ++v) {
            fmpz_mpoly_degree_fmpz(&degree.v, f.p, v, ctx);
            if (fmpz_sgn(&degree.v) > 0) names.push_back(v);
        }
        return names;
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
        for (int v : variables(f)) {
            if (g.constant()) break;
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
        fmpz_mpoly_univar_t u;
        fmpz_mpoly_univar_init(u, ctx);
        fmpz_mpoly_to_univar(u, f.p, v, ctx);
        FibreTerms terms;
        for (slong i = fmpz_mpoly_univar_length(u, ctx); i-- > 0;) {
            Z exponent;
            fmpz_set(&exponent.v, u->exps + i);
            MP coefficient(ctx);
            fmpz_mpoly_univar_swap_term_coeff(coefficient.p, u, i, ctx);
            terms.emplace_back(std::move(exponent), std::move(coefficient));
        }
        fmpz_mpoly_univar_clear(u, ctx);
        return fibre_polynomial(std::move(terms));
    }
    MP constant_term(const Uni& p) const {
        return p.len && fmpz_is_zero(&p.at(0).first.v) ? p.at(0).second : MP(ctx);
    }
    // Homogeneous evaluation at v = -p0/c. Skip gaps between successive
    // exponents, including leading powers common to all terms.
    MP substituted(const MP& q, int v, const MP& p0, const fmpz_t c) const {
        const Uni parts = split(q, v);
        if (!parts.len || fmpz_is_zero(&parts.degree().v)) return q;
        if (p0.zero()) return constant_term(parts);
        MP minus(ctx), result(parts.lead()), factor(ctx), term(ctx);
        fmpz_mpoly_neg(minus.p, p0.p, ctx);
        Z gap, distance, scalar;
        auto multiply_power = [&]() {
            if (!fmpz_mpoly_pow_fmpz(factor.p, minus.p, &gap.v, ctx))
                throw Limit("The Euler substitution could not expand a polynomial power.");
            fmpz_mpoly_mul(result.p, result.p, factor.p, ctx);
        };
        for (size_t k = parts.len - 1; k > 0; --k) {
            const auto& next = parts.at(k - 1);
            fmpz_sub(&gap.v, &parts.at(k).first.v, &next.first.v);
            multiply_power();
            fmpz_sub(&distance.v, &parts.degree().v, &next.first.v);
            if (!fmpz_pow_fmpz(&scalar.v, c, &distance.v))
                throw Limit("The Euler substitution could not materialize a coefficient.");
            fmpz_mpoly_scalar_mul_fmpz(term.p, next.second.p, &scalar.v, ctx);
            fmpz_mpoly_add(result.p, result.p, term.p, ctx);
        }
        gap = parts.at(0).first;
        if (!fmpz_is_zero(&gap.v)) multiply_power();
        return result;
    }

    Uni derivative(const Uni& g) const {
        FibreTerms d;
        for (size_t k = 0; k < g.len; ++k) {
            const auto& [power, coefficient] = g.at(k);
            if (fmpz_is_zero(&power.v)) continue;
            Z exponent;
            fmpz_sub_ui(&exponent.v, &power.v, 1);
            MP term(ctx);
            fmpz_mpoly_scalar_mul_fmpz(term.p, coefficient.p, &power.v, ctx);
            d.emplace_back(std::move(exponent), std::move(term));
        }
        return fibre_polynomial(std::move(d));
    }
    // A pseudo-remainder of a by b: lc(b)^k * a = q*b + r with deg r < deg b,
    // then divided by integers and by factors of lc(b) common to its
    // coefficients. Where lc(b) is not zero it is the remainder up to a
    // function without zeros.
    Uni remainder(const Uni& a, const Uni& b) const {
        std::map<Z, MP, PowerLess> coefficients(a.terms->begin(), a.terms->begin() + a.len);
        const Z& db = b.degree();
        const MP& lb = b.lead();
        const bool monic = fmpz_mpoly_equal_si(lb.p, 1, ctx);
        MP term(ctx);
        while (!coefficients.empty() && fmpz_cmp(&coefficients.rbegin()->first.v, &db.v) >= 0) {
            auto leading = coefficients.extract(std::prev(coefficients.end()));
            Z shift;
            fmpz_sub(&shift.v, &leading.key().v, &db.v);
            if (!monic)
                for (auto& entry : coefficients) fmpz_mpoly_mul(entry.second.p, entry.second.p, lb.p, ctx);
            for (size_t k = 0; k + 1 < b.len; ++k) {
                const auto& [power, coefficient] = b.at(k);
                Z target;
                fmpz_add(&target.v, &power.v, &shift.v);
                auto entry = coefficients.try_emplace(std::move(target), ctx).first;
                fmpz_mpoly_mul(term.p, leading.mapped().p, coefficient.p, ctx);
                MP& value = entry->second;
                fmpz_mpoly_sub(value.p, value.p, term.p, ctx);
                if (value.zero()) coefficients.erase(entry);
            }
        }
        if (coefficients.empty()) return fibre_polynomial({});
        FibreTerms r;
        for (auto& entry : coefficients) r.emplace_back(entry.first, std::move(entry.second));
        MP content(r[0].second);
        for (size_t k = 1; k < r.size() && !fmpz_mpoly_equal_si(content.p, 1, ctx); ++k)
            content = gcd(content, r[k].second);
        Z integer;
        for (slong i = 0; i < fmpz_mpoly_length(content.p, ctx); ++i) {
            Z coefficient;
            fmpz_mpoly_get_term_coeff_fmpz(&coefficient.v, content.p, i, ctx);
            fmpz_gcd(&integer.v, &integer.v, &coefficient.v);
        }
        if (!fmpz_is_one(&integer.v))
            for (auto& entry : r) fmpz_mpoly_scalar_divexact_fmpz(entry.second.p, entry.second.p, &integer.v, ctx);
        normalize(content);
        while (!content.constant() && !lb.constant()) {
            MP common = gcd(content, lb);
            normalize(common);
            if (common.constant()) break;
            for (auto& entry : r) entry.second = quotient(entry.second, common);
            content = quotient(content, common);
        }
        return fibre_polynomial(std::move(r));
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
                } else if (fmpz_is_zero(&h.g.degree().v)) {
                    out.push_back(h);
                } else if (!t.some) {
                    todo.push_back({h.g, true, rest, h.z, h.u});
                } else {
                    const bool first = fmpz_cmp(&t.g.degree().v, &h.g.degree().v) >= 0;
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
    Z open_part(const Set& z, const Set& u) {
        if (has(z, 0) || !disjoint(z, u)) return {};
        Z value = closed(z);
        if (!u.empty()) {
            const auto excluded = closed(with(z, forbidden(u)));
            fmpz_sub(&value.v, &value.v, &excluded.v);
        }
        return value;
    }
    // Free affine coordinates contribute one. The equations alone index the
    // cache: they determine the active coordinate set in every projection.
    Z closed(const Set& equations) {
        if (has(equations, 0)) return {};
        if (equations.empty()) { Z one; fmpz_one(&one.v); return one; }
        const auto known = values.find(equations);
        if (known != values.end()) return known->second;
        Set names;
        for (int id : equations) names = united(names, support[id]);
        // Equations on disjoint coordinate sets define a Cartesian product.
        // Split the connected components of the variable/equation incidence graph.
        std::map<int, Set> groups;
        if (equations.size() > 1 && names.size() > 1) {
            std::vector<int> parent(n, -1);
            auto root = [&](int v) {
                while (parent[v] >= 0) {
                    if (parent[parent[v]] >= 0) parent[v] = parent[parent[v]];
                    v = parent[v];
                }
                return v;
            };
            for (int id : equations) {
                const auto& vars = support[id];
                for (int v : vars) {
                    int a = root(vars.front()), b = root(v);
                    if (a != b) parent[b] = a;
                }
            }
            for (int id : equations) groups[root(support[id].front())].push_back(id);
        }
        Z value;
        if (groups.size() <= 1) value = projected(names, equations);
        else {
            fmpz_one(&value.v);
            for (const auto& group : groups) {
                auto factor = closed(group.second);
                fmpz_mul(&value.v, &value.v, &factor.v);
                if (fmpz_is_zero(&value.v)) break;
            }
        }
        values.emplace(equations, value);
        return value;
    }
    Z projected(const Set& names, const Set& equations) {
        if (names.size() == 1) {
            MP g(conditions[static_cast<size_t>(equations[0])]);
            for (size_t i = 1; i < equations.size(); ++i) g = gcd(g, conditions[static_cast<size_t>(equations[i])]);
            Z count;
            fmpz_mpoly_degree_fmpz(&count.v, g.p, names[0], ctx);
            return count;
        }
        // An equation c*v + p0 with c a nonzero constant removes v: first those
        // that give v a value, then any.
        for (int pass = 0; pass < 2; ++pass) {
            for (int id : equations) {
                const MP& p = conditions[static_cast<size_t>(id)];
                const auto& here = support[static_cast<size_t>(id)];
                if (pass == 0 && here.size() != 1) continue;
                for (int v : names) {
                    Z degree;
                    fmpz_mpoly_degree_fmpz(&degree.v, p.p, v, ctx);
                    if (!fmpz_is_one(&degree.v)) continue;
                    const Uni parts = split(p, v);
                    if (!parts.lead().constant()) continue;
                    Z c;
                    fmpz_mpoly_get_fmpz(&c.v, parts.lead().p, ctx);
                    const MP p0 = constant_term(parts);
                    Set remaining;
                    for (int other : equations) {
                        if (other == id) continue;
                        const int next = condition(substituted(conditions[static_cast<size_t>(other)], v, p0, &c.v));
                        if (next >= 0) remaining = with(remaining, next);
                    }
                    return closed(remaining);
                }
            }
        }
        // The projection along the variable of least degree.
        int variable = names[0];
        Z least_max, least_sum;
        fmpz_set_si(&least_max.v, -1);
        for (int v : names) {
            Z maximum, sum, degree;
            for (int id : equations) {
                fmpz_mpoly_degree_fmpz(&degree.v, conditions[id].p, v, ctx);
                if (fmpz_cmp(&degree.v, &maximum.v) > 0) maximum = degree;
                fmpz_add(&sum.v, &sum.v, &degree.v);
            }
            const int cmp = fmpz_cmp(&maximum.v, &least_max.v);
            if (fmpz_sgn(&least_max.v) < 0 || cmp < 0 || (cmp == 0 && fmpz_cmp(&sum.v, &least_sum.v) < 0)) {
                least_max = maximum;
                least_sum = sum;
                variable = v;
            }
        }
        std::vector<Uni> fibre;
        Set below;
        for (int id : equations) {
            if (has(support[static_cast<size_t>(id)], variable)) fibre.push_back(split(conditions[static_cast<size_t>(id)], variable));
            else below = with(below, id);
        }
        Z value;
        for (const Case& piece : gcd_cases(fibre, below, Set())) {
            if (!piece.some || fmpz_is_one(&piece.g.degree().v)) {
                const auto part = open_part(piece.z, piece.u);
                fmpz_add(&value.v, &value.v, &part.v);
            } else if (fmpz_cmp_si(&piece.g.degree().v, 1) > 0) {
                std::vector<Uni> pair = {piece.g, derivative(piece.g)};
                for (const Case& repeated : gcd_cases(pair, piece.z, piece.u)) {
                    if (!repeated.some) throw Limit("A polynomial of known degree became zero.");
                    const auto part = open_part(repeated.z, repeated.u);
                    Z count;
                    fmpz_sub(&count.v, &piece.g.degree().v, &repeated.g.degree().v);
                    fmpz_addmul(&value.v, &part.v, &count.v);
                }
            }
        }
        return value;
    }
    Z integrate(const Set& closed_set, const std::vector<Set>& exclusions);
};

// Indicator inclusion-exclusion for nonprincipal excluded ideals.
Z Euler::State::integrate(const Set& closed_set, const std::vector<Set>& exclusions) {
    std::map<Set, Z> pieces;
    fmpz_one(&pieces[closed_set].v);
    Set inequations;
    for (const Set& forbidden : exclusions) {
        if (forbidden.empty()) return {};
        if (has(forbidden, 0)) continue;
        if (forbidden.size() == 1) {
            inequations = with(inequations, forbidden[0]);
            continue;
        }
        // The indicator of X minus V(I) is that of X minus that of X and V(I).
        std::map<Set, Z> next(pieces);
        for (const auto& piece : pieces) {
            auto& weight = next[united(piece.first, forbidden)];
            fmpz_sub(&weight.v, &weight.v, &piece.second.v);
        }
        pieces.clear();
        for (const auto& piece : next)
            if (!fmpz_is_zero(&piece.second.v)) pieces.insert(piece);
    }
    Z total;
    for (const auto& piece : pieces) {
        const auto chi = open_part(piece.first, inequations);
        fmpz_addmul(&total.v, &piece.second.v, &chi.v);
    }
    return total;
}

Euler::Euler(const Ring& R) : state_(new State(R)) {}
Euler::~Euler() { delete state_; }

Z Euler::characteristic(const std::vector<Poly>& equations, const std::vector<std::vector<Poly>>& excluded) {
    std::vector<Set> exclusions;
    for (const auto& ideal : excluded) exclusions.push_back(state_->set_of(ideal));
    return state_->integrate(state_->set_of(equations), exclusions);
}

// External FLINT presentation. It avoids the variable and degree limits of
// packed Groebner monomials when an Euler integral needs no such basis.
Z Euler::from_text(const std::string& input) {
    std::istringstream in(input);
    std::string header;
    if (!std::getline(in, header)) throw std::invalid_argument("Expected coordinate names.");
    std::istringstream names_in(header);
    std::vector<std::string> names;
    for (std::string name; names_in >> name;) names.push_back(name);
    if (names.empty() || names.size() > INT_MAX) throw Limit("Invalid Euler dimension.");
    std::vector<const char*> printed;
    for (const auto& name : names) printed.push_back(name.c_str());
    State state(static_cast<int>(names.size()));
    struct Reader {
        fmpq_mpoly_ctx_t ctx;
        fmpq_mpoly_t f;
        explicit Reader(int n) { fmpq_mpoly_ctx_init(ctx, n, ORD_LEX); fmpq_mpoly_init(f, ctx); }
        ~Reader() { fmpq_mpoly_clear(f, ctx); fmpq_mpoly_ctx_clear(ctx); }
    } reader(state.n);
    auto ideal = [&](size_t count) {
        Set ids;
        for (size_t i = 0; i < count; ++i) {
            std::string row;
            if (!std::getline(in, row) || fmpq_mpoly_set_str_pretty(reader.f, row.c_str(), printed.data(), reader.ctx))
                throw std::invalid_argument("Invalid Euler condition.");
            MP f(state.ctx);
            fmpz_mpoly_set(f.p, reader.f->zpoly, state.ctx);
            const int id = state.condition(f);
            if (id >= 0) ids = with(ids, id);
        }
        return ids;
    };
    if (!std::getline(in, header)) throw std::invalid_argument("Expected Euler ideal sizes.");
    std::istringstream sizes(header);
    size_t count;
    if (!(sizes >> count)) throw std::invalid_argument("Missing Euler equations.");
    const Set closed = ideal(count);
    std::vector<Set> excluded;
    while (sizes >> count) excluded.push_back(ideal(count));
    return state.integrate(closed, excluded);
}

// Nullity of the bounded logarithmic differential equation for a squarefree
// plane polynomial. Each column is the contribution of one numerator monomial.
static long absolute_factors(const MP& f, int x, int y) {
    const auto ctx = f.ctx;
    const long m = fmpz_mpoly_degree_si(f.p, x, ctx);
    const long n = fmpz_mpoly_degree_si(f.p, y, ctx);
    if (m == 0 || n == 0) return m + n;
    const long rows = 4 * m * n, cols = m * (n + 1) + (m + 1) * n;
    if (rows > 2000000 / cols) throw Limit("The absolute factor matrix exceeds the exact rank budget.");
    fmpz_mat_t matrix;
    fmpz_mat_init(matrix, rows, cols);
    long exponents[MAX_VARIABLES];
    Z coefficient;
    for (slong term = 0; term < fmpz_mpoly_length(f.p, ctx); ++term) {
        fmpz_mpoly_get_term_exp_si(exponents, f.p, term, ctx);
        fmpz_mpoly_get_term_coeff_fmpz(&coefficient.v, f.p, term, ctx);
        const long a = exponents[x], b = exponents[y];
        long column = 0;
        for (long i = 0; i < m; ++i)
            for (long j = 0; j <= n; ++j, ++column)
                if (j != b) fmpz_addmul_si(fmpz_mat_entry(matrix, (a+i)*2*n+b+j-1, column), &coefficient.v, j-b);
        for (long i = 0; i <= m; ++i)
            for (long j = 0; j < n; ++j, ++column)
                if (a != i) fmpz_addmul_si(fmpz_mat_entry(matrix, (a+i-1)*2*n+b+j, column), &coefficient.v, a-i);
    }
    const long count = cols - fmpz_mat_rank(matrix);
    fmpz_mat_clear(matrix);
    return count;
}

std::pair<long, long> Euler::plane_components(const Poly& equation, int x, int y,
        const std::vector<Poly>& region, const std::vector<std::vector<Poly>>& removed) {
    State& s = *state_;
    const MP f = s.converted(equation);
    const int id = s.condition(f);
    if (id < 0) throw Limit("Expected a nonzero plane equation.");
    if (id == 0) return {0, 0};
    struct Factors {
        fmpz_mpoly_factor_t list;
        const fmpz_mpoly_ctx_struct* ctx;
        explicit Factors(const fmpz_mpoly_ctx_struct* c) : ctx(c) { fmpz_mpoly_factor_init(list, ctx); }
        ~Factors() { fmpz_mpoly_factor_clear(list, ctx); }
    } factors(s.ctx);
    if (!fmpz_mpoly_factor(factors.list, s.conditions[id].p, s.ctx)) throw Limit("Plane factorization failed.");
    long geometric = 0, rational = 0;
    for (slong i = 0; i < fmpz_mpoly_factor_length(factors.list, s.ctx); ++i) {
        MP factor(s.ctx), quotient(s.ctx);
        fmpz_mpoly_factor_get_base(factor.p, factors.list, i, s.ctx);
        auto vanishes = [&](const std::vector<Poly>& equations) {
            return std::all_of(equations.begin(), equations.end(), [&](const Poly& g) {
                return fmpz_mpoly_divides(quotient.p, s.converted(g).p, factor.p, s.ctx);
            });
        };
        if (!vanishes(region) || std::any_of(removed.begin(), removed.end(), vanishes)) continue;
        ++rational;
        geometric += absolute_factors(factor, x, y);
    }
    return {geometric, rational};
}

}  // namespace atlas

extern "C" char* atlas_euler(const char* input, size_t length) noexcept {
    try {
        const auto chi = atlas::Euler::from_text(std::string(input, length));
        char* decimal = fmpz_get_str(nullptr, 10, &chi.v);
        char* answer = atlas::copy_text(decimal);
        flint_free(decimal);
        return answer;
    } catch (const std::exception& e) {
        atlas::set_error(e.what());
    } catch (...) {
        atlas::set_error("Unknown Euler characteristic failure.");
    }
    return nullptr;
}
