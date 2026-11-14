#include "local_division.hpp"
#include "local_io.hpp"
#include "local_point_basis.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas::local {
namespace {

using Exponent = mpz_class;

Exponent degree(const Powers& powers) {
    Exponent sum = 0;
    for (const auto& e : powers) sum += e;
    return sum;
}

Exponent max_degree(const Polynomial& p) {
    Exponent highest = 0;
    for (const auto& term : p) highest = std::max(highest, degree(term.first));
    return highest;
}

bool divides(const Powers& divisor, const Powers& target) {
    for (std::size_t i = 0; i < target.size(); ++i)
        if (divisor[i] > target[i]) return false;
    return true;
}

void accumulate(Polynomial& p, Powers powers, const mpq_class& coefficient) {
    if (coefficient == 0) return;
    auto [it, added] = p.try_emplace(std::move(powers), coefficient);
    if (!added) {
        it->second += coefficient;
        if (it->second == 0) p.erase(it);
    }
}

}
std::optional<mpz_class> finite_cutoff(const std::vector<Powers>& heads, std::size_t width) {
    // If x_i^b_i belongs to the initial ideal for each i, all graded
    // pieces of degree >= N vanish. Nakayama gives m^N=0 locally.
    Powers bounds(width, 0);
    for (const auto& h : heads) {
        std::size_t axis = width;
        for (std::size_t i = 0; i < width; ++i) {
            if (!h[i]) continue;
            if (axis != width) { axis = width; break; }
            axis = i;
        }
        if (axis < width && (!bounds[axis] || h[axis] < bounds[axis])) bounds[axis] = h[axis];
    }
    if (std::find(bounds.begin(), bounds.end(), 0UL) != bounds.end()) return std::nullopt;
    Exponent cutoff = 1;
    for (auto b : bounds) cutoff += b - 1;
    return cutoff;
}

namespace {
struct Reducer {
    Polynomial row;
    Powers head;
    Exponent ecart;
    // Coefficient of the original dividend in a saved Mora row.
    Polynomial unit;
};

void subtract_multiple(Polynomial& p, const Polynomial& row, const Powers& shift,
                       const mpq_class& factor, std::optional<Exponent> cutoff = std::nullopt) {
    for (const auto& [m, c] : row) {
        Powers shifted = m;
        for (std::size_t i = 0; i < shifted.size(); ++i)
            shifted[i] += shift[i];
        if (!cutoff || degree(shifted) < *cutoff) accumulate(p, std::move(shifted), -factor * c);
    }
}

mpq_class rational_power(const mpq_class& q, Exponent n) {
    if (!n || q == 1) return 1;
    if (q == -1) return mpz_odd_p(n.get_mpz_t()) ? -1 : 1;
    if (!n.fits_ulong_p()) throw std::overflow_error("Rational power exceeds addressable memory.");
    mpq_class result;
    mpz_pow_ui(mpq_numref(result.get_mpq_t()), mpq_numref(q.get_mpq_t()), n.get_ui());
    mpz_pow_ui(mpq_denref(result.get_mpq_t()), mpq_denref(q.get_mpq_t()), n.get_ui());
    return result;
}

// Jump along repeated reductions by a homogeneous binomial. For x-y
// this sends x^n to y^n without constructing any intermediate monomials.
void reduce_homogeneous_binomial(Polynomial& pending, const Powers& m, const Reducer& r,
                                 const std::vector<Reducer>& reducers) {
    auto tail = r.row.begin();
    if (tail->first == r.head) ++tail;
    const auto& t = tail->first;
    Exponent count = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (t[i] < r.head[i]) {
            Exponent stride = r.head[i] - t[i];
            const Exponent steps = (m[i] - r.head[i]) / stride + 1;
            if (!count || steps < count) count = steps;
        }
    }
    Powers end = m;
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (t[i] <= r.head[i]) end[i] -= count * (r.head[i] - t[i]);
        else {
            Exponent stride = t[i] - r.head[i];
            end[i] += count * stride;
        }
    }
    auto source = pending.extract(m);
    // In (x-2*y, y), the terminal monomial of x^N is already zero;
    // the intermediate coefficient 2^N need never be materialized.
    if (std::any_of(reducers.begin(), reducers.end(), [&](const Reducer& other) {
            return other.row.size() == 1 && divides(other.head, end);
        })) return;
    const mpq_class ratio = -tail->second / r.row.at(r.head);
    mpq_class coefficient = source.mapped() * rational_power(ratio, count);
    accumulate(pending, std::move(end), coefficient);
}

// Value of the shifted axes at the center. Accumulate rational powers
// of equal or reciprocal magnitudes before exponentiating.
mpq_class center_coefficient(mpq_class c, const Powers& m, const std::vector<mpq_class>& point) {
    std::map<mpq_class, mpz_class> exponents;
    for (std::size_t i = 0; i < m.size(); ++i) {
        if (!m[i] || point[i] == 0) continue;
        mpq_class mag = abs(point[i]);
        if (point[i] < 0 && mpz_odd_p(m[i].get_mpz_t())) c = -c;
        if (mag > 1) exponents[mag] += m[i];
        if (mag < 1) exponents[1 / mag] -= m[i];
    }
    for (const auto& [q, n] : exponents) {
        mpz_class magnitude = abs(n);
        mpq_class power = rational_power(q, magnitude);
        if (n < 0) c /= power;
        else c *= power;
    }
    return c;
}

}

NormalForm normal_form(const Basis& basis, const Polynomial& centered, std::size_t width, bool lex,
                       std::optional<mpz_class> known_cutoff) {
    Powers zero(width, 0);
    NormalForm result{{{zero, 1}}, {}, false};
    if (centered.empty()) return result;
    std::vector<Reducer> reducers;
    std::vector<Powers> heads;
    for (const auto& g : basis) {
        if (g.empty()) continue;
        if (g.count(zero)) return result;
        auto head = local_head(g, lex);
        heads.push_back(head);
        bool associate = std::all_of(g.begin(), g.end(), [&](const auto& term) {
            return divides(head, term.first);
        });
        if (associate) reducers.push_back({{{head, 1}}, head, 0, {}});
        else reducers.push_back({g, head, max_degree(g) - degree(head), {}});
    }
    auto reducible = [&](const Powers& m) {
        return std::any_of(heads.begin(), heads.end(), [&](const Powers& h) { return divides(h, m); });
    };
    if (std::all_of(reducers.begin(), reducers.end(), [](const Reducer& r) { return r.row.size() == 1; })) {
        for (const auto& [m, c] : centered)
            if ((!known_cutoff || degree(m) < *known_cutoff) && !reducible(m)) result.remainder.emplace(m, c);
        return result;
    }
    const bool homogeneous = std::all_of(reducers.begin(), reducers.end(),
                                          [](const Reducer& r) { return !r.ecart; });
    const auto cutoff = known_cutoff ? std::move(known_cutoff) : finite_cutoff(heads, width);
    const bool full = homogeneous || cutoff.has_value();
    Polynomial pending;
    for (const auto& [m, c] : centered) {
        if (cutoff && degree(m) >= *cutoff) continue;
        if (full && !reducible(m)) result.remainder.emplace(m, c);
        else pending.emplace(m, c);
    }
    while (!pending.empty()) {
        Powers head = local_head(pending, lex);
        std::size_t choice = reducers.size();
        for (std::size_t i = 0; i < reducers.size(); ++i) {
            if (divides(reducers[i].head, head) &&
                    (choice == reducers.size() || reducers[i].ecart < reducers[choice].ecart)) choice = i;
        }
        if (choice == reducers.size()) {
            if (!full) {
                result.weak = std::any_of(pending.begin(), pending.end(),
                                          [&](const auto& term) { return reducible(term.first); });
                result.remainder = std::move(pending);
                break;
            }
            accumulate(result.remainder, head, pending.at(head));
            pending.erase(head);
            continue;
        }
        if (homogeneous && reducers[choice].row.size() == 2) {
            reduce_homogeneous_binomial(pending, head, reducers[choice], reducers);
            continue;
        }
        if (!full && reducers[choice].ecart > 0) {
            Exponent ecart = max_degree(pending) - degree(head);
            if (reducers[choice].ecart > ecart)
                reducers.push_back({pending, head, ecart, result.unit});
        }
        const auto& r = reducers[choice];
        Powers shift = head;
        for (std::size_t i = 0; i < width; ++i) shift[i] -= r.head[i];
        mpq_class factor = pending.at(head) / r.row.at(r.head);
        subtract_multiple(pending, r.row, shift, factor, cutoff);
        subtract_multiple(result.unit, r.unit, shift, factor);
    }
    return result;
}

Polynomial point_jet_below(const mpz_class& cutoff, const Polynomial& original, const std::vector<mpq_class>& point) {
    for (const auto& [e,c] : original)
        for (size_t i=0; i<e.size(); ++i)
            if (point[i] != 0 && abs(point[i]) != 1 && e[i] > 4096)
                return grouped_point_jet(original,point,cutoff);
    struct Term { Powers powers; mpq_class coefficient; Exponent room; };
    Polynomial jet;
    for (const auto& [m, c] : original) {
        Powers base = m;
        for (std::size_t i = 0; i < m.size(); ++i) if (point[i] != 0) base[i] = 0;
        Exponent fixed = degree(base);
        if (fixed >= cutoff) continue;
        std::vector<Term> terms{{std::move(base), center_coefficient(c, m, point), cutoff - 1 - fixed}};
        for (std::size_t i = 0; i < m.size(); ++i) {
            if (point[i] == 0 || !m[i]) continue;
            auto size = terms.size();
            for (std::size_t j = 0; j < size; ++j) {
                Term term = terms[j];
                Exponent bound = std::min(m[i], term.room);
                for (Exponent k = 1; k <= bound; ++k) {
                    term.powers[i] = k;
                    term.coefficient *= m[i] - k + 1;
                    term.coefficient /= point[i] * k;
                    terms.push_back({term.powers, term.coefficient, term.room - k});
                }
            }
        }
        for (auto& term : terms) accumulate(jet, std::move(term.powers), term.coefficient);
    }
    return jet;
}

namespace {
std::optional<Polynomial> query_jet_below(const Exponent& cutoff, const Polynomial& original,
                                          const std::vector<mpq_class>& point) {
    mpz_class full = 0, retained = 0;
    for (const auto& [m, c] : original) {
        mpz_class box = 1, truncated_box = 1;
        Exponent fixed_degree = 0;
        unsigned long axes = 0;
        for (std::size_t i = 0; i < m.size(); ++i) {
            if (point[i] == 0) fixed_degree += m[i];
            else if (m[i]) {
                box *= mpz_class(m[i]) + 1;
                ++axes;
            }
        }
        full += box;
        if (fixed_degree >= cutoff) continue;
        Exponent room = cutoff - 1 - fixed_degree;
        for (std::size_t i = 0; i < m.size(); ++i)
            if (point[i] != 0) truncated_box *= mpz_class(std::min(m[i], room)) + 1;
        mpz_class simplex, top = mpz_class(room) + axes;
        mpz_bin_ui(simplex.get_mpz_t(), top.get_mpz_t(), axes);
        retained += std::min(simplex, truncated_box);
    }
    if (full < 128 || 4 * retained > full) return std::nullopt;
    return point_jet_below(cutoff, original, point);
}
}



std::optional<Polynomial> finite_query_jet(const Basis& basis, const Polynomial& original,
                                           const std::vector<mpq_class>& point, bool lex) {
    if (std::none_of(point.begin(), point.end(), [](const mpq_class& q) { return q != 0; }))
        return std::nullopt;
    std::vector<Powers> heads;
    for (const auto& g : basis) {
        if (g.empty()) continue;
        auto head = local_head(g, lex);
        if (degree(head) == 0) return Polynomial{};
        heads.push_back(std::move(head));
    }
    auto cutoff = finite_cutoff(heads, point.size());
    return cutoff ? query_jet_below(*cutoff, original, point) : std::nullopt;
}

std::optional<Polynomial> homogeneous_normal_form(const Basis& centered, const Polynomial& original,
    const std::vector<mpq_class>& point, bool lex, bool prefer_modular, const LocalStage& stage) {
    auto basis = homogeneous_reducer(centered, point.size(), lex, prefer_modular);
    if (!basis) return std::nullopt;
    if (stage) stage("homogeneous_native_standard_basis", "local_standard_basis");
    std::optional<Polynomial> shifted;
    if (std::any_of(point.begin(), point.end(), [](const mpq_class& q) { return q != 0; })) {
        if (const auto& cutoff = basis->finite_cutoff())
            shifted = query_jet_below(*cutoff, original, point);
        if (!shifted) shifted = std::move(translate({original}, point).front());
    }
    const auto& query = shifted ? *shifted : original;
    if (stage) stage("center_query_at_point", nullptr);
    auto remainder = basis->reduce(query);
    if (remainder) {
        if (stage) stage("homogeneous_native_normal_form", "local_normal_form");
    } else {
        remainder = normal_form(basis->materialize(), query, point.size(), lex).remainder;
        if (stage) stage("local_normal_form", "local_normal_form");
    }
    return remainder;
}

}
