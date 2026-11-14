// Sparse division in the original affine coordinates at a rational point.
#include "local_point_normal_form.hpp"
#include "local_point_basis.hpp"
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include <algorithm>

namespace atlas::local {
namespace {
mpz_class degree(const Powers& e) {
    mpz_class d = 0;
    for (const auto& k : e) d += k;
    return d;
}

mpz_class max_degree(const Polynomial& f) {
    mpz_class d = 0;
    for (const auto& term : f) d = std::max(d, degree(term.first));
    return d;
}

bool divides(const Powers& head, const Powers& e) {
    return std::equal(head.begin(), head.end(), e.begin(), std::less_equal<>());
}

void add_multiple(Polynomial& target, const Polynomial& source, const mpq_class& scale) {
    for (const auto& [e, c] : source) {
        auto it = target.try_emplace(e, 0).first;
        it->second += scale * c;
        if (it->second == 0) target.erase(it);
    }
}

PointRow inverse(PointRow p) {
    for (auto& q : p) q = -q;
    return p;
}

// A translation preserves total degree. A multiplier (x-p)^shift is
// expanded only when that particular monomial is used by Mora division.
struct Divisor {
    Polynomial row;
    Powers head;
    mpq_class coefficient;
    mpz_class ecart;
    Polynomial unit;
};

NormalForm mora(const Basis& basis, Polynomial pending, const PointRow& point, bool lex, bool track_unit) {
    const size_t width = point.size();
    Polynomial unit{{Powers(width),1}};
    if (basis.empty()) return {unit, std::move(pending), false};
    std::vector<Divisor> reducers;
    for (const auto& row : basis) if (!row.empty()) {
        auto [h, c] = leading_term_at_point(row, point, lex);
        if (degree(h) == 0) return {unit, {}, false};
        reducers.push_back({row, h, c, max_degree(row) - degree(h), {}});
    }
    const auto back = inverse(point);
    std::map<Powers, Polynomial> shifts;
    while (!pending.empty()) {
        auto [h, c] = leading_term_at_point(pending, point, lex);
        size_t choice = reducers.size();
        for (size_t j = 0; j < reducers.size(); ++j)
            if (divides(reducers[j].head, h) &&
                (choice == reducers.size() || reducers[j].ecart < reducers[choice].ecart)) choice = j;
        if (choice == reducers.size()) break;
        const mpz_class ecart = max_degree(pending)-degree(h);
        if (reducers[choice].ecart > ecart)
            reducers.push_back({pending, h, c, ecart, track_unit ? unit : Polynomial{}});
        const auto& divisor = reducers[choice];
        Powers shift = h;
        for (size_t i = 0; i < width; ++i) shift[i] -= divisor.head[i];
        const mpq_class scale = -c/divisor.coefficient;
        if (degree(shift) == 0) {
            add_multiple(pending, divisor.row, scale);
            if (track_unit) add_multiple(unit, divisor.unit, scale);
        } else {
            auto it = shifts.find(shift);
            if (it == shifts.end()) {
                auto factor = translate_unbounded({Polynomial{{shift,1}}}, back);
                it = shifts.emplace(std::move(shift), std::move(factor.front())).first;
            }
            add_multiple(pending, primary::multiply(it->second, divisor.row), scale);
            if (track_unit && !divisor.unit.empty())
                add_multiple(unit, primary::multiply(it->second, divisor.unit), scale);
        }
    }
    return {std::move(unit), std::move(pending), true};
}

// In characteristic zero, a term divisible by x^h exists exactly when
// the mixed derivative of order h is nonzero as a polynomial. Translation
// commutes with differentiation and preserves nonzero polynomials, so the
// test in the original coordinates also decides divisibility in the shift.
bool may_reduce(const Polynomial& f, const std::vector<Powers>& heads) {
    for (const auto& [e, c] : f)
        for (const auto& h : heads) if (divides(h, e)) return true;
    return false;
}

// Project onto exponents < bound on one centered axis, then express the
// result in original coordinates. Carry the other axes as sparse labels.
Polynomial below_axis(const Polynomial& f, size_t axis, const mpz_class& bound, const PointRow& point) {
    Polynomial retained;
    if (f.empty()) return retained;
    if (point[axis] == 0) {
        for (const auto& t : f) if (t.first[axis] < bound) retained.insert(t);
        return retained;
    }
    using Slices = std::map<Powers, Polynomial>;
    Slices slices;
    for (const auto& [e, c] : f) {
        auto key = e;
        key[axis] = 0;
        slices[key].emplace(Powers{e[axis]}, c);
    }
    for (const auto& [label, slice] : slices) {
        Polynomial restored;
        if (slice.rbegin()->first[0] < bound) restored = slice;
        else {
            auto jet = point_jet_below(bound, slice, {point[axis]});
            restored = translate_unbounded({jet}, {-point[axis]}).front();
        }
        for (const auto& [e, c] : restored) {
            auto m = label;
            m[axis] = e[0];
            retained.emplace(std::move(m), c);
        }
    }
    return retained;
}

bool monomial_associate(const Polynomial& f, const Powers& head, const PointRow& point) {
    for (size_t i = 0; i < head.size(); ++i) if (head[i] != 0) {
        if (point[i] == 0) {
            if (std::any_of(f.begin(), f.end(), [&](const auto& term) { return term.first[i] < head[i]; }))
                return false;
            continue;
        }
        std::map<Powers, Polynomial> slices;
        for (const auto& [e,c] : f) {
            auto label = e;
            label[i] = 0;
            slices[label].emplace(Powers{e[i]}, c);
        }
        for (const auto& slice : slices)
            if (order_at_point(slice.second, {point[i]}) < head[i]) return false;
    }
    return true;
}

// For an ideal of coordinate powers, truncate the shifted axes together.
// Powers of rational coordinates may cancel before any one power is representable.
std::optional<Polynomial> coordinate_power_remainder(const Basis& basis, const std::vector<Powers>& heads,
        const Polynomial& query, const PointRow& point, const mpz_class& query_degree) {
    Powers bounds(point.size());
    for (const auto& head : heads) {
        auto first = std::find_if(head.begin(), head.end(), [](const auto& k) { return k != 0; });
        if (first == head.end() || std::any_of(first + 1, head.end(), [](const auto& k) { return k != 0; }))
            return std::nullopt;
        auto& bound = bounds[first - head.begin()];
        if (bound == 0 || *first < bound) bound = *first;
    }
    size_t index = 0;
    for (const auto& row : basis)
        if (!row.empty() && !monomial_associate(row, heads[index++], point)) return std::nullopt;

    // An unconstrained axis, or an axis whose degrees already fit its bound,
    // stays in the original coordinates and does not expand into a dense jet.
    auto shifts = point;
    for (size_t axis = 0; axis < point.size(); ++axis) {
        if (bounds[axis] == 0 || std::none_of(query.begin(), query.end(), [&](const auto& term) {
            return term.first[axis] >= bounds[axis];
        })) shifts[axis] = 0;
    }
    auto jet = grouped_point_jet(query, shifts, query_degree + 1, bounds);
    return translate_unbounded({jet}, inverse(shifts)).front();
}

// Each filter removes exactly the centered monomials divisible by one
// leading monomial. These projections commute, including on a shifted axis.
Polynomial project_standard(Polynomial f, const std::vector<Powers>& heads, const PointRow& point) {
    for (const auto& h : heads) {
        Polynomial multiple = f;
        for (size_t i = 0; i < h.size() && !multiple.empty(); ++i) if (h[i] != 0) {
            auto lower = below_axis(multiple, i, h[i], point);
            add_multiple(multiple, lower, -1);
        }
        add_multiple(f, multiple, -1);
    }
    return f;
}

// For one divisor A*x_i+B with a unit A, clearing the denominator of
// substitution for x_i can bypass an arbitrarily long series of Mora steps.
std::optional<NormalForm> linear_fraction(const Polynomial& row, const Powers& head,
        const Polynomial& query, const PointRow& point, bool lex) {
    if (degree(head) != 1) return std::nullopt;
    const size_t axis = std::find(head.begin(), head.end(),1)-head.begin();
    Polynomial a, b;
    for (const auto& [e, c] : row) {
        if (e[axis] > 1) return std::nullopt;
        auto m = e;
        m[axis] = 0;
        if (e[axis] != 0) a.emplace(std::move(m), c);
        else b.emplace(std::move(m), -c);
    }
    if (a.empty() || !nonzero_at_point(a, point)) return std::nullopt;
    const auto scalar = leading_term_at_point(a, point, lex).second;
    for (auto& term : a) term.second /= scalar;
    for (auto& term : b) term.second /= scalar;
    mpz_class top = 0;
    for (const auto& [e, c] : query) top = std::max(top, e[axis]);
    // A dense denominator power can cost more than the dividend. In
    // that case retain Mora's weak division with sparse multipliers.
    if (top > 16 && (a.size() > 1 || b.size() > 1)) return std::nullopt;
    Polynomial remainder;
    for (const auto& [e, c] : query) {
        const auto k = e[axis];
        auto m = e;
        m[axis] = 0;
        auto coefficient = primary::multiply(primary::power(a, top - k, point.size()),
                                            primary::power(b, k, point.size()));
        auto term = primary::multiply(Polynomial{{m, c}}, coefficient);
        add_multiple(remainder, term,1);
    }
    return NormalForm{primary::power(a, top, point.size()), std::move(remainder), false};
}
}

Polynomial mora_remainder_at_point(const Basis& reducers, const Polynomial& query,
        const PointRow& point, bool lex) {
    return mora(reducers, query, point, lex, false).remainder;
}

NormalForm normal_form_at_point(const Basis& generators, const Polynomial& query,
        const std::vector<std::string>& names, const PointRow& point,
        bool lex, bool prefer_modular, bool known_basis) {
    const Polynomial one{{Powers(point.size()),1}};
    if (query.empty() || generators.empty()) return {one, query, false};
    // If every generator belongs to m^(d+1), so does the entire local
    // ideal. A degree-d query has no divisible centered term.
    const mpz_class query_degree = max_degree(query);
    if (query_degree <= 4 && std::all_of(generators.begin(), generators.end(),[&](const auto& g) {
        return g.empty() || order_at_point(g, point) > query_degree;
    })) return {one, query, false};
    Basis basis = known_basis ? generators : standard_basis_at_point(generators, names, point, lex, prefer_modular);
    std::vector<Powers> heads;
    for (const auto& row : basis) if (!row.empty()) {
        auto h = leading_term_at_point(row, point, lex).first;
        if (degree(h) == 0) return {one, {}, false};
        heads.push_back(std::move(h));
    }
    if (!may_reduce(query, heads)) return {one, query, false};
    if (auto projected = coordinate_power_remainder(basis, heads, query, point, query_degree))
        return {one, std::move(*projected), false};
    const auto inverse_point = inverse(point);
    for (size_t i = 0, index = 0; i < basis.size(); ++i) {
        if (basis[i].empty()) continue;
        const auto& head = heads[index++];
        if (degree(head) != 1) continue;
        const size_t axis = std::find(head.begin(), head.end(), 1) - head.begin();
        Polynomial image;
        if (max_degree(basis[i]) == 1) {
            const auto coefficient = basis[i].at(head);
            for (const auto& [e, c] : basis[i]) if (!e[axis]) {
                auto m = e;
                m.erase(m.begin() + axis);
                image.emplace(std::move(m), -c / coefficient);
            }
        } else if (basis[i].size() <= 2 && monomial_associate(basis[i], head, point)) {
            if (point[axis] != 0) image.emplace(Powers(point.size() - 1), point[axis]);
        } else continue;

        // The equation x_i = image has local head x_i-p_i. After substituting
        // it, the other heads involve only the remaining variables. A basis
        // there lifts to a standard basis together with the pivot equation.
        // Avoid expanding a large power of a multi-term image speculatively.
        auto expensive = [&](const Polynomial& f) {
            return image.size() > 1 && std::any_of(f.begin(), f.end(),
                       [&](const auto& t) { return t.first[axis] > 16; });
        };
        if (expensive(query) || std::any_of(basis.begin(), basis.end(), expensive)) continue;
        std::map<mpz_class, Polynomial> powers;
        auto substitute = [&](const Polynomial& f) {
            if (image.empty()) return specialize_coordinate(f, axis, 0);
            if (image.size() == 1 && degree(image.begin()->first) == 0)
                return specialize_coordinate(f, axis, image.begin()->second);
            return primary::substitute(f, axis, image, point.size() - 1, powers);
        };
        Basis section;
        for (size_t j = 0; j < basis.size(); ++j) if (j != i) {
            auto row = substitute(basis[j]);
            if (!row.empty()) section.push_back(std::move(row));
        }
        auto smaller_names = names;
        auto smaller_point = point;
        smaller_names.erase(smaller_names.begin() + axis);
        smaller_point.erase(smaller_point.begin() + axis);
        auto result = normal_form_at_point(section, substitute(query), smaller_names,
                                           smaller_point, lex, prefer_modular, false);
        auto lift = [&](const Polynomial& row) {
            Polynomial lifted;
            for (const auto& [e, c] : row) {
                auto m = e;
                m.insert(m.begin() + axis, 0);
                lifted.emplace(std::move(m), c);
            }
            return lifted;
        };
        return {lift(result.unit), lift(result.remainder), result.weak};
    }
    bool monomials = true, homogeneous = true;
    size_t j = 0;
    for (const auto& row : basis) if (!row.empty()) {
        if (monomials) monomials = monomial_associate(row, heads[j], point);
        homogeneous &= max_degree(row)==degree(heads[j]);
        ++j;
    }
    const auto cutoff = finite_cutoff(heads, point.size());
    // A small total-degree jet keeps cancellations between different
    // rational coordinates together. For a wide monomial staircase use
    // one-axis filters, which skip untouched sparse powers.
    if (monomials && (!cutoff || *cutoff > 32))
        return {one, project_standard(query, heads, point), false};
    if (cutoff) {
        // Above the cutoff every homogeneous component of the associated
        // graded quotient vanishes, hence m^cutoff = 0 by Nakayama. Divisors
        // of order >= cutoff can be omitted in this truncated division.
        Basis jets;
        for (const auto& row : basis) {
            auto jet = point_jet_below(*cutoff, row, point);
            if (!jet.empty()) jets.push_back(std::move(jet));
        }
        auto f = point_jet_below(*cutoff, query, point);
        auto value = normal_form(jets, f, point.size(), lex, cutoff);
        return {one, translate_unbounded({value.remainder}, inverse_point).front(), false};
    }
    if (basis.size()== 1)
        if (auto fraction = linear_fraction(basis.front(), heads.front(), query, point, lex)) return *fraction;
    if (homogeneous) {
        // Only the reducible portion needs centering and homogeneous division.
        auto retained = project_standard(query, heads, point);
        auto reduce = query;
        add_multiple(reduce, retained, -1);
        if (reduce.empty()) return {one, query, false};
        auto centered = translate_unbounded(basis, point);
        auto f = translate_unbounded({reduce}, point).front();
        auto nf = normal_form(centered, f, point.size(), lex);
        add_multiple(retained, translate_unbounded({nf.remainder}, inverse(point)).front(),1);
        return {one, std::move(retained), false};
    }
    auto result = mora(basis, query, point, lex, true);
    result.weak = may_reduce(result.remainder, heads);
    return result;
}
}
