#include "local_point_basis.hpp"
#include "local_io.hpp"
#include "local_division.hpp"
#include "local_power_coordinates.hpp"
#include "local_point_normal_form.hpp"
#include "local_parameter_basis.hpp"
#include "primary_decomposition.hpp"
#include <algorithm>
#include <queue>
#include <set>

namespace atlas::local {
bool use_sparse_point_basis(const Basis& ideal, const PointRow& point) {
    if (std::none_of(point.begin(), point.end(), [](const auto& q) { return q != 0; })) return false;
    size_t count = 0;
    bool moderate = false;
    bool mixed_power = false;
    for (const auto& row : ideal) for (const auto& [e,c] : row) {
        size_t box = 1;
        unsigned moving_axes = 0;
        for (size_t i = 0; i < point.size(); ++i) if (point[i] != 0) {
            if (e[i] >= (4096+box-1)/box-1) return true;
            moderate |= e[i] >= 12;
            moving_axes += e[i] != 0;
            box *= 1+e[i].get_ui();
        }
        mixed_power |= moving_axes > 1 && box >= 64;
        count += box;
        if (count >= 16384) return true;
    }
    if (mixed_power && ideal.size() >= 3) return true;
    // Small power charts can spare a large homogeneous lift even when
    // the shift itself would fit comfortably, as for three quadrics in
    // x^17-1, y and z. Only select this path for a compressible axis.
    if (moderate && ideal.size() > 1) for (size_t i = 0; i < point.size(); ++i) if (point[i] != 0) {
        // An explicit coordinate section eliminates the whole shifted
        // power, even if another equation's linear term breaks its stride.
        bool large_axis = std::any_of(ideal.begin(), ideal.end(), [&](const auto& f) {
            return std::any_of(f.begin(), f.end(), [&](const auto& t) { return t.first[i] >= 12; });
        });
        if (large_axis) {
            Powers axis(point.size()), zero(point.size());
            axis[i] = 1;
            for (const auto& f : ideal) if (f.size() == 2) {
                auto a = f.find(axis), b = f.find(zero);
                if (a != f.end() && b != f.end() && b->second == -point[i]*a->second) return true;
            }
        }
        mpz_class step = 0;
        for (const auto& f : ideal) if (!f.empty() && step != 1) {
            const auto& first = f.begin()->first[i];
            for (const auto& [e,c] : f) {
                mpz_class difference = e[i]-first;
                mpz_gcd(step.get_mpz_t(), step.get_mpz_t(), difference.get_mpz_t());
                if (step == 1) break;
            }
        }
        if (step >= 8) return true;
    }
    return false;
}

namespace {
void normalize_at_point(Polynomial& f, const PointRow& point, bool lex) {
    if (f.empty()) return;
    auto [head, coefficient] = leading_term_at_point(f, point, lex);
    if (coefficient != 1) for (auto& term : f) term.second /= coefficient;
}

// Constant row operations and removal of monomial units preserve the local
// ideal. Sorting makes the result independent of the generator order.
Basis strip_units(const Basis& ideal, const PointRow& point) {
    Basis rows;
    for (const auto& f : ideal) if (!f.empty()) {
        auto common = f.begin()->first;
        for (size_t i = 0; i < point.size(); ++i) {
            if (point[i] == 0) common[i] = 0;
            else for (const auto& [e,c] : f) common[i] = std::min(common[i], e[i]);
        }
        const auto lead = f.rbegin()->second;
        Polynomial row;
        for (const auto& [e,c] : f) {
            Powers powers(e);
            for (size_t i = 0; i < point.size(); ++i) powers[i] -= common[i];
            row.emplace(std::move(powers), c/lead);
        }
        rows.push_back(std::move(row));
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

bool divides(const Powers& a, const Powers& b) {
    return std::equal(a.begin(), a.end(), b.begin(), std::less_equal<>());
}

// A separated factor h(x_i) is a unit times (x_i-p_i)^m in the local
// ring. Detect h without factoring or dividing wide dense polynomials:
// the univariate slices at distinct complementary monomials must all
// be rational multiples of the same slice.
bool simplify_separated_axis(Polynomial& row, size_t i, const PointRow& point) {
    if (point[i] == 0 || !std::any_of(row.begin(),row.end(),[&](const auto& t){return t.first[i]>=64;}))
        return false;
    std::map<Powers,Polynomial> slices;
    for (const auto& [e,c] : row) {
        auto key = e;
        key[i] = 0;
        slices[key].emplace(Powers{e[i]}, c);
    }
    const auto& h = slices.begin()->second;
    Polynomial other;
    for (const auto& [e,slice] : slices) {
        if (slice.size() != h.size()) return false;
        const mpq_class scale = slice.begin()->second / h.begin()->second;
        auto t = slice.begin();
        for (const auto& [a,c] : h) {
            if (a != t->first || scale*c != t->second) return false;
            ++t;
        }
        other.emplace(e, scale);
    }
    const auto multiplicity = order_at_point(h, {point[i]});
    if (multiplicity >= h.rbegin()->first.front()) return false;
    Powers shifted(point.size());
    shifted[i] = multiplicity;
    auto back = point;
    for (auto& q : back) q = -q;
    const auto coordinate_power = translate_unbounded({Polynomial{{shifted,1}}},back).front();
    row = primary::multiply(other,coordinate_power);
    return true;
}

std::optional<Basis> finite_jet_at_point(const Basis& ideal, const std::vector<std::string>& names,
                                        const PointRow& point, bool lex, bool prefer_modular) {
    // Bound only this speculative calculation. The sparse Mora algorithm
    // below handles ideals for which a short finite jet gives no certificate.
    // By the height theorem, fewer equations than coordinates cannot
    // define an isolated supported point in this regular local ring.
    if (point.size() > 4 || ideal.size() < point.size()) return std::nullopt;
    for (const auto& f : ideal) for (const auto& [e,c] : f) {
        for (size_t i = 0; i < point.size(); ++i) if (point[i] != 0 && abs(point[i]) != 1) {
            const auto bits = std::max(mpz_sizeinbase(point[i].get_num_mpz_t(), 2),
                                       mpz_sizeinbase(point[i].get_den_mpz_t(), 2));
            if (e[i] > 8192/bits) return std::nullopt;
        }
    }
    for (unsigned cap : {4U,8U}) {
        Basis jets;
        for (const auto& f : ideal) jets.push_back(point_jet_below(cap+1, f, point));
        auto rows = standard_basis(jets, names, lex, prefer_modular);
        std::vector<unsigned> powers(point.size());
        for (const auto& row : rows) {
            const auto h = local_head(row,lex);
            size_t i = 0;
            while (i < h.size() && h[i] == 0) ++i;
            if (i == h.size() || h[i] > cap ||
                std::any_of(h.begin()+i+1,h.end(),[](const auto& d){return d!=0;})) continue;
            const unsigned k = h[i].get_ui();
            if (!powers[i] || k < powers[i]) powers[i] = k;
        }
        unsigned cutoff = 1;
        for (unsigned k : powers) {
            if (!k) { cutoff = cap+1; break; }
            cutoff += k-1;
        }
        // The centered Taylor ideal contains m^cutoff when every monomial
        // in that degree is divisible by a pure-power head. As cutoff≤cap,
        // the jets agree with I mod m^(cap+1), so Nakayama gives m^cutoff
        // inside I itself. The jets and originals generate the same local
        // ideal, including scheme multiplicities.
        if (cutoff <= cap) {
            auto inverse = point;
            for (auto& q : inverse) q = -q;
            return translate_unbounded(rows,inverse);
        }
    }
    return std::nullopt;
}

// Weak Mora reduction supplies the standard-representation criterion for
// each S-pair. If its weak remainder has a new head, the centered initial
// ideal strictly increases. Dickson's lemma gives termination. All rows
// remain in the localized ideal, in the original coordinate symbols.
Basis sparse_point_standard_basis(const Basis& generators, const PointRow& point, bool lex) {
    Basis basis;
    std::vector<Powers> heads;
    using Pair = std::tuple<mpz_class,size_t,size_t>;
    std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>> pairs;
    auto insert = [&](Polynomial f) {
        if (f.empty()) return;
        // A new S-remainder can expose a separated unit factor even when
        // none of the input equations had one. Replacing this row by an
        // associate preserves the local ideal and its leading exponent.
        for (size_t i = 0; i < point.size(); ++i) simplify_separated_axis(f, i, point);
        auto [head,c] = leading_term_at_point(f, point, lex);
        for (size_t j = 0; j < basis.size(); ++j) {
            mpz_class degree_lcm = 0;
            bool disjoint = true;
            for (size_t k = 0; k < head.size(); ++k) {
                degree_lcm += std::max(head[k], heads[j][k]);
                if (head[k] != 0 && heads[j][k] != 0) disjoint = false;
            }
            if (!disjoint) pairs.emplace(degree_lcm,j,basis.size());
        }
        if (c != 1) for (auto& term : f) term.second /= c;
        heads.push_back(std::move(head));
        basis.push_back(std::move(f));
    };
    // Each input weak reduction preserves the generated ideal since its
    // certificate multiplies the input by a local unit.
    for (const auto& f : generators) insert(mora_remainder_at_point(basis, f, point, lex));
    std::map<Powers,Polynomial> shifts;
    PointRow inverse = point;
    for (auto& p : inverse) p = -p;
    auto shifted_multiple = [&](const Polynomial& f, const Powers& shift) {
        auto factor = shifts.find(shift);
        if (factor == shifts.end()) {
            auto polynomial = translate_unbounded({Polynomial{{shift,1}}}, inverse).front();
            factor = shifts.emplace(shift,std::move(polynomial)).first;
        }
        return primary::multiply(f,factor->second);
    };
    while (!pairs.empty()) {
        const auto [d,i,j] = pairs.top();
        pairs.pop();
        Powers left(point.size()), right(point.size());
        for (size_t k = 0; k < point.size(); ++k) {
            const auto& lcm = std::max(heads[i][k], heads[j][k]);
            left[k] = lcm-heads[i][k];
            right[k] = lcm-heads[j][k];
        }
        auto s = shifted_multiple(basis[i],left);
        for (const auto& [e,c] : shifted_multiple(basis[j],right)) {
            auto term = s.try_emplace(e,0).first;
            term->second -= c;
            if (term->second == 0) s.erase(term);
        }
        insert(mora_remainder_at_point(basis, s, point, lex));
    }
    Basis result;
    for (size_t j = 0; j < basis.size(); ++j) {
        bool minimal = true;
        for (size_t k = 0; k < heads.size(); ++k)
            if (k != j && divides(heads[k],heads[j]) && (heads[k] != heads[j] || k < j)) {
                minimal = false;
                break;
            }
        if (minimal) result.push_back(std::move(basis[j]));
    }
    std::sort(result.begin(),result.end());
    return result;
}

std::optional<Basis> coordinate_section(const Basis& ideal, const PointRow& point,
                                       const std::vector<std::string>& names, bool lex, bool prefer_modular) {
    for (size_t i = 0; i < names.size(); ++i) {
        Powers axis(names.size()), zero(names.size());
        axis[i] = 1;
        Polynomial equation{{axis,1}};
        if (point[i] != 0) equation[zero] = -point[i];
        if (std::find(ideal.begin(), ideal.end(), equation) == ideal.end()) continue;
        Basis projected;
        for (const auto& f : ideal) {
            auto row = specialize_coordinate(f, i, point[i]);
            if (!row.empty()) projected.push_back(std::move(row));
        }
        auto smaller_names = names;
        smaller_names.erase(smaller_names.begin()+i);
        auto center = point;
        center.erase(center.begin()+i);
        auto reduced = standard_basis_at_point(projected, smaller_names, center, lex, prefer_modular);
        Basis result{std::move(equation)};
        for (const auto& f : reduced) {
            Polynomial row;
            for (const auto& [e,c] : f) {
                auto powers = e;
                powers.insert(powers.begin()+i, 0);
                row.emplace(std::move(powers),c);
            }
            result.push_back(std::move(row));
        }
        std::sort(result.begin(), result.end());
        return result;
    }
    return std::nullopt;
}
}

Basis standard_basis_at_point(const Basis& ideal, const std::vector<std::string>& names,
                              const PointRow& point, bool lex, bool prefer_modular) {
    if (point.size() != names.size()) throw std::invalid_argument("Incorrect local point dimension.");
    if (ideal.empty()) return {};
    const Polynomial one{{Powers(names.size()), 1}};
    if (std::all_of(ideal.begin(), ideal.end(), [](const auto& f) { return f.size() <= 1; }) &&
        std::any_of(point.begin(), point.end(), [](const auto& q) { return q != 0; }))
        return project_monomials(ideal, point);
    if (use_sparse_point_basis(ideal, point)) {
        auto rows = strip_units(ideal, point);
        if (rows.empty()) return {};
        for (const auto& f : rows) if (nonzero_at_point(f, point)) return {one};
        if (auto section = coordinate_section(rows, point, names, lex, prefer_modular)) return *section;
        if (auto cover = point_power_coordinates(rows, point)) {
            // In centered variables the diagonal Jacobian is invertible.
            // Pullback rescales every monomial in a lowest form separately;
            // it preserves the leading exponent, including either tie order.
            auto basis = standard_basis_at_point(cover->generators, names, cover->point, lex, prefer_modular);
            Basis result;
            auto offsets = point;
            for (auto& q : offsets) q = -q;
            for (const auto& f : basis) {
                // A pure centered power in the compressed coordinates
                // pulls back to the same power of x_i-p_i times a unit.
                // Keeping the smaller polynomial also avoids a potentially
                // enormous scalar in its original-coordinate monic lift.
                if (!use_sparse_point_basis({f}, cover->point)) {
                    auto centered = translate_unbounded({f}, cover->point).front();
                    if (centered.size() == 1) {
                        const auto& h = centered.begin()->first;
                        result.push_back(translate_unbounded({Polynomial{{h,1}}}, offsets).front());
                        continue;
                    }
                }
                Polynomial row;
                for (const auto& [e,c] : f) {
                    auto powers = e;
                    for (size_t i = 0; i < powers.size(); ++i) powers[i] *= cover->degrees[i];
                    row.emplace(std::move(powers), c);
                }
                normalize_at_point(row, point, lex);
                result.push_back(std::move(row));
            }
            return result;
        }
        bool separated = false;
        for (auto& f : rows) for (size_t i=0; i<point.size(); ++i)
            separated |= simplify_separated_axis(f, i, point);
        if (separated) return standard_basis_at_point(rows, names, point, lex, prefer_modular);
        bool coprime = true;
        std::set<size_t> used;
        for (auto& f : rows) {
            auto [head, coefficient] = leading_term_at_point(f, point, lex);
            for (size_t i = 0; i < head.size(); ++i) if (head[i] != 0 && !used.insert(i).second) {
                coprime = false;
                break;
            }
            if (!coprime) break;
            if (coefficient != 1) for (auto& term : f) term.second /= coefficient;
        }
        // Buchberger's product criterion applies to the local standard
        // representation in the localization, including a principal ideal.
        if (coprime) return rows;
        if (auto model = independent_parameter_basis(rows,names,point,lex,prefer_modular)) return *model;
        if (auto finite = finite_jet_at_point(rows, names, point, lex, prefer_modular)) return *finite;
        return sparse_point_standard_basis(rows, point, lex);
    }
    auto inverse = point;
    for (auto& q : inverse) q = -q;
    auto rows = standard_basis(translate_unbounded(ideal, point), names, lex, prefer_modular);
    return translate_unbounded(rows, inverse);
}
}
