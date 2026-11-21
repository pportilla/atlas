// Contact integration for three quadratics (z-c_i)^2-D over a plane.
// The caller certifies a reduced SNC resolution of the seven base factors.
#pragma once
#include <gmpxx.h>
#include <json/json.h>
#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace atlas::quadratic_translates {
using J = Json::Value;
using Z = mpz_class;
using Q = mpq_class;
using Group = std::vector<int>;
using Factors = std::map<Z, Z>;
// All contact heights are doubled. A missing value is an identically zero
// root difference, including a pair that coincides after plane restriction.
using Height = std::optional<Z>;
using Matrix = std::array<std::array<Height, 6>, 6>;
constexpr std::array<std::pair<int,int>,3> pairs{{{0,1},{0,2},{1,2}}};

struct Axis {
    std::string name;
    Z nu;
    Z chi;
    // D, c0-c1, c0-c2, c1-c2, and the three differences squared minus 4D.
    std::array<Z,7> orders{};
};

inline Axis exceptional_axis(const J& row) {
    Axis axis{row["id"].asString(), Z(row["nu"].asString(), 10), Z(row["chi"].asString(), 10)};
    if (row["orders"].size() != axis.orders.size() || axis.nu <= 0)
        throw std::invalid_argument("Incomplete quadratic contact divisor.");
    for (Json::ArrayIndex i = 0; i < axis.orders.size(); ++i) {
        axis.orders[i] = Z(row["orders"][i].asString(), 10);
        if (axis.orders[i] <= 0) throw std::invalid_argument("Expected positive exceptional contact orders.");
    }
    return axis;
}

inline Height minimum(const Height& a, const Height& b) {
    if (!a) return b;
    if (!b) return a;
    return std::min(*a, *b);
}

// A sign choice matters only when the two summands have equal order
// and one of the two opposite-sign differences cancels to higher order.
inline int enhanced_pairs(const Axis& axis, int restriction = -1) {
    int mask = 0;
    for (int p = 0; p < 3; ++p) {
        if (restriction != 0 && restriction != 1+p && 2*axis.orders[1+p] == axis.orders[0] &&
            (restriction == 4+p || axis.orders[4+p] > axis.orders[0])) mask |= 1 << p;
    }
    return mask;
}

inline Matrix contact_matrix(const Axis& axis, int orientation, bool capped, int restriction = -1) {
    auto order = [&](int column, int scale) -> Height {
        if (column == restriction) return std::nullopt;
        return scale * axis.orders[column];
    };
    Matrix result{};
    const Height radical = order(0, 1);
    for (int i = 0; i < 6; ++i) result[i][i^1] = radical;
    for (size_t p = 0; p < pairs.size(); ++p) {
        const auto [a,b] = pairs[p];
        const Height centres = order(1 + p, 2);
        const Height low = minimum(centres, radical);
        const Height resultant = order(4 + p, 2);
        if (!low) throw std::invalid_argument("Several discriminant factors vanish identically on the restriction.");
        // (c_i-c_j-2 sqrt(D)) (c_i-c_j+2 sqrt(D)) has known valuation.
        // At most one of these two differences can have larger contact.
        const Height high = resultant ? Height(*resultant - *low) : std::nullopt;
        if ((high && *high < *low) || (high != low && centres != radical))
            throw std::invalid_argument("The pair valuations disagree with the quadratic resultant.");
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
            const Height h = capped ? low : i == j ? centres : i == ((orientation >> p)&1) ? high : low;
            result[2*a+i][2*b+j] = result[2*b+j][2*a+i] = h;
        }
    }
    return result;
}

// Each split has a coordinatewise least contact, so it applies on the
// whole positive valuation cone. Tests inside the loop reject inconsistent
// relative sign choices among the three quadratic resultants.
struct IncompatibleOrientation {};
template<class Visit>
void visit_tree(const std::vector<Matrix>& matrices, const Group& roots, Visit& visit) {
    if (roots.size() < 2) return;
    std::vector<Z> heights, multiplicities;
    for (const auto& matrix : matrices) {
        Height least;
        for (size_t i = 0; i < roots.size(); ++i)
            for (size_t j = 0; j < i; ++j)
                least = minimum(least, matrix[roots[i]][roots[j]]);
        if (!least) {
            if (matrices.size() == 1) return; // a repeated terminal root in a plane slice
            throw std::invalid_argument("Coincident roots over the generic discriminant complement.");
        }
        heights.push_back(*least);
        Z n = 0;
        for (const auto& d : matrix[roots.front()])
            n += d ? std::min(*least, *d) : *least;
        multiplicities.push_back(n);
    }
    auto deeper = [&](int a, int b) {
        for (size_t k = 0; k < matrices.size(); ++k) {
            const Height& d = matrices[k][a][b];
            if (!d || *d > heights[k]) return true;
        }
        return false;
    };
    std::vector<Group> children;
    for (int root : roots) {
        auto group = std::find_if(children.begin(), children.end(), [&](const auto& c) {
            return deeper(root, c.front());
        });
        if (group == children.end()) children.push_back({root});
        else group->push_back(root);
    }
    if (children.size() < 2) throw IncompatibleOrientation{};
    for (size_t i = 0; i < children.size(); ++i) for (size_t j = 0; j <= i; ++j)
        for (int a : children[i]) for (int b : children[j])
            if (a != b && deeper(a,b) != (i == j)) throw IncompatibleOrientation{};

    if (visit(roots, children, heights, multiplicities))
        for (const auto& child : children) visit_tree(matrices, child, visit);
}

inline const Group all_roots{0,1,2,3,4,5};

// Choose a sign for each of the three centre differences at common order.
// Every admissible choice must give the same integral on the stratum.
template<class Evaluate>
auto common_value(int varying, Evaluate evaluate) {
    using Value = decltype(evaluate(0));
    std::optional<Value> result;
    for (int mask = 0; mask < 8; ++mask) {
        if (mask & ~varying) continue;
        try {
            Value value = evaluate(mask);
            if (!result) result = std::move(value);
            else if (*result != value)
                throw std::invalid_argument("Different quadratic root labelings give different local contributions.");
        } catch (const IncompatibleOrientation&) {}
    }
    if (!result) throw std::invalid_argument("The quadratic contacts do not have a common labeling on the cone.");
    return std::move(*result);
}

// Linear denominators are primitive to make equality independent of their
// presentation. Their slopes are candidates of the integral, and their
// labels distinguish the base components supplying repeated linear factors.
using Linear = std::tuple<std::string,Z,Z>;
using Denominators = std::vector<Linear>;
using Terms = std::map<Denominators,Q>;

inline void add_term(Terms& sum, Denominators term, const Q& coefficient, int ball_size = 0) {
    if (ball_size) term.emplace_back("ball", ball_size, 1);
    std::sort(term.begin(), term.end());
    auto& value = sum[term];
    value += coefficient;
    if (value == 0) sum.erase(term);
}

inline Terms contact_integral(const std::vector<Axis>& axes, int orientation, bool capped) {
    std::vector<Matrix> matrices;
    for (const auto& axis : axes) matrices.push_back(contact_matrix(axis, orientation, capped));
    Terms result;
    auto split = [&](const Group& roots, const std::vector<Group>& children,
                     const std::vector<Z>& heights, const std::vector<Z>& multiplicities) {
        bool last_ball = capped;
        Denominators term;
        Q scale = 1;
        for (size_t k = 0; k < axes.size(); ++k) {
            const Z nu = 2*axes[k].nu + heights[k];
            const Z& n = multiplicities[k];
            last_ball &= heights[k] == axes[k].orders[0];
            if (n == 0) scale *= Q(2, nu);
            else {
                Z g = gcd(n, nu);
                term.emplace_back(axes[k].name, n/g, nu/g);
                scale *= Q(2, g);
            }
        }
        scale.canonicalize();
        add_term(result, term, -scale, roots.size());
        if (last_ball) {
            // No integral root beyond ord(D)/2; all valuations are constant inside this ball.
            add_term(result, term, scale);
            return false;
        }
        add_term(result, term, (1-static_cast<int>(children.size()))*scale);
        for (const auto& child : children) add_term(result, term, scale, child.size());
        return true;
    };
    visit_tree(matrices, all_roots, split);
    return result;
}

inline void integrate(Terms& total, const std::vector<Axis>& axes, const Z& chi) {
    if (chi == 0) return;
    const bool odd = std::any_of(axes.begin(), axes.end(), [](const Axis& axis) {
        return mpz_odd_p(axis.orders[0].get_mpz_t());
    });
    int varying = 0;
    for (const auto& axis : axes) varying |= enhanced_pairs(axis);
    // The monomial centre difference and radical must be comparable on
    // a base crossing. Otherwise a further zero of the resultant meets
    // that crossing and this single-cone formula does not apply.
    if (axes.size() == 2) for (size_t p = 1; p <= 3; ++p) {
        const Z a = 2*axes[0].orders[p] - axes[0].orders[0];
        const Z b = 2*axes[1].orders[p] - axes[1].orders[0];
        if (sgn(a)*sgn(b) < 0) throw std::invalid_argument("Incomparable radical and centre at a crossing.");
    }
    for (bool capped : {false,true}) {
        if (capped && !odd) continue;
        // If an angular exponent of D is odd, the even-order sublattice
        // has index two. The Kummer cover of the angular torus is a single
        // torus; dividing by the two lifts gives the other factor of two.
        // With even exponents the angular cover descends from an étale
        // double cover of the stratum, of Euler characteristic twice its own.
        const Q weight = odd ? (capped ? Q(3,4) : Q(1,4)) : Q(1);
        const Terms terms = common_value(capped ? 0 : varying, [&](int mask) {
            return contact_integral(axes, mask, capped);
        });
        for (const auto& [term, coefficient] : terms) add_term(total, term, weight * chi * coefficient);
    }
}

inline J encode_terms(const Terms& terms) {
    J data(Json::arrayValue), strata(Json::arrayValue);
    std::map<Linear,std::string> labels;
    auto register_factor = [&](const Linear& factor, const Z& scale) -> const std::string& {
        const auto& [name,n,nu] = factor;
        Linear scaled{name, n*scale, nu*scale};
        auto it = labels.find(scaled);
        if (it != labels.end()) return it->second;
        std::string label = name + " (" + std::get<1>(scaled).get_str() + "," + std::get<2>(scaled).get_str() + ")";
        J row;
        row["component"] = label;
        row["n"] = std::get<1>(scaled).get_str();
        row["nu"] = std::get<2>(scaled).get_str();
        if (scale != 1 || (name == "ball" && n != 1)) row["auxiliary"] = true;
        data.append(std::move(row));
        return labels.emplace(std::move(scaled), std::move(label)).first->second;
    };
    for (const auto& [term, coefficient] : terms) {
        if (term.empty()) throw std::invalid_argument("Quadratic integral has an unexpected constant term.");
        J factors(Json::arrayValue), row(Json::arrayValue);
        for (size_t k = 0; k < term.size(); ++k) {
            // Clear the rational weight by scaling one denominator. Keep
            // its primitive factor available once for candidate-pole reporting.
            const auto& primitive = register_factor(term[k], 1);
            if (k == 0 && coefficient.get_den() != 1)
                factors.append(register_factor(term[k], coefficient.get_den()));
            else factors.append(primitive);
        }
        row.append(factors);
        row.append(coefficient.get_num().get_str());
        strata.append(std::move(row));
    }
    J result;
    result["data"] = std::move(data);
    result["strata"] = std::move(strata);
    return result;
}

inline Factors transverse_factors(const Axis& axis, int restriction = -1) {
    // Restricting to a strict branch sets that factor identically zero.
    // A missing contact in the resulting plane slice keeps coincident
    // roots together with their original multiplicity.
    const bool ramified = restriction != 0 && mpz_odd_p(axis.orders[0].get_mpz_t());
    return common_value(enhanced_pairs(axis, restriction), [&](int mask) {
        const std::vector<Matrix> matrices{contact_matrix(axis, mask, false, restriction)};
        Factors factors{{6,1}};
        auto split = [&](const Group& roots, const std::vector<Group>& children,
                         const std::vector<Z>& heights, const std::vector<Z>& multiplicities) {
            const Z& n = multiplicities.front();
            Group conjugate = roots;
            if (ramified) {
                for (int& root : conjugate) root ^= 1;
                std::sort(conjugate.begin(), conjugate.end());
            }
            if (conjugate != roots) {
                // The pair of conjugate clusters descends to one divisor.
                if (conjugate < roots) return false;
                factors[n] += 1-static_cast<int>(children.size());
            } else if (mpz_odd_p(heights.front().get_mpz_t())) {
                // Resolve the first half-integral split (a square minus a
                // local parameter). It contributes (1-t^(n/2))/(1-t^n).
                if (!ramified || children.size() != 2 || mpz_odd_p(n.get_mpz_t()))
                    throw std::invalid_argument("Unexpected ramified Puiseux split.");
                ++factors[n/2];
                --factors[n];
            } else {
                if (mpz_odd_p(n.get_mpz_t()))
                    throw std::invalid_argument("Fractional multiplicity on an invariant cluster.");
                factors[n/2] += 1-static_cast<int>(children.size());
            }
            return true;
        };
        visit_tree(matrices, all_roots, split);
        for (auto it = factors.begin(); it != factors.end();) {
            if (it->second == 0) it = factors.erase(it);
            else ++it;
        }
        return factors;
    });
}

struct LocalMonodromy {
    std::string locus;
    Z power;
    Factors factors;
};
struct Integral {
    Terms terms;
    std::vector<LocalMonodromy> local;

    J monodromy_terms() const {
        J rows(Json::arrayValue);
        for (const auto& component : local) {
            if (component.power == 0) continue;
            for (const auto& [n,e] : component.factors) {
                J row;
                row["component"] = component.locus;
                row["multiplicity"] = n.get_str();
                const Z exponent = component.power*e;
                row["euler_characteristic"] = exponent.get_str();
                rows.append(std::move(row));
            }
        }
        return rows;
    }
};

inline Integral evaluate(const J& base) {
    std::map<std::string,Axis> divisors;
    Integral integral;
    add_term(integral.terms, {}, 1, 6);
    for (const auto& row : base["divisors"]) {
        auto axis = exceptional_axis(row);
        integrate(integral.terms, {axis}, axis.chi);
        integral.local.push_back({axis.name+" open", axis.chi, transverse_factors(axis)});
        // Lifting a base blowup adds six exceptional lines cyclically
        // permuted on a nearby fibre. Remove their contribution to descend.
        integral.local.push_back({axis.name+" lifted blowup", -1, {{6,1}}});
        divisors.emplace(axis.name, std::move(axis));
    }
    for (const auto& edge : base["edges"]) {
        const auto& a = divisors.at(edge[0].asString());
        const auto& b = divisors.at(edge[1].asString());
        integrate(integral.terms, {a,b}, 1);
        // Both entries of the least quasi-ordinary contact are positive.
        integral.local.push_back({a.name+" / "+b.name, 1, {{6,1}}});
    }
    for (const auto& branch : base["branches"]) {
        const auto& a = divisors.at(branch["divisor"].asString());
        const int column = branch["pair"].asInt();
        if (column < 0 || column >= 7) throw std::invalid_argument("Unknown discriminant branch.");
        Axis strict{"strict "+std::to_string(column),1,1};
        strict.orders[column] = 1;
        integrate(integral.terms, {a,strict}, 1);
        // Here the least contact is positive only on the exceptional
        // coordinate. The quasi-ordinary zeta is that of the plane restriction.
        integral.local.push_back({a.name+" / discriminant branch "+std::to_string(column), 1,
                                  transverse_factors(a,column)});
    }
    return integral;
}
} // namespace atlas::quadratic_translates

namespace atlas {
inline Json::Value quadratic_translate_zetas(const Json::Value& base) {
    auto integral = quadratic_translates::evaluate(base);
    auto result = quadratic_translates::encode_terms(integral.terms);
    result["monodromy"] = integral.monodromy_terms();
    return result;
}
}
