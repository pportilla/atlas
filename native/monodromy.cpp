// Exact arithmetic for A'Campo's formula and nearby monodromy witnesses.
#include "rational_sparse_json.hpp"
#include "monodromy_local_b.hpp"
#include "monodromy_transverse.hpp"
#include "primary_decomposition.hpp"
#include "groebner.hpp"
#include <flint/fmpq.h>
#include <flint/fmpq_mat.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>

namespace atlas { Json::Value arrangement_nearby_strata(const local::Polynomial& f); }

namespace {
using namespace atlas::sparse_json;
namespace primary = atlas::primary;

Z integer(const Json::Value& v) { return Z(v.asString(), 10); }
Q rational(const Json::Value& v) {
    Q q(v.asString(), 10);
    if (q.get_den() == 0) throw std::invalid_argument("Zero denominator in monodromy data.");
    q.canonicalize();
    return q;
}
using Factors = std::map<Z, Z>;
Json::Value factor_list(const Factors& powers) {
    Json::Value result(Json::arrayValue);
    for (const auto& [n, e] : powers) {
        if (!e) continue;
        Json::Value pair(Json::arrayValue);
        pair.append(n.get_str());
        pair.append(e.get_str());
        result.append(pair);
    }
    return result;
}
Json::Value acampo(const Json::Value& rows) {
    Factors powers;
    Z chi = 0;
    for (const auto& row : rows) {
        const Z n = integer(row[0]), euler = integer(row[1]);
        if (n <= 0) throw std::invalid_argument("Expected a positive divisor multiplicity.");
        powers[n] -= euler;
        chi += n * euler;
    }
    Json::Value result;
    result["factors"] = factor_list(powers);
    result["chi"] = chi.get_str();
    return result;
}
Z cyclotomic(const Json::Value& factors, const Z& order) {
    if (order == 0) throw std::invalid_argument("Expected a nonzero cyclotomic order.");
    Z exponent = 0;
    for (const auto& row : factors) if (integer(row[0]) % order == 0) exponent += integer(row[1]);
    return exponent;
}

// Only open singleton strata contribute to A'Campo's formula.
Json::Value acampo_strata(const Json::Value& request) {
    std::map<std::string, std::string> orders;
    for (const auto& row : request["data"])
        orders[row[0].asString()] = row[1].asString();
    Json::Value inputs(Json::arrayValue), contributions(Json::arrayValue);
    for (const auto& stratum : request["strata"]) {
        const auto& labels = stratum[0];
        if (labels.size() != 1) continue;
        const auto label = labels[0].asString();
        Json::Value pair(Json::arrayValue);
        pair.append(orders.at(label));
        pair.append(stratum[1]);
        inputs.append(pair);
        Json::Value row;
        row["component"] = label;
        row["multiplicity"] = pair[0];
        row["euler_characteristic"] = pair[1];
        contributions.append(row);
    }
    auto result = acampo(inputs);
    result["contributions"] = std::move(contributions);
    return result;
}
using Axis = std::pair<size_t, Z>;
using Axes = std::vector<size_t>;
struct CoordinateStrata {
    Json::Value rows{Json::arrayValue};
    std::set<Axes> loci;

    void add(Axes axes, Json::Value row) {
        for (size_t axis : axes) row["axes"].append(Json::UInt64(axis));
        std::sort(axes.begin(), axes.end());
        if (loci.insert(std::move(axes)).second) rows.append(std::move(row));
    }
};

void binomial_strata(const Terms& terms, size_t width, CoordinateStrata& result) {
    if (terms.size() != 2 || width < 3) return;
    const auto& a = terms.rbegin()->first;
    const auto& b = terms.begin()->first;
    std::vector<Axis> left, right;
    for (size_t i = 0; i < width; ++i) {
        if (a[i] && !b[i]) left.emplace_back(i, a[i]);
        if (b[i] && !a[i]) right.emplace_back(i, b[i]);
    }
    auto add = [&](const char* model, const std::vector<Axis>& normals, const Factors& factors) {
        Json::Value item;
        item["model"] = model;
        item["factors"] = factor_list(factors);
        Axes axes;
        for (const auto& [axis, power] : normals) {
            axes.push_back(axis);
            item["powers"].append(power.get_str());
        }
        result.add(std::move(axes), std::move(item));
    };
    // On the coordinate torus all other variables are units. Taking local
    // roots of the units leaves the indicated transverse Brieskorn germ.
    for (const auto& x : left) for (const auto& y : right) {
        if (x.second == 1 || y.second == 1) continue; // Smooth, covered by the branch.
        Factors factors;
        factors[lcm(x.second, y.second)] = gcd(x.second, y.second);
        --factors[x.second];
        --factors[y.second];
        add("X^a + Y^b", {x,y}, factors);
    }
    if (width >= 4) for (int swap = 0; swap < 2; ++swap) {
        const auto& product = swap ? right : left;
        const auto& powers = swap ? left : right;
        for (size_t i = 0; i < product.size(); ++i) {
            if (product[i].second != 1) continue;
            for (size_t j = i + 1; j < product.size(); ++j) {
                if (product[j].second != 1) continue;
                // A double suspension shifts reduced cohomology by two,
                // retaining the zeta of Z^m.
                for (const auto& z : powers) if (z.second > 1)
                    add("X*Y + Z^m", {product[i],product[j],z}, {{z.second,-1}});
            }
        }
    }
}
Q transverse_determinant(const Terms& terms, const std::vector<size_t>& normal) {
    const size_t n = normal.size();
    Matrix hessian(n, Row(n));
    for (const auto& [e, c] : terms) {
        Z degree = 0;
        size_t a = n, b = n;
        for (size_t j = 0; j < n; ++j) if (e[normal[j]]) {
            degree += e[normal[j]];
            if (a == n) a = j;
            else b = j;
        }
        if (degree < 2) return 0; // The whole coordinate stratum must be critical.
        if (degree == 2) {
            if (b == n) hessian[a][a] += 2*c;
            else { hessian[a][b] += c; hessian[b][a] += c; }
        }
    }
    fmpq_mat_t m;
    fmpq_mat_init(m, n, n);
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j)
        fmpq_set_mpq(fmpq_mat_entry(m, i, j), hessian[i][j].get_mpq_t());
    fmpq_t det;
    fmpq_init(det);
    fmpq_mat_det(det, m);
    Q result;
    fmpq_get_mpq(result.get_mpq_t(), det);
    fmpq_clear(det);
    fmpq_mat_clear(m);
    return result;
}
template<class Visitor>
void visit_morse(const Terms& terms, const Axes& active, size_t width, bool odd_only, Visitor visit) {
    // An unused variable would give a zero Hessian row. Bound enumeration
    // by the quick solver's eight active variables, including on cylinders.
    if (active.size() > 8) return;
    for (size_t k = 1; k <= active.size() && k < width; k += odd_only ? 2 : 1) {
        std::vector<bool> selected(active.size());
        std::fill_n(selected.begin(), k, true);
        do {
            Axes axes;
            for (size_t i = 0; i < active.size(); ++i) if (selected[i]) axes.push_back(active[i]);
            const Q determinant = transverse_determinant(terms, axes);
            if (determinant && visit(axes, determinant)) return;
        } while (std::prev_permutation(selected.begin(), selected.end()));
    }
}
bool strict_component(const std::string& label) {
    return label == "strict" || (label.size() > 1 && label.front() == 'S' &&
           label.find_first_not_of("0123456789", 1) == std::string::npos);
}
// If the residual factor is a unit at zero, these are exactly the local
// branches and their orders in the original coordinates.
Powers coordinate_divisor(const Terms& terms, size_t width) {
    if (terms.empty()) return {};
    auto powers = terms.begin()->first;
    for (const auto& [e, c] : terms) for (size_t i = 0; i < width; ++i)
        powers[i] = std::min(powers[i], e[i]);
    return terms.count(powers) && powers != Powers(width) ? powers : Powers{};
}
std::set<std::string> visible_components(const Json::Value& strata) {
    std::set<std::string> visible;
    for (const auto& stratum : strata) {
        if (integer(stratum[1]) == 0) continue;
        for (const auto& name : stratum[0]) visible.insert(name.asString());
    }
    return visible;
}
Json::Value nearby_strata(const Terms& terms, size_t width, const Json::Value& r) {
    CoordinateStrata result;
    if (width < 2 || terms.empty()) return result.rows;
    std::vector<bool> used(width), critical_line(width, true);
    for (const auto& [powers, coefficient] : terms) {
        size_t first = width, second = width, support_size = 0;
        for (size_t i = 0; i < width; ++i) if (powers[i]) {
            used[i] = true;
            if (support_size == 0) first = i;
            if (support_size == 1) second = i;
            ++support_size;
        }
        if (support_size == 0 || (support_size == 1 && powers[first] == 1)) return result.rows;
        // The i-axis is critical when every term has degree >= 2 in the
        // other coordinates. Only terms on one or two axes can obstruct it.
        if (support_size == 1) critical_line[first] = false;
        if (support_size == 2) {
            if (powers[first] == 1) critical_line[second] = false;
            if (powers[second] == 1) critical_line[first] = false;
        }
    }
    const bool has_critical_axis = std::find(critical_line.begin(), critical_line.end(), true) != critical_line.end();
    // The closure of a positive-dimensional critical coordinate stratum
    // contains an entire coordinate axis through the origin.
    if (has_critical_axis) {
        Axes active;
        for (size_t i = 0; i < width; ++i) if (used[i]) active.push_back(i);
        binomial_strata(terms, width, result);
        visit_morse(terms, active, width, false, [&](const Axes& axes, const Q& determinant) {
            // The critical coordinate space and nonsingular Hessian at parameters
            // equal to one certify a dense open with this transverse quadratic form.
            // The vanishing sphere has dimension k-1 and antipodal degree (-1)^k.
            Json::Value row;
            row["model"] = "sum of squares";
            row["determinant"] = determinant.get_str();
            row["factors"] = axes.size() % 2 ? factor_list({{2,-1}}) : Json::Value(Json::arrayValue);
            result.add(axes, std::move(row));
            return false;
        });
        if (active.size() < width && r.isMember("origin_factors")) {
            // Translation in any omitted coordinate preserves the entire germ.
            Json::Value row;
            row["model"] = "constant transverse type";
            row["factors"] = r["origin_factors"];
            result.add(active, std::move(row));
        }
    }
    // A visible strict divisor has generic transverse equation U^N. N>1
    // certifies a nonisolated critical locus. Two distinct branches through
    // zero intersect in positive dimension if the ambient dimension is >= 3.
    const auto visible = visible_components(r["strata"]);
    Json::Value branches(Json::arrayValue);
    const auto coordinate_orders = coordinate_divisor(terms, width);
    std::map<Z, Axes> coordinate_axes;
    for (size_t i = 0; i < coordinate_orders.size(); ++i) if (coordinate_orders[i] != 0)
        coordinate_axes[coordinate_orders[i]].push_back(i);
    std::map<Z, size_t> visible_counts;
    bool nonreduced = false;
    for (const auto& divisor : r["data"]) {
        const auto label = divisor["component"].asString();
        if (!strict_component(label) || !visible.count(label)) continue;
        const Z n = integer(divisor["n"]);
        if (n <= 0) throw std::invalid_argument("Expected a positive divisor multiplicity.");
        nonreduced |= n > 1;
        Json::Value row;
        row["model"] = "smooth component";
        row["component"] = label;
        row["multiplicity"] = n.get_str();
        row["factors"] = factor_list({{n,-1}});
        ++visible_counts[n];
        const auto found = coordinate_axes.find(n);
        if (found != coordinate_axes.end() && found->second.size() == 1)
            row["axes"].append(Json::UInt64(found->second.front()));
        branches.append(std::move(row));
    }
    if (has_critical_axis || nonreduced || (width >= 3 && branches.size() >= 2)) {
        // For a coordinate divisor every branch is one hyperplane. With all
        // branches of an order visible, their rows cover these axes even if
        // their individual names cannot be identified by multiplicity alone.
        std::set<size_t> covered_axes;
        for (const auto& [order, axes] : coordinate_axes)
            if (visible_counts[order] == axes.size()) covered_axes.insert(axes.begin(), axes.end());
        Json::Value rows(Json::arrayValue);
        for (const auto& row : result.rows) {
            if (row["axes"].size() == 1 && covered_axes.count(row["axes"][0].asUInt64())) continue;
            rows.append(row);
        }
        for (const auto& row : branches) rows.append(row);
        return rows;
    }
    return result.rows;
}
void add_term(Terms& f, const Powers& e, const Q& c) {
    if (c == 0) return;
    auto [it, inserted] = f.emplace(e, c);
    if (!inserted && (it->second += c) == 0) f.erase(it);
}
Terms derivative(const Terms& f, size_t axis) {
    Terms result;
    for (const auto& [e, c] : f) {
        if (e[axis] == 0) continue;
        Powers lowered = e;
        lowered[axis] -= 1;
        add_term(result, lowered, c * Q(e[axis]));
    }
    return result;
}
// Fully reduce f by a lex Gröbner basis. The map order is the lex order.
Terms lex_remainder(Terms f, const primary::Basis& basis, size_t& steps) {
    Terms remainder;
    while (!f.empty()) {
        if (steps == 0) throw std::runtime_error("Transverse composition division limit.");
        --steps;
        const auto top = std::prev(f.end());
        const Powers head = top->first;
        const Q coefficient = top->second;
        const auto divisor = std::find_if(basis.begin(), basis.end(), [&](const auto& g) {
            const auto& lead = g.rbegin()->first;
            for (size_t i = 0; i < lead.size(); ++i) if (lead[i] > head[i]) return false;
            return true;
        });
        if (divisor == basis.end()) {
            remainder.emplace(head, coefficient);
            f.erase(top);
            continue;
        }
        const auto& [lead, lc] = *divisor->rbegin();
        const Q scale = -coefficient / lc;
        for (const auto& [e, c] : *divisor) {
            Powers shifted = e;
            for (size_t i = 0; i < e.size(); ++i) shifted[i] += head[i] - lead[i];
            add_term(f, shifted, c * scale);
        }
    }
    return remainder;
}
// With x before u in lex order, f lies in QQ[h] exactly when its remainder
// modulo (h_i(x) - u_i) involves only u. The remainder G then has f = G(h).
std::optional<Terms> composition_factor(const Terms& f, const primary::Basis& h, size_t n, size_t& steps) {
    const size_t k = h.size(), width = n + k;
    auto lift = [&](const Terms& p) {
        Terms out;
        for (const auto& [e, c] : p) {
            Powers wide = e;
            wide.resize(width);
            out.emplace(std::move(wide), c);
        }
        return out;
    };
    primary::Basis graph;
    for (size_t i = 0; i < k; ++i) {
        Terms row = lift(h[i]);
        Powers u(width);
        u[n + i] = 1;
        add_term(row, u, -1);
        graph.push_back(std::move(row));
    }
    Terms germ;
    for (const auto& [e, c] : lex_remainder(lift(f), primary::reduced(graph, width, true), steps)) {
        if (std::any_of(e.begin(), e.begin() + n, [](const Z& a) { return a != 0; })) return std::nullopt;
        germ.emplace(Powers(e.begin() + n, e.end()), c);
    }
    return germ;
}
Terms substitute_all(const Terms& germ, const primary::Basis& h, size_t n) {
    std::vector<std::map<Z, Terms>> powers(h.size());
    Terms result;
    for (const auto& [a, c] : germ) {
        Terms term{{Powers(n), c}};
        for (size_t i = 0; i < h.size(); ++i) {
            if (a[i] == 0) continue;
            auto it = powers[i].find(a[i]);
            if (it == powers[i].end()) it = powers[i].emplace(a[i], primary::power(h[i], a[i], n)).first;
            term = primary::multiply(term, it->second);
        }
        for (const auto& [e, v] : term) add_term(result, e, v);
    }
    return result;
}
// All k x k minors of the Jacobian matrix of h, by Laplace expansion along
// its rows with the remaining columns memoized as a bit mask.
primary::Basis maximal_minors(const primary::Basis& h, size_t n) {
    const size_t k = h.size();
    std::vector<std::vector<Terms>> jacobian(k);
    for (size_t i = 0; i < k; ++i) for (size_t j = 0; j < n; ++j) jacobian[i].push_back(derivative(h[i], j));
    std::map<std::uint32_t, Terms> memo;
    std::function<Terms(std::uint32_t)> minor = [&](std::uint32_t columns) -> Terms {
        const size_t row = k - static_cast<size_t>(__builtin_popcount(columns));
        if (row == k) return Terms{{Powers(n), Q(1)}};
        if (const auto found = memo.find(columns); found != memo.end()) return found->second;
        Terms total;
        int sign = 1;
        for (size_t j = 0; j < n; ++j) {
            if (!(columns >> j & 1u)) continue;
            if (!jacobian[row][j].empty()) {
                const auto rest = minor(columns & ~(1u << j));
                if (!rest.empty())
                    for (const auto& [e, c] : primary::multiply(jacobian[row][j], rest)) add_term(total, e, sign * c);
            }
            sign = -sign;
        }
        return memo[columns] = total;
    };
    primary::Basis minors;
    for (std::uint32_t columns = 0; columns < (1u << n); ++columns) {
        if (static_cast<size_t>(__builtin_popcount(columns)) != k) continue;
        auto value = minor(columns);
        if (!value.empty()) minors.push_back(std::move(value));
    }
    return minors;
}
// Replace u_axis by image in a polynomial of the same width.
Terms substitute_variable(const Terms& g, size_t axis, const Terms& image, size_t width) {
    std::map<Z, Terms> powers;
    Terms result;
    for (const auto& [e, c] : g) {
        auto it = powers.find(e[axis]);
        if (it == powers.end()) it = powers.emplace(e[axis], primary::power(image, e[axis], width)).first;
        for (const auto& [v, d] : it->second) {
            Powers shifted = e;
            shifted[axis] = 0;
            for (size_t j = 0; j < width; ++j) shifted[j] += v[j];
            add_term(result, shifted, c * d);
        }
    }
    return result;
}
// If G = a u_i^m + b(u') u_i^(m-1) + ... with constant a and b(0) = 0, the
// triangular automorphism u_i -> u_i - b/(m a) removes the u_i^(m-1) term.
// The submersion changes by its inverse, h_i -> h_i + b(h)/(m a), which
// keeps h(0) = 0, the zero set of h and the rank of dh.
void tschirnhausen(Terms& germ, primary::Basis& h, size_t n) {
    const size_t k = h.size();
    for (size_t pass = 0; pass < k; ++pass) {
        bool changed = false;
        for (size_t i = 0; i < k; ++i) {
            Z m = 0;
            for (const auto& [e, c] : germ) m = std::max(m, e[i]);
            if (m < 2) continue;
            Powers pure(k);
            pure[i] = m;
            const auto lead = germ.find(pure);
            if (lead == germ.end()) continue;
            Terms shift;
            bool monic = true;
            for (const auto& [e, c] : germ) {
                if (e[i] == m && e != pure) monic = false;
                if (e[i] != m - 1) continue;
                Powers rest = e;
                rest[i] = 0;
                add_term(shift, rest, c / (Q(m) * lead->second));
            }
            if (!monic || shift.empty() || shift.count(Powers(k))) continue;
            Terms image = shift;
            for (auto& [e, c] : image) c = -c;
            Powers axis(k);
            axis[i] = 1;
            add_term(image, axis, 1);
            germ = substitute_variable(germ, i, image, k);
            for (const auto& [e, c] : substitute_all(shift, h, n)) add_term(h[i], e, c);
            changed = true;
        }
        if (!changed) return;
    }
}
// The Milnor fibre of a Brieskorn-Pham germ sum c_i u_i^a_i is the join of
// the fibres of u_i^a_i. Writing L_a for the divisor of 1-T^a, the reduced
// zeta of a join is (-1)^k times the product of the divisors L_a_i - L_1,
// with L_a * L_b = gcd(a,b) L_lcm(a,b). Removing H^0 subtracts L_1.
std::optional<Factors> brieskorn_pham_zeta(const Terms& germ, size_t k) {
    if (germ.size() != k) return std::nullopt;
    std::vector<Z> powers(k, 0);
    for (const auto& [e, c] : germ) {
        size_t axis = k;
        for (size_t i = 0; i < k; ++i) if (e[i] != 0) {
            if (axis != k) return std::nullopt;
            axis = i;
        }
        if (axis == k || powers[axis] != 0) return std::nullopt;
        powers[axis] = e[axis];
    }
    if (k > 16) return std::nullopt;
    Factors factors;
    for (std::uint32_t subset = 0; subset < (1u << k); ++subset) {
        Z order = 1, product = 1;
        for (size_t i = 0; i < k; ++i) if (subset >> i & 1u) {
            order = lcm(order, powers[i]);
            product *= powers[i];
        }
        const Z weight = product / order;
        factors[order] += __builtin_popcount(subset) % 2 ? -weight : weight;
    }
    factors[1] -= 1;
    return factors;
}
bool coordinate_function(const Terms& h) {
    if (h.size() != 1) return false;
    const auto& e = h.begin()->first;
    return std::accumulate(e.begin(), e.end(), Z(0)) == 1;
}
// Every component of V(h) through 0 has dimension at least n-k. If the
// locus where rank dh < k meets V(h) in dimension below n-k, the points of
// V(h) where (h, t) is a local coordinate system accumulate at 0.
bool submersion_accumulates(const primary::Basis& h, size_t n, int dim) {
    if (std::any_of(h.begin(), h.end(), [&](const Terms& g) { return g.count(Powers(n)) != 0; })) return false;
    auto certificate = h;
    for (auto& m : maximal_minors(h, n)) certificate.push_back(std::move(m));
    return primary::dimension_lex(primary::reduced(certificate, n, true), n) < dim;
}
// Scale h_i to a primitive integer polynomial with positive lex-leading
// coefficient, and G by the inverse substitution u_i -> u_i / lambda.
void primitive_coordinates(Terms& germ, primary::Basis& h) {
    for (size_t i = 0; i < h.size(); ++i) {
        Z numerators = 0, denominators = 1;
        for (const auto& [e, c] : h[i]) {
            numerators = gcd(numerators, Z(c.get_num()));
            denominators = lcm(denominators, Z(c.get_den()));
        }
        Q lambda(denominators, numerators);
        if (h[i].rbegin()->second < 0) lambda = -lambda;
        lambda.canonicalize();
        if (lambda == 1) continue;
        for (auto& [e, c] : h[i]) c *= lambda;
        for (auto& [a, c] : germ) {
            Q scale = 1;
            for (Z j = 0; j < a[i]; ++j) scale /= lambda;
            c *= scale;
        }
    }
}
// Certify f = G(h) for k polynomials h on which the submersion condition
// holds. There the germ of f is G times a smooth factor, and its local zeta
// is that of G at the origin. The normalizations are triangular or diagonal
// automorphisms of the target, and the returned pair is checked again.
std::optional<std::pair<Terms, primary::Basis>> certified_composition(
        const Terms& f, primary::Basis submersion, size_t n, int dim) {
    try {
        primary::CertificateBudget budget(1u << 18);
        size_t steps = 1u << 16;
        if (!submersion_accumulates(submersion, n, dim)) return std::nullopt;
        auto germ = composition_factor(f, submersion, n, steps);
        if (!germ || germ->empty()) return std::nullopt;
        tschirnhausen(*germ, submersion, n);
        primitive_coordinates(*germ, submersion);
        if (substitute_all(*germ, submersion, n) != f || !submersion_accumulates(submersion, n, dim))
            return std::nullopt;
        return std::make_pair(std::move(*germ), std::move(submersion));
    } catch (const std::runtime_error&) {
        return std::nullopt;  // This candidate exceeded its work limit.
    }
}
Terms lex_monic(Terms g) {
    const Q lead = g.rbegin()->second;
    for (auto& [e, c] : g) c /= lead;
    return g;
}
std::pair<Z, size_t> complexity(const Terms& g) {
    Z degree = 0;
    for (const auto& [e, c] : g) degree = std::max(degree, std::accumulate(e.begin(), e.end(), Z(0)));
    return {degree, g.size()};
}
// The k-subsets of the pool, simplest first. The generators of the prime
// come first and keep their position when they number exactly k.
std::vector<std::vector<size_t>> candidate_subsets(const primary::Basis& pool, size_t k, size_t generators) {
    std::vector<std::vector<size_t>> subsets;
    std::vector<size_t> chosen(k);
    std::iota(chosen.begin(), chosen.end(), 0);
    while (true) {
        subsets.push_back(chosen);
        size_t i = k;
        while (i > 0 && chosen[i - 1] == pool.size() - k + i - 1) --i;
        if (i == 0) break;
        ++chosen[i - 1];
        for (size_t j = i; j < k; ++j) chosen[j] = chosen[j - 1] + 1;
    }
    auto key = [&](const std::vector<size_t>& subset) {
        Z degree = 0;
        size_t terms = 0;
        for (size_t i : subset) {
            const auto [d, t] = complexity(pool[i]);
            degree += d;
            terms += t;
        }
        return std::make_tuple(!(generators == k && subset.back() < k), degree, terms, subset);
    };
    std::sort(subsets.begin(), subsets.end(), [&](const auto& a, const auto& b) { return key(a) < key(b); });
    return subsets;
}
// Search the critical components through the origin for a factorization
// f = G(h_1,...,h_k) through k polynomials vanishing on a component of
// codimension k. Candidates are the reduced generators of its prime and the
// irreducible factors of f and its partial derivatives on the component.
Json::Value transverse_compositions(const Terms& f, size_t n) {
    Json::Value rows(Json::arrayValue);
    // Bound the search like the coordinate Morse search: at most eight
    // variables, few terms and moderate degree, so wide exponents skip it.
    // The decomposition of the critical ideal uses the ordinary exact basis,
    // whose coefficients may grow beyond the certificate pass's limit.
    if (n < 3 || n > 8 || f.size() < 2 || f.size() > 128) return rows;
    for (const auto& [e, c] : f)
        if (std::accumulate(e.begin(), e.end(), Z(0)) > 32) return rows;
    std::set<std::pair<primary::Basis, Terms>> emitted;
    try {
        primary::Basis critical{f};
        for (size_t i = 0; i < n; ++i)
            if (auto d = derivative(f, i); !d.empty()) critical.push_back(std::move(d));
        // Keep the minimal primes, which are the components of the critical locus.
        std::vector<std::pair<primary::Basis, primary::Basis>> primes;
        for (const auto& part : primary::decompose(critical, n, false))
            if (std::none_of(primes.begin(), primes.end(), [&](const auto& p) { return p.first == part.prime; }))
                primes.emplace_back(part.prime, primary::reduced(part.prime, n, true));
        auto contains = [](const primary::Basis& lex_basis, const primary::Basis& generators) {
            return std::all_of(generators.begin(), generators.end(),
                               [&](const Terms& g) { return primary::member_lex(g, lex_basis); });
        };
        for (const auto& [h, lex_prime] : primes) {
            if (std::any_of(primes.begin(), primes.end(), [&](const auto& other) {
                    return other.first != h && contains(lex_prime, other.first); })) continue;
            if (std::any_of(h.begin(), h.end(), [&](const Terms& g) { return g.count(Powers(n)) != 0; })) continue;
            const int dim = primary::dimension_lex(lex_prime, n);
            if (dim < 1 || static_cast<size_t>(dim) + 2 > n) continue;
            const size_t k = n - static_cast<size_t>(dim);
            primary::Basis pool, factors;
            std::set<Terms> normalized;
            for (const auto& g : h)
                if (normalized.insert(lex_monic(g)).second) pool.push_back(g);
            const size_t generators = pool.size();
            for (const auto& g : critical)
                for (const auto& [factor, power] : primary::factors(g, n))
                    if (normalized.insert(lex_monic(factor)).second && primary::member_lex(factor, lex_prime))
                        factors.push_back(factor);
            std::stable_sort(factors.begin(), factors.end(),
                             [](const Terms& a, const Terms& b) { return complexity(a) < complexity(b); });
            for (auto& g : factors) if (pool.size() < 10) pool.push_back(std::move(g));
            if (pool.size() < k) continue;
            const auto subsets = candidate_subsets(pool, k, generators);
            std::optional<std::pair<Terms, primary::Basis>> found;
            for (size_t i = 0; i < subsets.size() && i < 64 && !found; ++i) {
                primary::Basis submersion;
                for (size_t j : subsets[i]) submersion.push_back(pool[j]);
                // Coordinate functions alone give the translation-invariant model.
                if (!std::all_of(submersion.begin(), submersion.end(), coordinate_function))
                    found = certified_composition(f, std::move(submersion), n, dim);
            }
            if (!found || !emitted.insert({found->second, found->first}).second) continue;
            const auto& [germ, submersion] = *found;
            Json::Value row;
            row["dimension"] = dim;
            row["equations"] = Json::arrayValue;
            for (const auto& g : submersion) row["equations"].append(term_list(g));
            row["germ"] = term_list(germ);
            if (const auto zeta = brieskorn_pham_zeta(germ, k)) row["factors"] = factor_list(*zeta);
            rows.append(std::move(row));
        }
    } catch (const std::runtime_error&) {
        // Rows already appended were certified before the work limit.
    }
    return rows;
}
Json::Value pole_data(const Json::Value& r) {
    std::map<Q, Json::Value> candidates;
    std::map<Q, Json::Value> pole_orders;
    const auto visible = visible_components(r["strata"]);
    for (const auto& row : r["data"]) {
        const Z n = integer(row["n"]);
        if (n <= 0) throw std::invalid_argument("Expected a positive divisor multiplicity.");
        Q slope(-integer(row["nu"]), n);
        slope.canonicalize();
        candidates[slope].append(row["component"]);
    }
    Json::Value out;
    out["poles"] = out["candidates"] = Json::arrayValue;
    for (const auto& pole : r["poles"]) {
        const Q value = rational(pole["value"]);
        pole_orders[value] = pole["order"];
        const auto q = value.get_den();
        const Z exponent = cyclotomic(r["factors"], q);
        Json::Value row;
        row["q"] = q.get_str();
        row["exponent"] = exponent.get_str();
        row["branch"] = Json::nullValue;
        row["witness"] = exponent != 0 ? Json::Value("origin zeta") :
                         q == 1 ? Json::Value("origin H^0") : Json::Value();
        if (!exponent && q != 1) for (Json::ArrayIndex i = 0; i < r["data"].size(); ++i) {
            const auto& b = r["data"][i];
            const std::string label = b["component"].asString();
            if (strict_component(label) && visible.count(label) && integer(b["n"]) % q == 0) {
                row["branch"] = i;
                row["witness"] = "nearby smooth component";
                break;
            }
        }
        row["nearby_matches"] = Json::arrayValue;
        for (Json::ArrayIndex i = 0; i < r["nearby_factors"].size(); ++i) {
            const Z nearby = cyclotomic(r["nearby_factors"][i], q);
            if (nearby == 0) continue;
            Json::Value match;
            match["index"] = i;
            match["exponent"] = nearby.get_str();
            row["nearby_matches"].append(std::move(match));
        }
        out["poles"].append(row);
    }
    for (const auto& [slope, names] : candidates) {
        Json::Value row;
        row["value"] = slope.get_str();
        row["components"] = names;
        if (r.get("candidate_evidence", false).asBool()) {
            const Z q = slope.get_den();
            row["eigenvalue_order"] = q.get_str();
            row["cyclotomic_exponent"] = cyclotomic(r["factors"], q).get_str();
        }
        auto order = pole_orders.find(slope);
        row["pole_order"] = order == pole_orders.end() ? Json::Value(0) : order->second;
        out["candidates"].append(row);
    }
    return out;
}
Json::Value compute(const Json::Value& r) {
    const std::string op = r["operation"].asString();
    if (op == "acampo") return acampo(r["rows"]);
    if (op == "acampo_strata") return acampo_strata(r);
    if (op == "cyclotomic") return cyclotomic(r["factors"], integer(r["order"])).get_str();
    if (op == "poles") return pole_data(r);
    if (op == "isolated_euler") {
        const Z mu = integer(r["mu"]);
        Z chi = 1 + (r["dimension"].asUInt() % 2 == 1 ? mu : -mu);
        return chi.get_str();
    }
    if (op == "projective_complement") {
        const auto degree = integer(r["degree"]);
        if (!degree) throw std::invalid_argument("Expected a nonzero degree.");
        Z base = 1 - degree, power;
        mpz_pow_ui(power.get_mpz_t(), base.get_mpz_t(), r["dimension"].asUInt64());
        Z chi = (1 - power) / degree;
        return chi.get_str();
    }
    const size_t width = r["width"].asUInt64();
    const auto terms = read_terms(r["terms"], width);
    if (op == "local_eigenvalues") {
        Json::Value result;
        const auto tests = atlas::monodromy::local_eigenvalues(terms, width, r["values"]);
        bool verified = true;
        for (const auto& test : tests) if (test["root"].isNull()) verified = false;
        result["status"] = verified ? "verified" : "refuted";
        result["by_pole"] = tests;
        return result;
    }
    if (op == "transverse_contacts") return atlas::monodromy::transverse_contacts(terms, width);
    if (op == "nearby_strata") {
        auto strata = nearby_strata(terms, width, r);
        if (width == 3) {
            for (const auto& row : atlas::arrangement_nearby_strata(terms)) strata.append(row);
        }
        for (const auto& contact : atlas::monodromy::transverse_contacts(terms, width)) {
            const auto& axes = contact["axes"];
            const bool recorded = std::any_of(strata.begin(), strata.end(), [&](const auto& stratum) {
                const auto& previous = stratum["axes"];
                return previous.size() == 2 && (axes == previous ||
                       (axes[0] == previous[1] && axes[1] == previous[0]));
            });
            if (!recorded) strata.append(contact);
        }
        return strata;
    }
    if (op == "transverse_compositions") {
        // The search is optional. With "seconds" its exact computations stop
        // at that deadline; the rows certified before it are kept, and the
        // answer says whether the search was cut short.
        if (!r.isMember("seconds") || r["seconds"].isNull()) return transverse_compositions(terms, width);
        const auto at = std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(std::max(0.0, r["seconds"].asDouble())));
        atlas::DeadlineScope limit(at);
        Json::Value out;
        out["rows"] = transverse_compositions(terms, width);
        out["expired"] = limit.expired() || std::chrono::steady_clock::now() > at;
        return out;
    }
    if (op == "morse") {
        Axes axes(width);
        std::iota(axes.begin(), axes.end(), 0);
        Json::Value result;
        visit_morse(terms, axes, width, true, [&](const Axes& normal, const Q& det) {
            result["determinant"] = det.get_str();
            for (size_t i : normal) result["axes"].append(Json::UInt64(i));
            return true;
        });
        return result;
    }
    if (op == "coordinate_divisor") {
        const auto powers = coordinate_divisor(terms, width);
        return powers.empty() ? Json::Value() : strings(powers);
    }
    throw std::invalid_argument("Unknown monodromy operation.");
}
}  // namespace
extern "C" const char* atlas_monodromy(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, compute);
}
