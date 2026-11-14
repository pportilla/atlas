// Exact reductions of the extended left-right tangent quotient at the origin.
#include "ae_special.hpp"
#include "local_division.hpp"
#include "local_io.hpp"
#include "local_monomial.hpp"
#include "local_point_quotient.hpp"
#include "primary_decomposition.hpp"
#include <flint/fmpq.h>
#include <flint/fmpq_mat.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>

namespace atlas::local {
namespace {
Json::Value answer(const char* method, std::optional<mpz_class> value) {
    Json::Value out;
    out["method"] = method;
    out["finite"] = value.has_value();
    out["value"] = value ? Json::Value(value->get_str()) : Json::Value();
    return out;
}

bool full_rank(const Basis& components, size_t width) {
    // Extract just the first jet, irrespective of any large exponents in the tail.
    if (width == 1) {
        return std::any_of(components.begin(), components.end(), [](const Polynomial& row) {
            return !row.empty() && row.begin()->first.front() == 1;
        });
    }
    fmpq_mat_t linear;
    fmpq_mat_init(linear, components.size(), width);
    Powers axis(width);
    for (size_t j = 0; j < width; ++j) {
        axis[j] = 1;
        for (size_t i = 0; i < components.size(); ++i) {
            const auto entry = components[i].find(axis);
            if (entry != components[i].end())
                fmpq_set_mpq(fmpq_mat_entry(linear, i, j), entry->second.get_mpq_t());
        }
        axis[j] = 0;
    }
    const auto rank = fmpq_mat_rref(linear, linear);
    fmpq_mat_clear(linear);
    return static_cast<size_t>(rank) == std::min(components.size(), width);
}

mpz_class nilpotence_index(const Basis& basis, const Polynomial& residue, size_t width,
                            const std::optional<mpz_class>& cutoff) {
    // In the finite Jacobian algebra the minimal polynomial of f is T^e.
    // Consequently 1,f,...,f^(e-1) are independent. Only products of one
    // element are needed, rather than a multiplication matrix for the whole algebra.
    if (residue.empty()) return 1;
    const mpz_class order = total_degree(local_head(residue, false));
    mpz_class nilpotence = 1, degree_bound = order;
    auto power = residue;
    while (!power.empty()) {
        ++nilpotence;
        degree_bound += order;
        // Every term of this power belongs to m^degree_bound. If m^cutoff
        // already vanishes, the next multiplication and division are unnecessary.
        if (cutoff && degree_bound >= *cutoff) break;
        power = normal_form(basis, primary::multiply(power, residue), width, false, cutoff).remainder;
    }
    return nilpotence;
}

// If each least-order derivative has a pure-power leading term on its
// own axis, Buchberger's product criterion gives a local standard basis.
// Its Artinian staircase is a rectangular box. Division can discard all
// terms of degree above the top corner (Nakayama on the associated graded).
std::optional<Json::Value> diagonal_jacobian(const Polynomial& f, const Basis& derivatives, size_t width) {
    std::vector<mpz_class> bounds(width);
    mpz_class mu = 1, top = 0;
    for (const auto& derivative : derivatives) {
        if (derivative.empty()) return std::nullopt;
        auto head = local_head(derivative, false);
        const auto nonzero = std::find_if(head.begin(), head.end(), [](const mpz_class& e) { return e != 0; });
        const size_t j = nonzero - head.begin();
        if (j == width || bounds[j] != 0 ||
            std::any_of(nonzero+1, head.end(), [](const mpz_class& e) { return e != 0; })) return std::nullopt;
        bounds[j] = *nonzero;
        mu *= bounds[j];
        top += bounds[j] - 1;
    }
    const mpz_class cutoff = top + 1;
    const mpz_class degree = total_degree(local_head(f, false));
    Polynomial residue;
    bool standard = true;
    // Subtract (sum x_i ∂_i f)/degree. Euler's identity then cancels every
    // term of the lowest degree while preserving f's class in the Jacobian algebra.
    for (const auto& [powers, coefficient] : f) {
        const auto term_degree = total_degree(powers);
        if (term_degree == degree || term_degree >= cutoff) continue;
        residue.emplace(powers, coefficient * (degree - term_degree) / degree);
        for (size_t j = 0; j < width && standard; ++j)
            if (powers[j] >= bounds[j]) standard = false;
    }
    if (!standard) residue = normal_form(derivatives, residue, width, false, cutoff).remainder;
    return answer("jacobian_target_powers", mu - nilpotence_index(derivatives, residue, width, cutoff));
}

Json::Value scalar_codimension(const Polynomial& f, const std::vector<std::string>& names) {
    const size_t n = names.size();
    // For a scalar germ a linear term alone certifies submersivity.
    Powers axis_term(n);
    for (size_t axis = 0; axis < n; ++axis) {
        axis_term[axis] = 1;
        if (f.count(axis_term)) return answer("regular_map", mpz_class(0));
        axis_term[axis] = 0;
    }
    // Separated sums: each individual univariate summand belongs to its
    // derivative ideal, and the Jacobian quotient is a tensor product of DVR quotients.
    std::vector<mpz_class> orders(n);
    bool separated = true;
    for (const auto& [powers, coefficient] : f) {
        size_t axis = n;
        for (size_t j = 0; j < n; ++j) if (powers[j]) {
            if (axis != n) { separated = false; break; }
            axis = j;
        }
        if (!separated) break;
        if (axis < n && (orders[axis] == 0 || powers[axis] < orders[axis])) orders[axis] = powers[axis];
    }
    if (separated) {
        mpz_class mu = 1;
        for (const auto& order : orders) {
            if (order == 0) return answer("nonisolated_critical_locus", std::nullopt);
            mu *= order - 1;
        }
        return answer("separated_jacobian", mu - 1); // regular germs were handled first
    }

    const auto derivatives = partial_derivatives(f, n);
    if (auto result = diagonal_jacobian(f, derivatives, n)) return *result;
    const auto basis = standard_basis(derivatives, names, false);
    std::vector<Powers> heads;
    for (const auto& g : basis) if (!g.empty()) heads.push_back(local_head(g, false));
    const auto mu = monomial_point_quotient(n, heads, false).dimension;
    // f vanishes on every component of its critical locus through 0 in characteristic zero.
    // Thus the powers of f span a finite space even when the critical locus has dimension >0.
    if (!mu) return answer("nonisolated_critical_locus", std::nullopt);
    if (*mu == 0) return answer("regular_map", mpz_class(0));
    const auto cutoff = finite_cutoff(heads, n);
    const auto residue = normal_form(basis, f, n, false, cutoff).remainder;
    return answer("jacobian_target_powers", *mu - nilpotence_index(basis, residue, n, cutoff));
}

// For moderate nonmonomial parametrizations, compare two source parameters X,t.
// Their equal-image equations all contain X-t. A common relation h(X)-h(t)
// with h'(0)=0 exhibits a multiple cover, including nonlinear changes of parameter.
bool ramified_polynomial_cover(const Basis& components) {
    const Polynomial* smallest = nullptr;
    for (const auto& f : components) if (!f.empty()) {
        if (f.rbegin()->first.front() > 256) return false;
        if (!smallest || f.rbegin()->first < smallest->rbegin()->first) smallest = &f;
    }

    // If the lowest-degree component h generates the other components, its
    // ramification at the origin proves the cover without a bivariate GCD.
    // A common value on two sheets t,s gives f(t)=f(s) in all components.
    // The powers have distinct degrees, so cancellation at the top is forced.
    unsigned long common_degree = 0;
    for (const auto& f : components) if (!f.empty())
        common_degree = std::gcd(common_degree, f.rbegin()->first.front().get_ui());
    if (common_degree == 1) return false;

    Basis powers;
    const auto degree = smallest->rbegin()->first.front().get_ui();
    const auto is_generated = [&](Polynomial remainder) {
        while (!remainder.empty()) {
            const auto top = remainder.rbegin()->first.front().get_ui();
            if (top % degree || top/degree > 32) return false;
            if (powers.empty()) powers.push_back(*smallest);
            while (powers.size() < top/degree)
                powers.push_back(primary::multiply(powers.back(), *smallest));
            const auto& multiple = powers[top/degree-1];
            const mpq_class factor = remainder.rbegin()->second / multiple.rbegin()->second;
            for (const auto& [e, c] : multiple) {
                auto& value = remainder[e];
                value -= factor*c;
                if (value == 0) remainder.erase(e);
            }
        }
        return true;
    };
    if (common_degree == degree && std::all_of(components.begin(), components.end(), [&](const Polynomial& f) {
            return &f == smallest || is_generated(f);
        })) return true;  // the rank check already excluded ord(h)=1

    Basis differences;
    for (const auto& f : components) {
        if (f.empty()) continue;
        Polynomial difference;
        for (const auto& [powers, coefficient] : f) {
            difference.emplace(Powers{powers[0], 0}, coefficient);
            difference.emplace(Powers{0, powers[0]}, -coefficient);
        }
        differences.push_back(std::move(difference));
    }
    const auto common = polynomial_gcd(differences, 2);
    if (!common || common->empty()) return false;
    for (const auto& [powers, coefficient] : *common) {
        if ((powers[0] != 0) == (powers[1] != 0) || powers[0] == 1 || powers[1] == 1)
            return false;
        const auto transposed = common->find(Powers{powers[1], powers[0]});
        if (transposed == common->end() || transposed->second != -coefficient) return false;
    }
    // Dividing f_i(X)-f_i(t) by the monic h(X)-h(t) has zero remainder.
    // In the free Q[h]-basis 1,X,...,X^(deg(h)-1) of Q[X], this forces
    // every nonconstant basis coefficient to vanish. Thus every f_i is in Q[h].
    return true;
}

// For a primitive monomial parametrization put S=<a_1,...,a_r>. Before
// quotienting by df, each target coordinate has genus(S) missing monomials.
// The vector t^j df kills one such direction precisely when j-1+a_i is a
// gap for some i. These vectors have disjoint supports for distinct j.
// The killed directions are j=0 and the gaps j-1 which are not pseudo-Frobenius.
std::optional<Json::Value> monomial_curve(const Basis& components) {
    std::set<mpz_class> degrees;
    mpz_class divisor = 0;
    bool monomial = true;
    for (const auto& row : components) {
        monomial &= row.size() <= 1;
        for (const auto& [powers, coefficient] : row) {
            mpz_gcd(divisor.get_mpz_t(), divisor.get_mpz_t(), powers[0].get_mpz_t());
            if (monomial) degrees.insert(powers[0]);
        }
    }
    // For p>1 a factorization t -> t^d gives infinitely many missing
    // target directions on a nonzero residue class modulo d, even for
    // polynomials with several terms (source vector fields have rank at most one).
    if (divisor > 1) return answer("multiple_cover", std::nullopt);
    if (!monomial) {
        if (ramified_polynomial_cover(components)) return answer("multiple_cover", std::nullopt);
        return std::nullopt;
    }
    mpz_class genus, type;
    if (degrees.size() == 2) {
        genus = (*degrees.begin()-1) * (*degrees.rbegin()-1) / 2;
        type = 1;
    } else {
        const auto& smallest = *degrees.begin();
        // A shortest path for each residue suffices even if the conductor is huge.
        // Avoid allocating a residue graph exceeding the size of this shortcut.
        if (smallest * degrees.size() > 1000000) return std::nullopt;
        const size_t m = smallest.get_ui();
        std::vector<std::pair<size_t, mpz_class>> edges;
        for (const auto& a : degrees) {
            const size_t residue = mpz_fdiv_ui(a.get_mpz_t(), m);
            if (residue) edges.emplace_back(residue, a);
        }
        // The Apéry set contains the least element of S in each residue mod m.
        using Item = std::pair<mpz_class, size_t>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
        std::vector<mpz_class> apery(m, -1);
        apery[0] = 0;
        queue.emplace(0, 0);
        while (!queue.empty()) {
            const auto [weight, index] = queue.top();
            queue.pop();
            if (apery[index] != weight) continue;
            for (const auto& [step, a] : edges) {
                const size_t next = (index + step) % m;
                mpz_class candidate = weight + a;
                if (apery[next] < 0 || candidate < apery[next]) {
                    apery[next] = candidate;
                    queue.emplace(std::move(candidate), next);
                }
            }
        }
        genus = 0;
        type = 0;
        for (size_t i = 1; i < m; ++i) {
            genus += (apery[i] - i) / m;
            // The last gap in a residue class is pseudo-Frobenius iff
            // adding each positive generator moves it into the semigroup.
            const bool maximal = std::all_of(edges.begin(), edges.end(), [&](const auto& edge) {
                return apery[i] - smallest + edge.second >= apery[(i + edge.first) % m];
            });
            if (maximal) ++type;
        }
    }
    return answer("monomial_curve_semigroup", (components.size()-1) * genus + type - 1);
}

// Restrict to a coordinate line and project onto one target component.
// If its full gradient vanishes there, the source tangent image projects to zero.
// At most one nonzero image series of order >1 contributes only Q[[h(t)]],
// a subspace of infinite codimension. The polynomial terms identify all possible
// lines at once, including terms whose transverse derivative survives restriction.
bool ramified_axis(const Basis& components, size_t width) {
    std::vector<size_t> image_count(width), zero_gradients(width);
    std::vector<bool> regular(width);
    for (const auto& row : components) {
        std::vector<bool> nonzero(width), restricted(width);
        bool has_linear_term = false;
        for (const auto& [powers, coefficient] : row) {
            size_t first = width, second = width;
            bool higher_support = false;
            for (size_t i = 0; i < width; ++i) if (powers[i] != 0) {
                if (first == width) first = i;
                else if (second == width) second = i;
                else { higher_support = true; break; }
            }
            if (higher_support) continue;
            if (second == width) {
                nonzero[first] = restricted[first] = true;
                if (powers[first] == 1) regular[first] = has_linear_term = true;
            } else {
                if (powers[second] == 1) nonzero[first] = true;
                if (powers[first] == 1) nonzero[second] = true;
            }
        }
        for (size_t i = 0; i < width; ++i) {
            if (restricted[i]) ++image_count[i];
            if (!has_linear_term && !nonzero[i]) ++zero_gradients[i];
        }
    }
    for (size_t i = 0; i < width; ++i)
        if (zero_gradients[i] && image_count[i] <= 1 && !regular[i]) return true;
    return false;
}

// For (u,v²,h), set s=v² and drop the even part of h by a target shear.
// Write h=vP(u,s). With A=Q[[u,s]], target fields span A+vP A in
// each position. Modulo those fields every position is v*(A/P).
// The odd multiples of ∂f/∂u_i and even multiples of ∂f/∂v eliminate
// the first n positions. Their complementary parities impose ∂P/∂u_i
// and P+2s∂P/∂s in the last position. This gives an ordinary ideal length.
std::optional<Json::Value> double_cover_hypersurface(const Basis& components, const std::vector<std::string>& names) {
    const size_t width = names.size(), count = components.size();
    if (count != width+1) return std::nullopt;
    std::vector<size_t> linear(width, count), square(width, count);
    for (size_t i = 0; i < count; ++i) {
        if (components[i].size() != 1) continue;
        const auto& powers = components[i].begin()->first;
        const auto axis = std::find_if(powers.begin(), powers.end(), [](const mpz_class& e) { return e != 0; });
        if (axis == powers.end() ||
            std::any_of(axis+1, powers.end(), [](const mpz_class& e) { return e != 0; })) continue;
        const size_t j = axis - powers.begin();
        if (*axis == 1) linear[j] = i;
        if (*axis == 2) square[j] = i;
    }
    const size_t missing = std::count(linear.begin(), linear.end(), count);
    if (missing != 1) return std::nullopt;
    const size_t v = std::find(linear.begin(), linear.end(), count) - linear.begin();
    if (square[v] == count) return std::nullopt;
    linear[v] = square[v];
    std::vector<bool> selected(count);
    for (const auto i : linear) selected[i] = true;
    const auto& h = components[std::find(selected.begin(), selected.end(), false) - selected.begin()];
    Polynomial odd;
    for (const auto& [powers, coefficient] : h) {
        if (!mpz_odd_p(powers[v].get_mpz_t())) continue;
        auto e = powers;
        e[v] /= 2;
        odd.emplace(std::move(e), coefficient);
    }
    // For P=sum of univariate terms, ordinary partials bound the u_i
    // orders by ord(P_i)-1; s∂P/∂s bounds s by ord(P_summand).
    // P itself then belongs to the derivative ideal, with unit quotients
    // interpreted in the completed local ring.
    const Powers zero(width);
    auto orders = zero;
    bool separated = true;
    for (const auto& [e, coefficient] : odd) {
        const auto nonzero = std::find_if(e.begin(), e.end(), [](const mpz_class& x) { return x != 0; });
        if (nonzero == e.end()) return answer("double_cover_hypersurface", mpz_class(0));
        if (std::any_of(nonzero+1, e.end(), [](const mpz_class& x) { return x != 0; })) {
            separated = false;
            break;
        }
        const auto j = static_cast<size_t>(nonzero-e.begin());
        if (orders[j] == 0 || orders[j] > *nonzero) orders[j] = *nonzero;
    }
    if (separated) {
        mpz_class length = 1;
        for (size_t j = 0; j < width; ++j)
            if (j != v && orders[j] == 1) return answer("double_cover_hypersurface", mpz_class(0));
        for (size_t j = 0; j < width; ++j) {
            if (orders[j] == 0) return answer("double_cover_hypersurface", std::nullopt);
            length *= orders[j] - (j != v);
        }
        return answer("double_cover_hypersurface", length);
    }
    auto ideal = partial_derivatives(odd, width);
    Polynomial logarithmic;
    for (const auto& [powers, coefficient] : odd)
        if (powers[v] != 0) logarithmic.emplace(powers, coefficient * powers[v]);
    ideal[v] = std::move(logarithmic);
    ideal.push_back(std::move(odd));
    return answer("double_cover_hypersurface", ideal_colength_at_origin(ideal, names));
}

// For (u,g(v)) the u component is removed by the source field d/du.
// Target and remaining source fields then act coefficient by coefficient in u.
// Thus any nonzero normal class of g gives infinitely many independent classes.
std::optional<Json::Value> trivial_unfolding(const Basis& components, const std::vector<std::string>& names) {
    for (size_t i = 0; i < components.size(); ++i) {
        if (components[i].size() != 1) continue;
        const auto& linear = components[i].begin()->first;
        if (total_degree(linear) != 1) continue;
        const auto axis = std::find(linear.begin(), linear.end(), 1) - linear.begin();
        bool split = true;
        Basis reduced;
        for (size_t j = 0; j < components.size() && split; ++j) if (j != i) {
            auto& row = reduced.emplace_back();
            for (const auto& [powers, coefficient] : components[j]) {
                if (powers[axis]) {
                    // Pure powers of u can be subtracted by a target shear.
                    if (total_degree(powers) != powers[axis]) {
                        split = false;
                        break;
                    }
                } else {
                    auto exponent = powers;
                    exponent.erase(exponent.begin()+axis);
                    row.emplace(std::move(exponent), coefficient);
                }
            }
        }
        if (!split) continue;
        auto variables = names;
        variables.erase(variables.begin()+axis);
        if (const auto slice = ae_special(reduced, variables)) {
            if ((*slice)["finite"].asBool() && (*slice)["value"].asString() == "0")
                return answer("trivial_unfolding", mpz_class(0));
            return answer("trivial_unfolding", std::nullopt);
        }
    }
    return std::nullopt;
}

}

std::optional<Json::Value> ae_special(const Basis& components, const std::vector<std::string>& names) {
    if (names.empty() || components.empty())
        throw std::invalid_argument("A map germ needs source and target coordinates.");
    if (std::all_of(components.begin(), components.end(), [](const Polynomial& f) { return f.empty(); }))
        return answer("constant_map", std::nullopt);
    if (components.size() == 1) {
        if (names.size() == 1) {
            const auto& order = components.front().begin()->first[0];
            if (order == 1) return answer("regular_map", mpz_class(0));
            // In the DVR, (f')=(t^{ord(f)-1}) and the target contributes exactly 1.
            return answer("univariate_order", order - 2);
        }
        return scalar_codimension(components.front(), names);
    }
    if (full_rank(components, names.size())) return answer("regular_map", mpz_class(0));
    if (names.size() == 1) return monomial_curve(components);

    // Along an omitted axis the map is constant and its differential has
    // constant deficient rank. Projecting to one missing target direction
    // gives all univariate series modulo constants as a quotient.
    std::vector<bool> used(names.size());
    for (const auto& row : components)
        for (const auto& [powers, coefficient] : row)
            for (size_t j = 0; j < used.size(); ++j) if (powers[j] != 0) used[j] = true;
    if (std::find(used.begin(), used.end(), false) != used.end())
        return answer("unused_source_coordinate", std::nullopt);
    if (auto slice = trivial_unfolding(components, names)) return slice;
    if (auto cover = double_cover_hypersurface(components, names)) return cover;
    if (ramified_axis(components, names.size())) return answer("ramified_axis", std::nullopt);
    return std::nullopt;
}
}
