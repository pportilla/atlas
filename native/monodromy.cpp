// Exact arithmetic for A'Campo's formula and nearby monodromy witnesses.
#include "rational_sparse_json.hpp"
#include <flint/fmpq.h>
#include <flint/fmpq_mat.h>
#include <algorithm>
#include <numeric>
#include <set>

namespace {
using namespace atlas::sparse_json;

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
    if (op == "nearby_strata") return nearby_strata(terms, width, r);
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
