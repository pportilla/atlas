// Exact arithmetic for A'Campo's formula and nearby monodromy witnesses.
#include "rational_sparse_json.hpp"
#include <flint/fmpq.h>
#include <flint/fmpq_mat.h>
#include <algorithm>
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
Json::Value acampo(const Json::Value& rows) {
    std::map<Z, Z> powers;
    for (const auto& row : rows) {
        const Z n = integer(row[0]);
        if (n <= 0) throw std::invalid_argument("Expected a positive divisor multiplicity.");
        powers[n] -= integer(row[1]);
    }
    Json::Value result;
    result["factors"] = Json::arrayValue;
    Z chi = 0;
    for (const auto& [n, e] : powers) {
        if (!e) continue;
        Json::Value pair(Json::arrayValue);
        pair.append(n.get_str());
        pair.append(e.get_str());
        result["factors"].append(pair);
        chi -= n * e;
    }
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
Json::Value binomial(const Terms& terms, size_t width, const Json::Value& missing) {
    Json::Value out(Json::arrayValue);
    if (terms.size() != 2 || width < 3) return out;
    const auto& a = terms.rbegin()->first;
    const auto& b = terms.begin()->first;
    std::vector<Axis> left, right;
    for (size_t i = 0; i < width; ++i) {
        if (a[i] && !b[i]) left.emplace_back(i, a[i]);
        if (b[i] && !a[i]) right.emplace_back(i, b[i]);
    }
    std::set<Z> orders;
    for (const auto& x : missing) if (integer(x) > 1) orders.insert(integer(x));
    auto witness = [&](const Z& q, std::string model, const std::vector<Axis>& axes, const Z& exponent) {
        Json::Value item;
        item["order"] = q.get_str();
        item["model"] = model;
        item["exponent"] = exponent.get_str();
        for (const auto& [axis, power] : axes) {
            item["axes"].append(Json::UInt64(axis));
            item["powers"].append(power.get_str());
        }
        out.append(item);
    };
    // A transverse plane binomial gives the primitive-root valuation of
    // (1-T^lcm(a,b))^gcd(a,b) / ((1-T^a)(1-T^b)).
    for (const auto& x : left) for (const auto& y : right) {
        Z period = lcm(x.second, y.second), copies = gcd(x.second, y.second);
        for (auto it = orders.begin(); it != orders.end();) {
            const auto& q = *it;
            if (period % q == 0) {
                Z e = copies - (x.second % q == 0) - (y.second % q == 0);
                if (e) { witness(q, "X^a + Y^b", {x,y}, e); it = orders.erase(it); continue; }
            }
            ++it;
        }
    }
    if (width >= 4) for (int swap = 0; swap < 2; ++swap) {
        const auto& product = swap ? right : left;
        const auto& powers = swap ? left : right;
        for (size_t i = 0; i < product.size(); ++i) {
            if (product[i].second != 1) continue;
            for (size_t j = i + 1; j < product.size(); ++j) {
                if (product[j].second != 1) continue;
                for (const auto& z : powers) for (auto it = orders.begin(); it != orders.end();) {
                    if (z.second % *it == 0) {
                        witness(*it, "X*Y + Z^m", {product[i],product[j],z}, -1);
                        it = orders.erase(it);
                    } else ++it;
                }
            }
        }
    }
    return out;
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
Json::Value morse(const Terms& terms, size_t width) {
    // Try odd codimension in the same order as the public certificate. A
    // nonsingular Hessian at parameters = 1 is a dense-open certificate.
    for (size_t k = 1; k < width; k += 2) {
        std::vector<bool> selected(width);
        std::fill_n(selected.begin(), k, true);
        do {
            std::vector<size_t> axes;
            for (size_t i = 0; i < width; ++i) if (selected[i]) axes.push_back(i);
            const Q determinant = transverse_determinant(terms, axes);
            if (determinant) {
                Json::Value result;
                result["determinant"] = determinant.get_str();
                for (auto i : axes) result["axes"].append(Json::UInt64(i));
                return result;
            }
        } while (std::prev_permutation(selected.begin(), selected.end()));
    }
    return {};
}
Json::Value pole_data(const Json::Value& r) {
    std::map<Q, Json::Value> candidates;
    std::map<Q, Json::Value> pole_orders;
    std::set<std::string> visible;
    for (const auto& stratum : r["strata"]) {
        if (integer(stratum[1]) == 0) continue;
        for (const auto& name : stratum[0]) visible.insert(name.asString());
    }
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
            const bool strict = label == "strict" || (label.size() > 1 && label.front() == 'S' &&
                label.find_first_not_of("0123456789", 1) == std::string::npos);
            if (strict && visible.count(label) && integer(b["n"]) % q == 0) {
                row["branch"] = i;
                row["witness"] = "nearby smooth component";
                break;
            }
        }
        out["poles"].append(row);
    }
    for (const auto& [slope, names] : candidates) {
        Json::Value row;
        row["value"] = slope.get_str();
        row["components"] = names;
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
    if (op == "binomial") return binomial(terms, width, r["orders"]);
    if (op == "morse") return morse(terms, width);
    if (op == "coordinate_divisor") {
        if (terms.empty()) return {};
        auto powers = terms.begin()->first;
        for (const auto& [e, c] : terms) for (size_t i = 0; i < width; ++i) powers[i] = std::min(powers[i], e[i]);
        return terms.count(powers) && powers != Powers(width) ? strings(powers) : Json::Value();
    }
    throw std::invalid_argument("Unknown monodromy operation.");
}
}  // namespace
extern "C" const char* atlas_monodromy(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, compute);
}
