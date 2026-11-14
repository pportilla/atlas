// Sparse geometric factor certificates and the ordered projection schedule.
#include "rational_sparse_json.hpp"
#include "primary_decomposition.hpp"

#include <algorithm>
#include <numeric>
#include <set>

namespace atlas::components {
using namespace sparse_json;
namespace {

Z degree(const Powers& powers) { return std::accumulate(powers.begin(), powers.end(), Z(0)); }

struct Support {
    std::vector<Powers> rows;
    Powers lead, bounds;
    Z degree = -1;
    explicit Support(const Json::Value& data) {
        if (!data.isArray() || data.empty()) throw std::invalid_argument("Expected a nonempty support.");
        bounds.resize(data[0].size());
        for (const auto& powers : data) {
            auto e = read_exponents(powers, bounds.size());
            Z total = components::degree(e);
            if (total > degree || (total == degree && e > lead)) {
                degree = total;
                lead = e;
            }
            for (size_t j = 0; j < e.size(); ++j) bounds[j] = std::max(bounds[j], e[j]);
            rows.push_back(std::move(e));
        }
    }
};

template<class Visit>
bool enumerate(Powers& row, const Powers& bounds, size_t j, const Z& remaining, Visit& visit) {
    if (j == row.size()) return visit(row);
    const Z limit = std::min(bounds[j], remaining);
    for (row[j] = 0; row[j] <= limit; ++row[j]) {
        const Z rest = remaining-row[j];
        if (!enumerate(row, bounds, j+1, rest, visit)) return false;
    }
    return true;
}

Json::Value factor_supports(const Support& p, const Z& effort) {
    Json::Value layouts(Json::arrayValue), result;
    const Z max_unknowns = 6+2*effort, max_splits = 32*effort;
    Z checked = 0;
    auto split = [&](const Powers& alpha) {
        Z left_degree = degree(alpha);
        if (!left_degree) return true;
        if (++checked > max_splits) return false;
        Powers beta = p.lead;
        for (size_t j=0; j<beta.size(); ++j) beta[j] -= alpha[j];
        Json::Value sides(Json::arrayValue);
        Z count = 0;
        for (const Powers* head : {&alpha, static_cast<const Powers*>(&beta)}) {
            Json::Value terms(Json::arrayValue);
            terms.append(strings(*head));
            Z top_degree = degree(*head);
            auto add = [&](const Powers& term) {
                if (degree(term) == top_degree && term >= *head) return true;
                if (++count > max_unknowns) return false;
                terms.append(strings(term));
                return true;
            };
            Powers term(head->size());
            if (!enumerate(term, p.bounds, 0, top_degree, add)) return false;
            sides.append(std::move(terms));
        }
        layouts.append(std::move(sides));
        return true;
    };
    Powers alpha(p.bounds.size());
    result["complete"] = enumerate(alpha, p.lead, 0, p.degree/2, split);
    result["lead"] = strings(p.lead);
    result["layouts"] = std::move(layouts);
    return result;
}

using IntMatrix = std::vector<Powers>;
IntMatrix identity(size_t n) {
    IntMatrix matrix(n, Powers(n));
    for (size_t i=0; i<n; ++i) matrix[i][i]=1;
    return matrix;
}
Json::Value matrix_rows(const IntMatrix& matrix) {
    Json::Value rows(Json::arrayValue);
    for (const auto& row : matrix) {
        Json::Value entries(Json::arrayValue);
        for (size_t j=0; j<row.size(); ++j) {
            if (row[j] == 0) continue;
            Json::Value term(Json::arrayValue);
            term.append(static_cast<Json::UInt64>(j));
            term.append(row[j].get_str());
            entries.append(std::move(term));
        }
        rows.append(std::move(entries));
    }
    return rows;
}

Json::Value projection(size_t n, Z code) {
    if (!n || code < 0) throw std::invalid_argument("Invalid projection index.");
    auto matrix = identity(n), inverse = matrix;
    if (code > 0 && code < n) {
        size_t k = code.get_ui();
        std::rotate(matrix.rbegin(), matrix.rbegin()+k, matrix.rend());
        std::rotate(inverse.begin(), inverse.begin()+k, inverse.end());
    } else if (n > 1 && code == n) {
        Z power=1;
        for (size_t j=0; j+1<n; ++j) {
            power *= 2;
            matrix.back()[j] = power;
            inverse.back()[j] = -power;
        }
    } else if (code > n && n > 1) {
        code -= n;
        const Z alphabet = Z(n)*(n-1)*2;
        while (code > 0) {
            --code;
            Z index=code%alphabet;
            code/=alphabet;
            int sign = mpz_odd_p(index.get_mpz_t()) ? -1 : 1;
            index/=2;
            size_t i=Z(index/(n-1)).get_ui(), j=Z(index%(n-1)).get_ui();
            if (j>=i) ++j;
            for (size_t k=0; k<n; ++k) {
                matrix[i][k] += sign*matrix[j][k];
                inverse[k][j] -= sign*inverse[k][i];
            }
        }
    }
    Json::Value pair(Json::arrayValue);
    pair.append(matrix_rows(matrix));
    pair.append(matrix_rows(inverse));
    return pair;
}

Json::Value factor_conditions(const Json::Value& q) {
    const size_t n = q["width"].asUInt(), field = q["marked"].asBool() ? 1 : 0;
    const auto f = read_terms(q["polynomial"], n+field);
    const auto& sides = q["sides"];
    if (sides.size() != 2 || sides[0].empty() || sides[1].empty())
        throw std::invalid_argument("Expected two factor supports.");
    const size_t unknowns = sides[0].size()+sides[1].size()-2;
    const Powers a = read_exponents(sides[0][0], n), b = read_exponents(sides[1][0], n);
    Powers leading(n);
    for (size_t j = 0; j < n; ++j) leading[j] = a[j]+b[j];

    // Work in Q[unknowns,theta]. The Python caller appends the marked
    // field relation to the resulting equations before testing existence.
    std::map<Powers, Terms> equations;
    Terms leading_coefficient;
    for (const auto& [e,c] : f) {
        Powers x(e.begin(), e.begin()+n), parameter(unknowns+field);
        if (field) parameter.back() = e.back();
        equations[x][parameter] -= c;
        if (x == leading) leading_coefficient[parameter] += c;
    }
    if (leading_coefficient.empty()) throw std::invalid_argument("Missing leading coefficient.");
    size_t variable = 0;
    std::vector<std::vector<std::pair<Powers,Terms>>> factors(2);
    for (Json::ArrayIndex side=0; side<2; ++side) {
        for (Json::ArrayIndex i=0; i<sides[side].size(); ++i) {
            Powers parameter(unknowns+field);
            Terms coefficient;
            if (i==0 && side==1) coefficient = leading_coefficient;
            else {
                if (i) parameter[variable++] = 1;
                coefficient[parameter] = 1;
            }
            factors[side].emplace_back(read_exponents(sides[side][i], n), std::move(coefficient));
        }
    }
    // Each coefficient of A*B-f in the geometric variables must vanish.
    for (const auto& [ma, ca] : factors[0]) {
        for (const auto& [mb, cb] : factors[1]) {
            Powers x(n);
            for (size_t j=0; j<n; ++j) x[j] = ma[j]+mb[j];
            auto& equation = equations[x];
            for (const auto& [ea, va] : ca) for (const auto& [eb, vb] : cb) {
                Powers e(unknowns+field);
                for (size_t k=0; k<e.size(); ++k) e[k] = ea[k]+eb[k];
                equation[e] += va*vb;
            }
        }
    }
    Json::Value out(Json::arrayValue);
    for (const auto& [monomial, equation] : equations) {
        auto row = term_list(equation);
        if (!row.empty()) out.append(std::move(row));
    }
    return out;
}

// A rational map presents each coordinate as a numerator and denominator.
// Clear the same denominator in both members of a fraction before composing.
Json::Value homogenize_fraction(const Json::Value& q) {
    const size_t width = q["width"].asUInt();
    if (q["pair"].size() != 2) throw std::invalid_argument("Expected a numerator and a denominator.");
    std::vector<Terms> pair;
    Powers degrees(width);
    for (const auto& source : q["pair"]) {
        pair.push_back(read_terms(source, width));
        for (const auto& [e, c] : pair.back()) {
            if (c == 0) continue;
            for (size_t i = 0; i < width; ++i) degrees[i] = std::max(degrees[i], e[i]);
        }
    }
    Json::Value result(Json::arrayValue);
    for (const auto& source : pair) {
        Terms polynomial;
        for (const auto& [e, c] : source) {
            auto powers = e;
            for (size_t i = 0; i < width; ++i) powers.push_back(degrees[i] - e[i]);
            polynomial.emplace(std::move(powers), c);
        }
        result.append(term_list(polynomial));
    }
    return result;
}

Json::Value run(const Json::Value& q) {
    const auto op = q["operation"].asString();
    if (op == "prepare_overlap") return primary::prepare_overlap(q);
    if (op == "generic_overlap") return primary::generic_overlap(q);
    if (op == "homogenize_fraction") return homogenize_fraction(q);
    if (op == "factor_conditions") return factor_conditions(q);
    if (op == "projection") return projection(q["width"].asUInt(), Z(q["index"].asString(), 10));
    Support p(q["support"]);
    if (op == "factor_supports") return factor_supports(p, Z(q["effort"].asString(), 10));
    if (op == "finite_graph") {
        const auto width = p.bounds.size();
        if (!width) throw std::invalid_argument("Expected a coefficient parameter.");
        std::set<size_t> pivots;
        int eliminant = -1;
        for (size_t i = 0; i < p.rows.size(); ++i) {
            const auto& head = p.rows[i];
            const auto pivot = std::find_if(head.begin(), head.end()-1, [](const auto& power) { return power != 0; });
            if (pivot == head.end()-1) {
                if (q["require_eliminant"].asBool()) {
                    if (eliminant >= 0 || head.back() < 2) return Json::Value();
                    eliminant = static_cast<int>(i);
                }
            } else {
                if (degree(head) != 1) return Json::Value();
                pivots.insert(static_cast<size_t>(pivot-head.begin()));
            }
        }
        if (width-1-pivots.size() != q["dimension"].asUInt64()) return Json::Value();
        if (q["require_eliminant"].asBool() && eliminant < 0) return Json::Value();
        return Json::Value(eliminant);
    }
    if (op == "profile") {
        Json::Value r;
        r["degrees"] = strings(p.bounds);
        r["degree"] = p.degree.get_str();
        r["homogeneous"] = std::all_of(p.rows.begin(), p.rows.end(), [&](const auto& e){return degree(e)==p.degree;});
        return r;
    }
    if (op == "homogenize") {
        Json::Value rows(Json::arrayValue);
        for (auto& e : p.rows) {
            e.push_back(p.degree-degree(e));
            rows.append(strings(e));
        }
        return rows;
    }
    throw std::invalid_argument("Unknown geometric component operation.");
}
} // namespace
} // namespace atlas::components

extern "C" const char* atlas_component_geometry(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, atlas::components::run);
}

// Chart intersections are computed by the caller's exact ideal backend.
// Only a certified inequality rules out an entire previously glued class.
extern "C" int atlas_partition_components(size_t count, const size_t* charts, const size_t* strata,
                                          int (*compare)(size_t, size_t), size_t* classes) noexcept {
    try {
        std::iota(classes, classes + count, size_t(0));
        auto root = [&](size_t i) {
            while (classes[i] != i) {
                classes[i] = classes[classes[i]];
                i = classes[i];
            }
            return i;
        };
        for (size_t i = 0; i < count; ++i) {
            std::set<size_t> distinct;
            for (size_t j = 0; j < i; ++j) {
                if (charts[i] == charts[j] || strata[i] != strata[j]) continue;
                const auto first = root(i), second = root(j);
                if (first == second || distinct.count(second)) continue;
                switch (compare(i, j)) {
                    case 1:  classes[first] = second; break;
                    case -1: distinct.insert(second); break;
                    case 0:  break;
                    default: return -1;
                }
            }
        }
        for (size_t i = 0; i < count; ++i) classes[i] = root(i);
        return 0;
    } catch (...) {
        return -1;
    }
}
