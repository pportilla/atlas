// Integration in the fibre coordinate of three split graphs over a plane log resolution.
#pragma once
#include <gmpxx.h>
#include <json/json.h>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace atlas {
namespace three_graph_integration {
using J = Json::Value;
using Z = mpz_class;
using Labels = std::vector<std::string>;

struct Contact {
    Z d, e;
    // The pair with the longest contact, or -1 if all three orders agree.
    int pair;
};

struct Integral {
    J data{Json::arrayValue}, monodromy{Json::arrayValue};
    std::map<Labels, Z> terms;

    void factor(const std::string& label, const Z& n, const Z& nu, bool auxiliary = false) {
        J row;
        row["component"] = label;
        row["n"] = n.get_str();
        row["nu"] = nu.get_str();
        if (auxiliary) row["auxiliary"] = true;
        data.append(row);
    }

    void term(Labels denominators, const Z& chi, const char* extra = nullptr) {
        if (extra) denominators.emplace_back(extra);
        std::sort(denominators.begin(), denominators.end());
        terms[denominators] += chi;
    }

    // For a fixed base arc with contact orders d,d,e, integrate the rooted
    // balls in the fibre. Put H=1+s, P=1+2s, T=1+3s. The first split
    // carries -1-1/T+1/H+1/P, the second -1-1/P+2/H. The interval
    // before the first split also supplies a constant 1/T, taken once
    // after integrating over the smooth base. These coefficients include
    // the punctured residue lines at the two splitting stages.
    void splits(const Labels& first, const Labels& second, const Z& chi) {
        term(first, -chi);
        term(first, -chi, "triple ball");
        term(first, chi, "strict");
        term(first, chi, "pair ball");
        term(second, -chi);
        term(second, -chi, "pair ball");
        term(second, 2*chi, "strict");
    }

    // Signed exponent in the det^( (-1)^j ) convention. Atlas's common
    // A'Campo reducer negates it for its reciprocal convention.
    void lefschetz(const std::string& label, const Z& n, const Z& chi) {
        if (chi == 0) return;
        J row;
        row["component"] = label;
        row["multiplicity"] = n.get_str();
        row["euler_characteristic"] = chi.get_str();
        monodromy.append(row);
    }

    J finish() {
        J result;
        result["data"] = std::move(data);
        result["monodromy"] = std::move(monodromy);
        J strata(Json::arrayValue);
        for (const auto& [names, coefficient] : terms) {
            if (coefficient == 0) continue;
            J row(Json::arrayValue), labels(Json::arrayValue);
            for (const auto& name : names) labels.append(name);
            row.append(labels);
            row.append(coefficient.get_str());
            strata.append(row);
        }
        result["strata"] = std::move(strata);
        return result;
    }
};

inline std::string first(const std::string& component) { return component + " triple split"; }
inline std::string second(const std::string& component) { return component + " pair split"; }

inline Contact contact(const J& row) {
    if (!row["orders"].isArray() || row["orders"].size() != 3)
        throw std::invalid_argument("Three root-difference valuations are required.");
    std::vector<Z> orders;
    for (const auto& value : row["orders"]) orders.emplace_back(value.asString(), 10);
    const int longest = int(std::max_element(orders.begin(), orders.end()) - orders.begin());
    std::sort(orders.begin(), orders.end());
    if (orders[0] <= 0 || orders[0] != orders[1] ||
        orders[0] != Z(row["d"].asString(),10) || orders[2] != Z(row["e"].asString(),10))
        throw std::invalid_argument("Inconsistent contact orders on the exceptional divisor.");
    return {orders[0], orders[2], orders[0] == orders[2] ? -1 : longest};
}
} // namespace three_graph_integration

inline Json::Value three_graph_zetas(const Json::Value& base) {
    using namespace three_graph_integration;
    using three_graph_integration::Z;
    Integral sum;
    std::map<std::string, Contact> contacts;
    sum.factor("strict", 1, 1);
    sum.factor("triple ball", 3, 1, true);
    sum.factor("pair ball", 2, 1, true);
    sum.factor("contact branch", 2, 2);
    sum.term({"triple ball"}, 1);
    for (const auto& row : base["divisors"]) {
        const std::string id = row["id"].asString();
        const Z nu(row["nu"].asString(), 10), chi(row["chi"].asString(), 10);
        const auto c = contact(row);
        contacts.emplace(id, c);
        const Z n1 = 3*c.d, n2 = c.d + 2*c.e;
        sum.factor(first(id), n1, nu + c.d);
        sum.factor(second(id), n2, nu + c.e);
        sum.splits({first(id)}, {second(id)}, chi);
        // Transverse plane triple: (1-t^3)/((1-t^(3d))(1-t^(d+2e))).
        sum.lefschetz(id + " open, initial", 3, chi);
        sum.lefschetz(first(id), n1, -chi);
        sum.lefschetz(second(id), n2, -chi);
        // Blowing up a base point adds a projective line in place of each
        // of the three points z^3=epsilon. Remove their permutation class.
        sum.lefschetz(id + " base blowup correction", 3, -1);
    }
    for (const auto& edge : base["edges"]) {
        const std::string a = edge[0].asString(), b = edge[1].asString();
        const int p = contacts.at(a).pair, q = contacts.at(b).pair;
        if (p != -1 && q != -1 && p != q)
            throw std::invalid_argument("The longest root contacts are incomparable at a crossing.");
        sum.splits({first(a), first(b)}, {second(a), second(b)}, 1);
        // Quasi-ordinary crossing: the first difference is divisible by
        // both base coordinates, giving the zeta of the degree-three fibre.
        sum.lefschetz(a + " / " + b, 3, 1);
    }
    for (const auto& branch : base["branches"]) {
        const std::string id = branch["divisor"].asString();
        const auto& c = contacts.at(id);
        const int p = branch["pair"].asInt();
        if (p < 0 || p > 2 || (c.pair != -1 && p != c.pair))
            throw std::invalid_argument("The strict contact branch is incompatible with the exceptional contact.");
        // A reduced strict contact has valuations 0,0,1 and discrepancy 1.
        // Its first-stage factor equals 1, its second-stage factor is 2+2s.
        sum.splits({first(id)}, {second(id), "contact branch"}, 1);
        // Restrict to the contact branch. Two roots coalesce identically,
        // and the third differs by a unit times u^d.
        const auto label = id + " / contact pair " + std::to_string(p);
        sum.lefschetz(label, 3, 1);
        sum.lefschetz(label, 3*c.d, -1);
    }
    return sum.finish();
}
} // namespace atlas
