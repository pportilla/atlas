// Propose polynomial graphs from repeated binomial Newton faces. Every
// proposal still needs the independent graph substitution, fan and SNC tests.
#pragma once

#include "local_io.hpp"
#include <algorithm>
#include <array>
#include <iterator>
#include <json/json.h>
#include <optional>
#include <set>

namespace atlas::toric_graph_candidates {
using local::Polynomial;
using local::Powers;
using Variables = std::vector<std::string>;

struct RepeatedFace {
    Polynomial binomial, multiplier;
    unsigned long power;
};

inline std::optional<RepeatedFace> repeated_face(const Polynomial& face) {
    if (face.size() < 3) return std::nullopt;
    // In characteristic zero every degree-e binomial power has all e+1
    // coefficients. Read them in descending lexicographic order along its edge.
    const unsigned long e = face.size() - 1;
    const auto& [high, leading] = *face.rbegin();
    const auto& low = face.begin()->first;
    Powers offset(high.size()), a(high.size()), b(high.size());
    for (size_t i = 0; i < high.size(); ++i) {
        offset[i] = std::min(high[i], low[i]);
        a[i] = high[i] - offset[i];
        b[i] = low[i] - offset[i];
        if (a[i] % e != 0 || b[i] % e != 0) return std::nullopt;
        a[i] /= e;
        b[i] /= e;
    }
    const Powers constant(high.size());
    if (a == constant || b == constant) return std::nullopt;
    const mpq_class ratio = std::next(face.rbegin())->second / (e * leading);
    mpq_class coefficient = leading;
    Powers expected = high;
    unsigned long j = 0;
    for (auto term = face.rbegin(); term != face.rend(); ++term, ++j) {
        if (term->first != expected || term->second != coefficient) return std::nullopt;
        for (size_t i = 0; i < high.size(); ++i) expected[i] += b[i] - a[i];
        coefficient *= ratio * (e - j);
        coefficient /= j + 1;
    }
    return RepeatedFace{{{a, 1}, {b, ratio}}, {{offset, leading}}, e};
}

inline std::optional<size_t> pure_axis(const Powers& powers) {
    std::optional<size_t> axis;
    for (size_t i = 0; i < powers.size(); ++i) if (powers[i] != 0) {
        if (axis) return std::nullopt;
        axis = i;
    }
    return axis;
}

inline bool manageable_power(const mpz_class& power, size_t terms) {
    // The independent graph substitution expands an auxiliary power into
    // at most binom(power+terms-1,terms-1) sparse terms before cancellation.
    mpz_class count = 1;
    for (unsigned long j = 1; j < terms; ++j) {
        count *= power + j;
        count /= j;
        if (count > 4096) return false;
    }
    return true;
}

inline void proposals(const Variables& names, const std::string& auxiliary,
                      const Polynomial& input, const Polynomial& face,
                      Json::Value& result) {
    const auto factor = repeated_face(face);
    if (!factor) return;
    Polynomial remainder = input;
    for (const auto& [exponent, coefficient] : face) {
        auto position = remainder.find(exponent);
        if (position == remainder.end() || position->second != coefficient) return;
        remainder.erase(position);
    }
    // The pure terms of the remainder survive in F before graph substitution.
    Powers intercepts(names.size());
    for (const auto& term : remainder) if (auto axis = pure_axis(term.first)) {
        auto& current = intercepts[*axis];
        if (!current || term.first[*axis] < current) current = term.first[*axis];
    }
    if (std::find(intercepts.begin(), intercepts.end(), 0) != intercepts.end()) return;

    std::vector<bool> provided(names.size(), false);
    for (const auto& term : factor->binomial)
        if (auto axis = pure_axis(term.first)) provided[*axis] = true;
    Polynomial filler;
    mpq_class next = 1;
    for (size_t i = 0; i < names.size(); ++i) if (!provided[i]) {
        Powers axis(names.size());
        axis[i] = intercepts[i];
        filler.emplace(std::move(axis), next++);
    }

    // For an initial graph with two or more terms, wt(w) is at least
    // m=min wt(b,D). On a monomial attaining m a pure intercept gives
    // wt(F) <= m * min_i(intercept_i/exponent_i). Therefore q greater
    // than every such ratio makes w^(q-1)*B strictly higher on that orbit.
    mpq_class bound = 0;
    const std::array<const Polynomial*, 2> base_terms{&factor->binomial, &filler};
    for (const Polynomial* terms : base_terms)
        for (const auto& term : *terms) {
            std::optional<mpq_class> lowest;
            for (size_t i = 0; i < names.size(); ++i) if (term.first[i] != 0) {
                mpq_class ratio(intercepts[i], term.first[i]);
                ratio.canonicalize();
                if (!lowest || ratio < *lowest) lowest = ratio;
            }
            bound = std::max(bound, *lowest);
        }
    mpz_class exponent = bound.get_num() / bound.get_den() + 1;
    exponent = std::max(exponent, mpz_class(2));
    if (!manageable_power(factor->power, 1 + filler.size()) ||
        !manageable_power(exponent, factor->binomial.size() + filler.size()) ||
        !manageable_power(factor->power, factor->binomial.size() + filler.size())) return;

    auto show = [&](const Polynomial& f) { return local::format_polynomial(f, names); };
    const std::string shift = filler.empty() ? auxiliary : auxiliary + "+(" + show(filler) + ")";
    const std::string graph = shift + "-(" + show(factor->binomial) + ")";
    Json::Value candidate(Json::objectValue);
    candidate["auxiliary"] = auxiliary;
    candidate["equation"] = graph;
    candidate["divisor"] = "(" + show(remainder) + ")+(" + show(factor->multiplier) +
        ")*(" + shift + ")^" + std::to_string(factor->power) +
        "+" + auxiliary + "^" + mpz_class(exponent - 1).get_str() + "*(" + graph + ")";
    result.append(std::move(candidate));
}

inline Json::Value propose(const Json::Value& request) {
    Variables names;
    for (const auto& entry : request["variables"]) names.push_back(entry.asString());
    Json::Value result(Json::arrayValue);
    if (names.size() < 2 || names.size() >= 8 || !request["faces"].isArray()) return result;
    std::string auxiliary = "w";
    for (unsigned i = 1; std::find(names.begin(), names.end(), auxiliary) != names.end(); ++i)
        auxiliary = "w" + std::to_string(i);
    const Polynomial input = local::parse_expression(names, request["polynomial"].asString());
    std::set<Polynomial> seen;
    for (const auto& expression : request["faces"]) {
        const auto face = local::parse_expression(names, expression.asString());
        if (seen.insert(face).second) proposals(names, auxiliary, input, face, result);
    }
    return result;
}
} // namespace atlas::toric_graph_candidates
