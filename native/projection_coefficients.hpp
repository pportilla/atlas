#pragma once
#include "rational_sparse_json.hpp"
#include "sparse_flint.hpp"

namespace atlas {

// Identify each projection coefficient in QQ or QQ[t]/(m) on a point or
// split fan chart. Terms outside the declared monomials must vanish in K.
inline Json::Value projection_coefficients(const Json::Value& request) {
    using namespace sparse_json;
    const size_t n = request["mapping"].size();
    const bool marked = !request["modulus"].isNull();
    if (!n || request["width"].asUInt64() != n + marked)
        throw std::invalid_argument("Invalid projection dimensions.");
    std::map<Powers, size_t> columns;
    if (request.isMember("pivot")) {
        size_t radial = request["pivot"].asUInt64();
        if (radial >= n) throw std::invalid_argument("Invalid point chart pivot.");
        for (size_t j = 0; j < n; ++j) {
            Powers powers(n);
            powers[radial] = powers[j] = 1;
            columns.emplace(std::move(powers), j);
        }
    } else {
        for (const auto& monomial : request["monomials"])
            columns.emplace(read_exponents(monomial,n), columns.size());
    }
    if (columns.size() != n) throw std::invalid_argument("Expected distinct chart monomials.");
    local::SparseLexRing ring(1);
    PolyValue modulus(ring.ctx), coefficient(ring.ctx), quotient(ring.ctx), remainder(ring.ctx);
    if (marked) {
        ring.write(modulus.p, read_terms(request["modulus"], 1));
        if (fmpq_mpoly_is_fmpq(modulus.p, ring.ctx))
            throw std::invalid_argument("Expected a coefficient field polynomial.");
    }
    Json::Value result(Json::arrayValue);
    for (const auto& image : request["mapping"]) {
        std::map<Powers, Terms> groups;
        for (const auto& [powers, value] : read_terms(image, n + marked)) {
            Powers base(powers.begin(), powers.begin() + n);
            groups[std::move(base)][{marked ? powers.back() : sparse_json::Z(0)}] = value;
        }
        Json::Value row(Json::arrayValue);
        for (size_t j = 0; j < n; ++j) row.append(Json::Value(Json::arrayValue));
        for (auto& [powers, terms] : groups) {
            if (marked) {
                ring.write(coefficient.p, terms);
                fmpq_mpoly_divrem(quotient.p, remainder.p, coefficient.p, modulus.p, ring.ctx);
                terms = ring.read(remainder.p);
            }
            if (terms.empty()) continue;
            const auto column = columns.find(powers);
            if (column == columns.end()) return {};
            row[static_cast<Json::ArrayIndex>(column->second)] = term_list(terms);
        }
        result.append(std::move(row));
    }
    return result;
}
} // namespace atlas
