// Ideals of A/B. Representatives are normal modulo B; testing membership and
// computing a standard basis use the inverse image of the ideal in A.
#include "quotient_ring.hpp"
#include "ideal_arithmetic.hpp"
#include "ideal_powers.hpp"
#include "local_io.hpp"
#include <algorithm>
#include <sstream>

namespace atlas::quotient_ring {
Basis Ring::project_generators(Basis rows) const {
    for (auto& f : rows) f = normal(f);
    return rows;
}

Ring Ring::inverse_image(const Basis& rows) const {
    return Ring(ideal_arithmetic::sum(basis_, rows), width_, lex_);
}

Basis Ring::ideal_basis(const Basis& rows) const {
    auto preimage = inverse_image(rows);
    Basis basis;
    // A normal representative of an ideal member has a leading monomial outside
    // in(B), so its divisor in the preimage basis must also lie outside in(B).
    // Nonleading terms of these reduced basis rows are already normal modulo B.
    for (auto& f : preimage.basis_)
        if (standard(local::global_head(f, lex_))) basis.push_back(std::move(f));
    return basis;
}

Basis Ring::ideal_product(const Basis& left, const Basis& right) const {
    auto products = ideal_arithmetic::product(left, right, width_);
    return ideal_arithmetic::sum(project_generators(std::move(products)), {});
}

Basis Ring::ideal_power(const Basis& rows, mpz_class exponent) const {
    if (exponent < 0) throw InvalidInput("The ideal exponent must be nonnegative.");
    if (zero_) return {};
    if (basis_.empty()) return ideal_powers::power(rows, width_, lex_, exponent, false);
    if (!exponent) return {Polynomial{{Powers(width_), 1}}};
    auto factor = ideal_basis(rows);
    if (factor.empty() || exponent == 1) return factor;
    if (factor.size() == 1) return ideal_basis({power(factor.front(), exponent)});
    // Reduce intermediate ideals so nilpotents disappear during powering and
    // repeated products of equivalent generators do not grow exponentially.
    std::optional<Basis> result;
    while (exponent > 0) {
        if (mpz_odd_p(exponent.get_mpz_t())) {
            result = result ? ideal_basis(ideal_product(*result, factor)) : factor;
            if (result->empty()) return {};
        }
        exponent >>= 1;
        if (exponent > 0) {
            factor = ideal_basis(ideal_product(factor, factor));
            if (factor.empty()) return {};
        }
    }
    return std::move(*result);
}

Json::Value Ring::ideal_operation(const std::string& operation, Basis left, Basis right,
                                  const Polynomial& query, const mpz_class& exponent,
                                  const std::vector<std::string>& names) const {
    left = project_generators(std::move(left));
    if (operation == "quotient-ring")
        return inverse_image(left).render(std::monostate{}, names);

    Json::Value result = render(std::monostate{}, names);
    Json::Value relations;
    result.removeMember("relations", &relations);
    result["ring"]["relations"] = std::move(relations);
    if (operation == "normal-form" || operation == "membership") {
        auto remainder = left.empty() ? normal(query) : inverse_image(left).normal(query);
        if (operation == "membership") {
            result["type"] = "boolean";
            result["value"] = remainder.empty();
        } else {
            result["type"] = "quotient-element";
            result["representative"] = local::format_polynomial(remainder, names);
        }
        return result;
    }

    Basis rows;
    if (operation == "ideal") rows = std::move(left);
    else if (operation == "ideal-sum") rows = ideal_arithmetic::sum(std::move(left), project_generators(std::move(right)));
    else if (operation == "ideal-product") rows = ideal_product(left, project_generators(std::move(right)));
    else if (operation == "ideal-power") rows = ideal_power(left, exponent);
    else if (operation == "groebner") rows = ideal_basis(left);
    else throw InvalidInput("Unsupported operation on a quotient ideal.");

    result["type"] = "quotient-ideal";
    Json::Value generators(Json::arrayValue);
    for (const auto& f : rows) generators.append(local::format_polynomial(f, names));
    result["generators"] = std::move(generators);
    return result;
}

namespace {
// Restrict the live-session path to small binomials in at most two variables.
// This bounds reduction and the bivariate preimage bases used in membership.
// Larger inputs use the cancellable worker.
bool small_polynomial(const Polynomial& f) {
    if (f.size() > 2) return false;
    for (const auto& [powers, c] : f) {
        unsigned long degree = 0;
        for (const auto& k : powers) {
            if (k < 0 || k > 6) return false;
            degree += k.get_ui();
        }
        if (degree > 6 || mpz_sizeinbase(mpq_numref(c.get_mpq_t()), 2) > 128 ||
            mpz_sizeinbase(mpq_denref(c.get_mpq_t()), 2) > 128) return false;
    }
    return true;
}
bool small_rows(const Basis& rows) {
    return std::all_of(rows.begin(), rows.end(), small_polynomial);
}
}

std::string ideal_request(std::istream& input, bool prepared, bool bounded) {
    Json::Value result;
    try {
        std::string coordinates, order, operation, exponent, counts;
        if (!std::getline(input, coordinates) || !std::getline(input, order) || !std::getline(input, operation) ||
            !std::getline(input, exponent) || !std::getline(input, counts) || (order != "lex" && order != "grevlex"))
            throw InvalidInput("Incomplete quotient ideal request.");
        std::istringstream variables(coordinates), sizes(counts);
        std::vector<std::string> names;
        for (std::string name; variables >> name;) names.push_back(std::move(name));
        long relations_count, left_count, right_count;
        std::string extra;
        if (!(sizes >> relations_count >> left_count >> right_count) || sizes >> extra ||
            relations_count < 0 || left_count < 0 || right_count < 0)
            throw InvalidInput("Expected three nonnegative quotient ideal generator counts.");
        auto read_row = [&]() {
            std::string row;
            if (!std::getline(input, row)) throw InvalidInput("Missing quotient ideal polynomial.");
            return local::decode_row(row, names.size());
        };
        auto read_rows = [&](long count) {
            Basis rows;
            for (long i = 0; i < count; ++i) rows.push_back(read_row());
            return rows;
        };
        auto relations = read_rows(relations_count);
        auto left = read_rows(left_count);
        auto right = read_rows(right_count);
        Polynomial query;
        if (operation == "membership" || operation == "normal-form") query = read_row();
        if (input.peek() != std::char_traits<char>::eof()) throw InvalidInput("Unexpected quotient ideal operands.");
        const auto power = parse_exponent(exponent);
        if (bounded && (names.size() > 2 || relations.size() > 3 ||
                        left.size() > 4 || right.size() > 4 || power > 2 || !small_rows(relations) ||
                        !small_rows(left) || !small_rows(right) || !small_polynomial(query))) {
            result["deferred"] = true;
        } else {
            auto ring = prepared ? Ring::from_basis(std::move(relations), names.size(), order == "lex")
                                 : Ring(relations, names.size(), order == "lex");
            result = ring.ideal_operation(operation, std::move(left), std::move(right), query, power, names);
        }
    } catch (const std::invalid_argument& error) {
        result["error"]["code"] = "invalid_request";
        result["error"]["message"] = error.what();
    }
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer, result) + "\n";
}
}
