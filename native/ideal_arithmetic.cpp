// Generators of a sum or product of two ideals in the same QQ polynomial ring.
#include "ideal_arithmetic.hpp"
#include "local_io.hpp"
#include "sparse_flint.hpp"
#include <json/json.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace atlas {
namespace {
using local::Basis;
using local::Polynomial;
using local::Powers;

Basis read_ideal(std::istream& in, size_t count, size_t width) {
    Basis rows;
    for (size_t i = 0; i < count; ++i) {
        std::string row;
        if (!std::getline(in, row)) throw std::invalid_argument("Missing ideal generator.");
        rows.push_back(local::decode_row(row, width));
    }
    return rows;
}

ideal_arithmetic::GeneratorsKind kind(const Basis& rows) {
    bool nonzero = false;
    for (const auto& f : rows) {
        if (ideal_arithmetic::unit_generator(f)) return ideal_arithmetic::GeneratorsKind::unit;
        nonzero |= !f.empty();
    }
    return nonzero ? ideal_arithmetic::GeneratorsKind::general : ideal_arithmetic::GeneratorsKind::zero;
}

Basis without_zeros(Basis rows) {
    rows.erase(std::remove_if(rows.begin(), rows.end(), [](const auto& f) { return f.empty(); }), rows.end());
    return rows;
}

Basis normalized(Basis rows, ideal_arithmetic::GeneratorsKind how) {
    if (how == ideal_arithmetic::GeneratorsKind::zero) return {};
    if (how == ideal_arithmetic::GeneratorsKind::unit) {
        const auto one = std::find_if(rows.begin(), rows.end(), ideal_arithmetic::unit_generator);
        return {Polynomial{{one->begin()->first, 1}}};
    }
    return without_zeros(std::move(rows));
}

// Translating all exponents preserves their ordering and creates no collisions.
Polynomial times_monomial(const Polynomial& f, const Polynomial& monomial) {
    const auto& [powers, coefficient] = *monomial.begin();
    Polynomial out;
    for (const auto& [e, c] : f) {
        Powers shifted(e);
        for (size_t j = 0; j < powers.size(); ++j) shifted[j] += powers[j];
        out.emplace_hint(out.end(), std::move(shifted), c * coefficient);
    }
    return out;
}
}

namespace ideal_arithmetic {
bool unit_generator(const Polynomial& f) {
    return f.size() == 1 && std::all_of(f.begin()->first.begin(), f.begin()->first.end(),
                                        [](const auto& k) { return k == 0; });
}

Basis sum(Basis left, Basis right) {
    const auto a = kind(left), b = kind(right);
    const auto action = combine(false,a,b);
    if (action == Combination::left) return normalized(std::move(left),a);
    if (action == Combination::right) return normalized(std::move(right),b);
    return without_zeros(concatenate_generators(std::move(left),std::make_move_iterator(right.begin()),
                                                 std::make_move_iterator(right.end())));
}

Basis generator_products(const Basis& left, const Basis& right, size_t width) {
    if (left.empty() || right.empty()) return {};
    Basis out;
    out.reserve(left.size() * right.size());
    const auto monomial = [](const auto& f) { return f.size() <= 1; };
    if (std::all_of(left.begin(), left.end(), monomial) || std::all_of(right.begin(), right.end(), monomial)) {
        multiply_generator_pairs(left,right,[&](const auto& f, const auto& g) {
            if (f.empty() || g.empty()) out.emplace_back();
            else out.push_back(f.size() == 1 ? times_monomial(g,f) : times_monomial(f,g));
        });
        return out;
    }
    local::SparseLexRing ring(width);
    std::vector<PolyValue> factors;
    for (const auto& f : right) ring.write(factors.emplace_back(ring.ctx).p, f);
    PolyValue factor(ring.ctx), multiple(ring.ctx);
    auto pack_left = [&](const Polynomial& row) -> const PolyValue& {
        ring.write(factor.p,row);
        return factor;
    };
    multiply_generator_pairs(left,factors,[&](const auto& f, const auto& g) {
        fmpq_mpoly_mul(multiple.p, f.p, g.p, ring.ctx);
        out.push_back(ring.read(multiple.p));
    },pack_left);
    return out;
}

Basis product(const Basis& left, const Basis& right, size_t width) {
    const auto a = kind(left), b = kind(right);
    const auto action = combine(true,a,b);
    if (action == Combination::left) return normalized(left,a);
    if (action == Combination::right) return normalized(right,b);
    return without_zeros(generator_products(left, right, width));
}

Basis binary(Basis left, Basis right, size_t width, bool multiply) {
    return multiply ? product(left,right,width) : sum(std::move(left),std::move(right));
}
} // namespace ideal_arithmetic

std::string ideal_arithmetic_request(std::istream& in, bool product) {
    std::string coordinates, order, sizes;
    if (!std::getline(in, coordinates) || !std::getline(in, order) ||
        (order != "lex" && order != "grevlex") || !std::getline(in, sizes))
        throw std::invalid_argument("Expected ideal coordinates, order and generator counts.");
    std::istringstream counts(sizes);
    long left_count, right_count;
    std::string extra;
    if (!(counts >> left_count >> right_count) || left_count < 0 || right_count < 0 || counts >> extra)
        throw std::invalid_argument("Expected two nonnegative generator counts.");
    std::vector<std::string> names;
    std::istringstream variables(coordinates);
    for (std::string name; variables >> name;) names.push_back(std::move(name));
    auto left = read_ideal(in, left_count, names.size());
    auto right = read_ideal(in, right_count, names.size());
    if (in.peek() != std::char_traits<char>::eof()) throw std::invalid_argument("Unexpected extra ideal generators.");

    Json::Value result;
    result["type"] = "ideal";
    auto& ring = result["ring"];
    ring["field"] = "QQ";
    ring["order"] = order;
    ring["variables"] = Json::Value(Json::arrayValue);
    for (const auto& name : names) ring["variables"].append(name);
    const auto generators = ideal_arithmetic::binary(std::move(left),std::move(right),names.size(),product);
    auto& rows = result["generators"];
    rows = Json::Value(Json::arrayValue);
    for (const auto& f : generators) rows.append(local::format_polynomial(f, names));
    thread_local const auto writer = [] {
        Json::StreamWriterBuilder options;
        options["indentation"] = "";
        return std::unique_ptr<Json::StreamWriter>(options.newStreamWriter());
    }();
    std::ostringstream text;
    writer->write(result,&text);
    text << '\n';
    return text.str();
}
} // namespace atlas
