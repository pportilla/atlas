// Exact, prefix-skipping decoding of the exhaustive resolution certificate syntax.
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "rational_sparse_json.hpp"
#include <numeric>

namespace {
using namespace atlas::sparse_json;
struct InvalidCode {};

struct Reader {
    std::string bits;
    size_t at = 0;
    explicit Reader(const Z& code) {
        if (code < 0) throw std::invalid_argument("Expected a nonnegative resolution candidate code.");
        Z sentinel = code+1;
        bits = sentinel.get_str(2).substr(1);
    }
    Z natural() {
        size_t start = at, last = at+(bits.size()-at+1)/2;
        while (at < last && bits[at] != '1') ++at;
        if (at == last) throw InvalidCode{};
        const size_t length = at-start+1;
        Z value(bits.substr(at,length),2);
        at += length;
        return value-1;
    }
    Z integer() {
        Z value = natural();
        if (mpz_odd_p(value.get_mpz_t())) return (value+1)/2;
        return -value/2;
    }
    size_t size(size_t minimum=0) {
        const Z count = natural()+minimum;
        if (count > bits.size()-at) throw InvalidCode{};
        return count.get_ui();
    }
};

std::string polynomial(Reader& reader, size_t variables, size_t projective, size_t field_degree, bool boundary) {
    Terms p;
    Z degree = -1;
    const size_t count = reader.size(1);
    bool mixed = false;
    for (size_t j = 0; j < count; ++j) {
        auto c = reader.integer();
        if (!c) throw InvalidCode{};
        Z index = 0;
        if (field_degree > 1) {
            index = reader.natural();
            if (index >= field_degree) throw InvalidCode{};
        }
        Powers e(variables+projective);
        for (auto& exponent : e) exponent = reader.natural();
        if (boundary) {
            Z total = std::accumulate(e.begin()+variables,e.end(),Z(0));
            if (degree >= 0 && degree != total) mixed = true;
            degree = total;
        }
        if (field_degree > 1) e.push_back(index);
        auto [it,fresh] = p.emplace(e,c);
        if (!fresh && (it->second += c) == 0) p.erase(it);
    }
    // Consume the full polynomial before rejecting a nonhomogeneous one.
    // Prefix skipping must retain the original exhaustive schedule.
    if (p.empty() || mixed) throw InvalidCode{};
    return atlas::local::encode_row(p);
}

Json::Value candidate(Reader& reader, size_t n) {
    Json::Value result;
    Z requested = reader.natural()+1;
    if (requested > 1 && requested > reader.bits.size()-reader.at) throw InvalidCode{};
    const size_t degree = requested.get_ui();
    if (degree > 1) {
        Powers coefficients;
        Terms relation;
        for (size_t k = 0; k < degree; ++k) {
            auto c = reader.integer();
            if (c) relation.emplace(Powers{k},c);
            coefficients.push_back(std::move(c));
        }
        auto root = reader.natural();
        if (root >= degree) throw InvalidCode{};
        relation.emplace(Powers{degree},1);
        const auto factors = atlas::primary::factors(relation,1);
        if (factors.size() != 1 || factors.front().second != 1) throw InvalidCode{};
        coefficients.emplace_back(1);
        std::reverse(coefficients.begin(),coefficients.end());
        result["coefficients"] = strings(coefficients);
        result["index"] = Json::UInt64(root.get_ui());
    }
    Json::Value center(Json::arrayValue), boundary(Json::arrayValue);
    const size_t generators = reader.size(1);
    for (size_t j=0; j<generators; ++j) center.append(polynomial(reader,n,0,1,false));
    const size_t components = reader.size();
    for (size_t j=0; j<components; ++j) {
        Json::Value rows(Json::arrayValue);
        const size_t size = reader.size(1);
        for (size_t k=0; k<size; ++k) rows.append(polynomial(reader,n,generators,degree,true));
        boundary.append(std::move(rows));
    }
    result["center"] = std::move(center);
    result["boundary"] = std::move(boundary);
    return result;
}

Json::Value decode(const Json::Value& request) {
    const auto n = request["width"].asUInt64();
    const auto op = request["operation"].asString();
    if (op == "first") {
        Z first = (Z(0b1111011) << (n+1))-2;
        return first.get_str(16);
    }
    if (op != "decode") throw std::invalid_argument("Unknown resolution search operation.");
    const Z code(request["code"].asString(),16);
    Reader reader(code);
    Json::Value out;
    try {
        auto data = candidate(reader,n);
        if (reader.at == reader.bits.size()) out["candidate"] = std::move(data);
    } catch (const InvalidCode&) {
        // Every completion of the consumed prefix fails the same test.
    }
    const size_t unread = reader.bits.size()-reader.at;
    Z next = ((((code+1) >> unread)+1) << unread)-1;
    out["next"] = next.get_str(16);
    return out;
}
} // namespace

extern "C" const char* atlas_resolution_candidate(const char* input) noexcept {
    return atlas::sparse_json::invoke(input,decode);
}
