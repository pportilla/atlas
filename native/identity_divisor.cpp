// Check the factorization of a divisor on the germ of an identity chart.
#include "rational_sparse_json.hpp"
#include "branch_order.hpp"
#include "projection_coefficients.hpp"
#include "local_point_basis.hpp"
#include "sparse_flint.hpp"
#include "polynomial_factors.hpp"
#include <flint/fmpq.h>
#include <set>

namespace {
using namespace atlas::sparse_json;
using Factors = std::map<Terms, Z>;

struct Factorization {
    fmpq_mpoly_factor_t value;
    const fmpq_mpoly_ctx_struct* ctx;
    explicit Factorization(const fmpq_mpoly_ctx_struct* c) : ctx(c) { fmpq_mpoly_factor_init(value, ctx); }
    ~Factorization() { fmpq_mpoly_factor_clear(value, ctx); }
};
Factors active_factors(const Terms& f, atlas::local::SparseLexRing& ring) {
    const size_t width = fmpq_mpoly_ctx_nvars(ring.ctx);
    if (f.empty()) throw std::invalid_argument("Zero divisor equation.");
    // A coordinate monomial times a unit needs no global factorization,
    // even if the unit itself has huge degree.
    Powers min = f.begin()->first;
    for (const auto& [e, c] : f) for (size_t i = 0; i < width; ++i) min[i] = std::min(min[i], e[i]);
    Factors active;
    if (f.count(min)) {
        for (size_t i = 0; i < width; ++i) if (min[i]) {
            Powers axis(width);
            axis[i] = 1;
            active[{{axis, 1}}] = min[i];
        }
        return active;
    }
    atlas::PolyValue poly(ring.ctx);
    ring.write(poly.p, f);
    Factorization factors(ring.ctx);
    atlas::ordered_factors(factors.value, poly.p, ring.ctx);
    for (slong i = 0; i < factors.value->num; ++i) {
        const auto row = ring.read(factors.value->poly + i);
        if (!row.count(Powers(width))) fmpz_get_mpz(active[row].get_mpz_t(), factors.value->exp + i);
    }
    return active;
}

// Archived branch labels sort by first-axis degree, multiplicity, then
// descending nested coefficient lists with missing powers treated as zeros.
int coefficient_order(const Terms& a, const Terms& b, size_t width) {
    if (!width) {
        Q left = a.empty() ? Q(0) : a.begin()->second;
        Q right = b.empty() ? Q(0) : b.begin()->second;
        return cmp(left, right);
    }
    auto split = [width](const Terms& terms) {
        Z degree = terms.empty() ? Z(width == 1 ? -1 : 0) : terms.rbegin()->first.front();
        std::map<Z, Terms> offsets;
        for (const auto& [e, c] : terms)
            offsets[degree - e.front()].emplace(Powers(e.begin() + 1, e.end()), c);
        return std::make_pair(degree, std::move(offsets));
    };
    auto [degree_a, left] = split(a);
    auto [degree_b, right] = split(b);
    const Z limit = std::min(degree_a, degree_b);
    std::set<Z> offsets;
    for (const auto& entry : left) if (entry.first <= limit) offsets.insert(entry.first);
    for (const auto& entry : right) if (entry.first <= limit) offsets.insert(entry.first);
    for (const auto& k : offsets) {
        const int order = coefficient_order(left[k], right[k], width - 1);
        if (order) return order;
    }
    return cmp(degree_a, degree_b);
}
} // namespace
namespace atlas {
std::vector<size_t> archived_branch_order(const std::vector<std::pair<local::Polynomial, mpz_class>>& factors,
                                          size_t width) {
    std::vector<size_t> indices;
    for (size_t i = 0; i < factors.size(); ++i) indices.push_back(i);
    std::stable_sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        const auto& [a, m] = factors[i];
        const auto& [b, n] = factors[j];
        const auto& da = a.rbegin()->first[0];
        const auto& db = b.rbegin()->first[0];
        if (da != db) return da < db;
        if (m != n) return m < n;
        return coefficient_order(a, b, width) < 0;
    });
    return indices;
}
} // namespace atlas
namespace {
Json::Value branch_order(const Json::Value& input, size_t width) {
    std::vector<std::pair<Terms, Z>> factors;
    for (const auto& row : input) {
        auto p = read_terms(row[0], width);
        if (p.empty() || width == 0) throw std::invalid_argument("Expected nonconstant branch equation.");
        factors.emplace_back(std::move(p), Z(row[1].asString(), 10));
    }
    Json::Value result(Json::arrayValue);
    for (size_t i : atlas::archived_branch_order(factors,width)) result.append(Json::UInt64(i));
    return result;
}
Json::Value weighted_orders(const Json::Value& request, size_t width) {
    Powers totals(width);
    for (const auto& record : request["rows"]) {
        const Z multiplicity(record[0].asString(), 10);
        if (multiplicity < 1) throw std::invalid_argument("Invalid divisor multiplicity.");
        auto orders = read_exponents(record[1], width);
        for (size_t i = 0; i < width; ++i) totals[i] += multiplicity * orders[i];
    }
    return strings(totals);
}
// Preserve the ordered display of generators for a product of ideal powers.
// Repeated indices enumerate unordered selections within a single factor.
Json::Value strict_product_generators(const Json::Value& request) {
    std::vector<std::string> products{request["unit"].asString() == "1" ? "" :
                                      "(" + request["unit"].asString() + ")"};
    for (const auto& component : request["components"]) {
        const auto& generators = component["ideal"];
        Z multiplicity(component["multiplicity"].asString(), 10);
        if (multiplicity <= 0 || !multiplicity.fits_ulong_p()) throw std::invalid_argument("Invalid ideal power.");
        if (generators.empty()) return Json::Value(Json::arrayValue);
        std::vector<Json::ArrayIndex> choice(multiplicity.get_ui());
        std::vector<std::string> powers;
        for (;;) {
            std::string product;
            for (auto i : choice) {
                if (!product.empty()) product += '*';
                product += "(" + generators[i].asString() + ")";
            }
            powers.push_back(std::move(product));
            size_t k = choice.size();
            while (k && choice[k-1] == generators.size()-1) --k;
            if (!k) break;
            const auto next = choice[k-1] + 1;
            std::fill(choice.begin()+k-1, choice.end(), next);
        }
        std::vector<std::string> expanded;
        for (const auto& left : products) for (const auto& right : powers)
            expanded.push_back(left.empty() ? right : left + "*" + right);
        products = std::move(expanded);
    }
    Json::Value result(Json::arrayValue);
    for (const auto& product : products) result.append(product.empty() ? "1" : product);
    return result;
}
Json::Value verify(const Json::Value& request) {
    size_t width = request["width"].asUInt64();
    const auto op = request["operation"].asString();
    if (op == "strict_product_generators") return strict_product_generators(request);
    if (op == "weight_profile") {
        std::set<Z> distinct;
        for (const auto& item : request["weights"]) distinct.emplace(item.asString(), 10);
        Json::Value out;
        out["repeated"] = !distinct.empty() && *distinct.rbegin() > 1;
        out["uniform"] = distinct.size() == 1;
        return out;
    }
    if (op == "projection_coefficients") return atlas::projection_coefficients(request);
    if (op == "branch_order") return branch_order(request["parts"], width);
    if (op == "weighted_orders") return weighted_orders(request, width);
    if (op == "origin_indices") {
        Json::Value indices(Json::arrayValue);
        const Powers origin(width);
        const auto& equations = request["equations"];
        for (Json::ArrayIndex i = 0; i < equations.size(); ++i) {
            const auto terms = read_terms(equations[i], width);
            const auto value = terms.find(origin);
            if (value == terms.end() || value->second == 0) indices.append(i);
        }
        return indices;
    }
    if (op == "reduced_at_origin") {
        const auto terms = read_terms(request["terms"], width);
        Powers exponent(width);
        auto nonzero = [&] {
            const auto it = terms.find(exponent);
            return it != terms.end() && it->second != 0;
        };
        if (nonzero()) return true;  // A unit in the local ring.
        for (size_t i = 0; i < width; ++i) {
            exponent[i] = 1;
            if (nonzero()) return true;  // A nonzero differential proves smoothness.
            exponent[i] = 0;
        }
        atlas::local::SparseLexRing ring(width);
        atlas::PolyValue source(ring.ctx), support(ring.ctx), repeated(ring.ctx);
        ring.write(source.p, terms);
        atlas::rational_squarefree_part(support.p, source.p, ring.ctx);
        // The ratio by the squarefree support is a unit in the local ring
        // precisely when no repeated geometric component passes through zero.
        if (!fmpq_mpoly_divides(repeated.p, source.p, support.p, ring.ctx))
            throw std::logic_error("The squarefree support must divide the input.");
        std::vector<ulong> origin(width, 0);
        fmpq_mpoly_get_coeff_fmpq_ui(ring.coefficient, repeated.p, origin.data(), ring.ctx);
        return !fmpq_is_zero(ring.coefficient);
    }
    if (op == "transverse_germ_factors") {
        atlas::local::SparseLexRing ring(width);
        const auto active = active_factors(read_terms(request["terms"], width), ring);
        atlas::local::Basis equations;
        for (const auto& factor : active) equations.push_back(factor.first);
        if (!atlas::local::independent_at_origin(equations, width)) return {};
        std::vector<std::pair<Terms, Z>> factors(active.begin(), active.end());
        Json::Value result(Json::arrayValue);
        for (auto i : atlas::archived_branch_order(factors, width)) {
            Json::Value row(Json::arrayValue);
            row.append(term_list(factors[i].first));
            row.append(factors[i].second.get_str());
            result.append(row);
        }
        return result;
    }
    if (op != "certify") throw std::invalid_argument("Unknown divisor operation.");
    Terms f = read_terms(request["terms"], width);
    if (f.empty()) return false;
    std::vector<std::pair<Terms, Z>> parts;
    Z expected_order = 0;
    for (const auto& item : request["parts"]) {
        Z n(item[1].asString(), 10);
        if (n <= 0) return false;
        parts.emplace_back(read_terms(item[0], width), n);
        expected_order += n;
    }
    if (request.get("intrinsic", false).asBool() && (parts.size() != 1 || parts[0].second != 1))
        return false;
    if (parts.size() == 1 && parts[0].second == 1 && parts[0].first == f) return true;
    bool reaches_order = false;
    for (const auto& [e, c] : f) {
        Z degree = 0;
        for (const auto& k : e) degree += k;
        if (degree < expected_order) return false;
        if (degree == expected_order) reaches_order = true;
    }
    if (!reaches_order) return false;
    atlas::local::SparseLexRing ring(width);
    auto active = active_factors(f, ring);
    for (const auto& [g, n] : parts) {
        auto prime = active.find(g);
        if (prime == active.end()) {
            auto factors = active_factors(g, ring);
            if (factors.size() != 1 || factors.begin()->second != 1) return false;
            prime = active.find(factors.begin()->first);
        }
        if (prime == active.end() || prime->second != n) return false;
        active.erase(prime);
    }
    return active.empty();
}
} // namespace
extern "C" const char* atlas_identity_divisor(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, verify);
}
