// Recognize two conjugate plane branches and give their actual equations over
// the splitting field of their tangent lines. The fan verifier checks the
// original divisor, the marked field and the Jacobian on the identity chart.
#include "local_io.hpp"
#include "polynomial_factors.hpp"
#include "primary_decomposition.hpp"
#include "rational_sparse_json.hpp"
#include <algorithm>

namespace {
using atlas::local::Polynomial;
using atlas::local::Powers;

mpq_class coefficient(const Polynomial& f, unsigned x, unsigned y) {
    const auto found = f.find({x, y});
    return found == f.end() ? mpq_class(0) : found->second;
}

Json::Value crossing(const Json::Value& request) {
    const size_t width = request.get("width", 2).asUInt();
    if (width < 2 || width > 8) return {};
    const auto source = atlas::local::decode_row(request["polynomial"].asString(), width);
    std::vector<size_t> axes;
    for (size_t j = 0; j < width; ++j) {
        if (std::any_of(source.begin(), source.end(),
                        [j](const auto& term) { return term.first[j] != 0; }))
            axes.push_back(j);
        if (axes.size() > 2) return {};
    }
    if (axes.size() != 2) return {};
    Polynomial input;
    for (const auto& [e, c] : source) input.emplace(Powers{e[axes[0]], e[axes[1]]}, c);
    const auto rational = atlas::primary::factors(input, 2);
    const Polynomial* support = nullptr;
    mpz_class multiplicity;
    for (const auto& [f, power] : rational) {
        if (f.count({0, 0})) continue;
        if (support) return {};
        support = &f;
        multiplicity = power;
    }
    if (!support || support->count({1, 0}) || support->count({0, 1})) return {};
    // Two conjugates have equal degree. Bound this trial to branches of
    // degree at most three; the general plane solver handles larger inputs.
    mpz_class degree = 0;
    for (const auto& term : *support) {
        const mpz_class total = term.first[0] + term.first[1];
        if (total > degree) degree = total;
    }
    if (degree != 2 && degree != 4 && degree != 6) return {};
    const auto leading = coefficient(*support, 2, 0);
    const auto mixed = coefficient(*support, 1, 1);
    const mpq_class discriminant = mixed*mixed -
        4*leading*coefficient(*support, 0, 2);
    if (mpz_perfect_square_p(discriminant.get_num_mpz_t()) &&
        mpz_perfect_square_p(discriminant.get_den_mpz_t())) return {};

    // If the reduced polynomial has two conjugate branches through the
    // rational origin with independent tangents, each is defined over
    // Q(sqrt(discriminant)): Galois acts on the branches as on their tangents.
    const Polynomial modulus{{{0, 0, 2}, 1}, {{0, 0, 0}, -discriminant}};
    Json::Value answer;
    auto append_branch = [&](const Polynomial& branch) {
        Polynomial extended;
        for (const auto& [e, c] : branch) {
            Powers powers(width + 1);
            powers[axes[0]] = e[0];
            powers[axes[1]] = e[1];
            powers.back() = e[2];
            extended.emplace(std::move(powers), c);
        }
        answer["branches"].append(atlas::local::encode_row(extended));
    };
    if (degree == 2) {
        // A zero leading coefficient would make the discriminant a square.
        // The monic branch equations are x + (B +/- sqrt(D))/(2A)*y.
        for (int sign : {-1, 1}) {
            Polynomial branch{{{1, 0, 0}, 1}, {{0, 1, 1}, sign/(2*leading)}};
            if (mixed != 0) branch[{0, 1, 0}] = mixed/(2*leading);
            append_branch(branch);
        }
    } else {
        Polynomial marked;
        for (const auto& [e, c] : *support) marked.emplace(Powers{e[0], e[1], 0}, c);
        // The rational irreducible is squarefree in characteristic zero.
        // Every geometric factor depends on x: Galois permutes them
        // transitively, and the tangent form has a nonzero x^2 coefficient.
        const auto factors = atlas::marked_section_factors(marked, modulus, 3, 0);
        if (factors.size() != 2) return {};
        for (const auto& factor : factors) append_branch(factor.first);
    }
    for (size_t i = 0; i < width; ++i) {
        Json::Value ray(Json::arrayValue);
        for (size_t j = 0; j < width; ++j) ray.append(int(i == j));
        answer["cone"].append(std::move(ray));
    }
    answer["modulus"] = atlas::local::encode_row({{{2}, 1}, {{0}, -discriminant}});
    answer["multiplicity"] = multiplicity.get_str();
    return answer;
}
} // namespace

extern "C" const char* atlas_conjugate_plane_crossing(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, crossing);
}
