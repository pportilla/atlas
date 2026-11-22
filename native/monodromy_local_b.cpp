// Decide weak monodromy for the supplied rational slopes at the origin.
#include "monodromy_local_b.hpp"
#include "monodromy_weyl.hpp"
#include "local_monomial.hpp"
#include <algorithm>
#include <optional>

namespace atlas::monodromy {
namespace {
using namespace sparse_json;
using monodromy_weyl::Annihilator;

// A coordinate monomial times a local unit has the local b-function
// product over all i and 1 <= j <= a_i of (s + j/a_i). A smooth
// equation is the same calculation with one exponent equal to 1.
std::optional<Powers> coordinate_model(const Terms& f) {
    const auto& first = f.begin()->first;
    if (std::all_of(f.begin(), f.end(), [&](const auto& term) {
            return local::monomial_divides(first, term.first);
        })) return first;
    for (const auto& [powers, coefficient] : f)
        if (local::total_degree(powers) == 1) return Powers{1};
    return std::nullopt;
}

bool monomial_root(const Powers& powers, const Q& alpha) {
    return alpha >= -1 && alpha < 0 &&
           std::any_of(powers.begin(), powers.end(), [&](const Z& e) {
               return e > 0 && mpz_divisible_p(e.get_mpz_t(), alpha.get_den_mpz_t());
           });
}
}  // namespace

Json::Value local_eigenvalues(const sparse_json::Terms& terms, std::size_t width,
                              const Json::Value& values) {
    using namespace sparse_json;
    if (!values.isArray()) throw std::invalid_argument("Expected an array of rational poles.");
    Terms f;
    for (const auto& [powers, c] : terms) {
        if (powers.size() != width) throw std::invalid_argument("Invalid polynomial dimension.");
        if (c != 0) f.emplace(powers, c);
    }
    if (f.empty()) throw std::invalid_argument("Monodromy requires a nonzero polynomial.");
    const auto model = coordinate_model(f);
    std::optional<Annihilator> annihilator;
    std::map<Q, Json::Value> by_residue;
    Json::Value result(Json::objectValue);
    for (const auto& value : values) {
        const std::string key = value.asString();
        Q slope(key, 10);
        if (slope.get_den() == 0) throw std::invalid_argument("Zero denominator in monodromy pole.");
        slope.canonicalize();
        Z remainder;
        mpz_fdiv_r(remainder.get_mpz_t(), slope.get_num_mpz_t(), slope.get_den_mpz_t());
        const Q residue(remainder, slope.get_den());
        auto [found, inserted] = by_residue.try_emplace(residue);
        if (inserted) {
            auto& decision = found->second;
            decision["root"] = Json::nullValue;
            decision["tested_roots"] = Json::arrayValue;
            // Saito, On microlocal b-function, Prop. 0.3 and Thm. 0.4:
            // all local roots lie in [-width,0), including the ordinary -1.
            // Malgrange--Kashiwara identifies their exponentials with the
            // union of Milnor cohomology eigenvalues in the germ.
            Q alpha = residue - 1;
            for (std::size_t shift = 0; shift < width; ++shift, --alpha) {
                decision["tested_roots"].append(alpha.get_str());
                if (!model && !annihilator) annihilator.emplace(f, width);
                if (model ? monomial_root(*model, alpha) : annihilator->local_root(alpha)) {
                    decision["root"] = alpha.get_str();
                    break;
                }
            }
        }
        result[key] = found->second;
    }
    return result;
}

}  // namespace atlas::monodromy
