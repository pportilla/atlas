#include "monodromy_transverse.hpp"
#include "local_point_quotient.hpp"
#include "primary_decomposition.hpp"
#include <algorithm>

namespace atlas::monodromy {
namespace {
using sparse_json::Terms;

bool generically_nonzero(const Terms& f, size_t u, size_t v) {
    return std::any_of(f.begin(), f.end(), [&](const auto& term) {
        return term.second != 0 && term.first[u] == 0 && term.first[v] == 0;
    });
}

// Along A=0 in each transverse plane, D=A_v d/du - A_u d/dv is a
// nowhere-zero tangent field wherever the transverse gradient of A is nonzero.
// Thus the first D^m B nonzero along the parameter space gives the contact.
// Keeping the parameters symbolic certifies the same order on a dense open.
mpz_class generic_contact(const Terms& a, const Terms& b, size_t width, size_t u, size_t v) {
    const auto da = local::partial_derivatives(a, width);
    if (!generically_nonzero(da[u], u, v) && !generically_nonzero(da[v], u, v)) return 0;
    auto derivatives = local::partial_derivatives(b, width);
    if (!generically_nonzero(derivatives[u], u, v) &&
        !generically_nonzero(derivatives[v], u, v)) return 0;
    for (mpz_class order = 1; ; ++order) {
        auto tangent = primary::multiply(da[v], derivatives[u]);
        for (const auto& [e, c] : primary::multiply(da[u], derivatives[v])) {
            if ((tangent[e] -= c) == 0) tangent.erase(e);
        }
        if (tangent.empty()) return 0;
        if (generically_nonzero(tangent, u, v)) return order;
        derivatives = local::partial_derivatives(tangent, width);
    }
}
}

Json::Value transverse_contacts(const Terms& f, size_t width) {
    Json::Value rows(Json::arrayValue);
    if (width < 3 || width > 8 || f.empty()) return rows;
    try {
        // The shared factorization budget also checks the input term count
        // and total degree before asking FLINT for the rational factors.
        primary::CertificateBudget budget(16000);
        const auto factors = primary::factors(f, width);
        if (factors.size() < 2) return rows;
        for (size_t u = 0; u < width; ++u) for (size_t v = u + 1; v < width; ++v) {
            std::vector<const Terms*> sheets;
            for (const auto& [factor, multiplicity] : factors) {
                if (generically_nonzero(factor, u, v)) continue;
                if (multiplicity != 1 || sheets.size() == 2) {
                    sheets.clear();
                    break;
                }
                sheets.push_back(&factor);
            }
            if (sheets.size() != 2) continue;
            if (sheets[1]->size() < sheets[0]->size()) std::swap(sheets[0], sheets[1]);
            const auto order = generic_contact(*sheets[0], *sheets[1], width, u, v);
            if (order < 2) continue;

            // Straighten the first sheet; the second is its graph at distance
            // w^m times a unit. Completing the square gives X^2+Y^(2m)
            // times a unit, absorbed by weighted rescaling. The coordinates
            // depend analytically on the remaining parameters. Its zeta in
            // Atlas conventions is (1-T^(2m))/(1-T^2).
            Json::Value row;
            row["model"] = "two smooth branches";
            row["axes"].append(Json::UInt64(u));
            row["axes"].append(Json::UInt64(v));
            row["contact_order"] = order.get_str();
            row["factors"][0].append("2");
            row["factors"][0].append("-1");
            row["factors"][1].append(mpz_class(2 * order).get_str());
            row["factors"][1].append("1");
            rows.append(std::move(row));
        }
    } catch (const std::runtime_error&) {
        // This optional shortcut only returns the rows already certified.
        // The local Bernstein–Sato test handles every outstanding order.
    }
    return rows;
}
}
