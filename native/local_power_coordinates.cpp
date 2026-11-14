#include "local_dimension.hpp"
#include "local_power_coordinates.hpp"
#include <algorithm>

namespace atlas::local {

// In each nonzero coordinate, differences between exponents determine a
// possible power substitution. The common monomial removed from an equation
// is a unit at the point. x -> x^d is étale there over QQ, so local dimension
// agrees at p and at its image, even for nonreduced schemes.
std::optional<PowerCoordinates> point_power_coordinates(const Basis& ideal, const std::vector<mpq_class>& point) {
    const size_t width = point.size();
    Powers stride(width, 0);
    for (const auto& row : ideal) {
        if (row.empty()) continue;
        const auto& first = row.begin()->first;
        for (const auto& [e, coefficient] : row)
            for (size_t i = 0; i < width; ++i) if (point[i] != 0 && stride[i] != 1) {
                mpz_class delta = e[i] - first[i];
                mpz_gcd(stride[i].get_mpz_t(), stride[i].get_mpz_t(), delta.get_mpz_t());
            }
    }
    std::vector<size_t> axes;
    auto image = point;
    for (size_t i = 0; i < width; ++i) {
        if (stride[i] <= 1) continue;
        if (point[i] == -1) image[i] = mpz_odd_p(stride[i].get_mpz_t()) ? -1 : 1;
        else if (point[i] != 1) {
            // Choose the other sparse methods when the image coordinate
            // would itself require a large integer allocation.
            const size_t bits = std::max(mpz_sizeinbase(point[i].get_num_mpz_t(), 2),
                                         mpz_sizeinbase(point[i].get_den_mpz_t(), 2));
            if (stride[i] > 65536 / bits) continue;
            const auto power = stride[i].get_ui();
            mpz_pow_ui(image[i].get_num_mpz_t(), point[i].get_num_mpz_t(), power);
            mpz_pow_ui(image[i].get_den_mpz_t(), point[i].get_den_mpz_t(), power);
        }
        axes.push_back(i);
    }
    if (axes.empty()) return std::nullopt;
    Basis reduced;
    reduced.reserve(ideal.size());
    for (const auto& row : ideal) if (!row.empty()) {
        auto minimum = row.begin()->first;
        for (const auto& [e, coefficient] : row)
            for (auto i : axes) minimum[i] = std::min(minimum[i], e[i]);
        Polynomial image_row;
        for (const auto& [e, coefficient] : row) {
            auto exponent = e;
            for (auto i : axes) {
                exponent[i] -= minimum[i];
                mpz_divexact(exponent[i].get_mpz_t(), exponent[i].get_mpz_t(), stride[i].get_mpz_t());
            }
            image_row.emplace(std::move(exponent), coefficient);
        }
        reduced.push_back(std::move(image_row));
    }
    Powers degrees(width, 1);
    for (auto i : axes) degrees[i] = stride[i];
    return PowerCoordinates{std::move(reduced), std::move(image), std::move(degrees)};
}

std::optional<int> dimension_power_coordinates(const Basis& ideal,
        const std::vector<std::string>& names, const std::vector<mpq_class>& point,
        bool lex, bool prefer_modular) {
    auto cover = point_power_coordinates(ideal, point);
    if (!cover) return std::nullopt;
    return dimension_at_point(cover->generators, names, cover->point, lex, false, prefer_modular).dimension;
}
}
