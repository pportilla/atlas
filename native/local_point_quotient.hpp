#pragma once

#include "local_point_basis.hpp"
#include <stdexcept>

namespace atlas::local {
struct HypersurfaceOutside : std::invalid_argument {
    explicit HypersurfaceOutside(bool origin) : std::invalid_argument(
        origin ? "The hypersurface must pass through the origin." :
                 "The hypersurface must pass through the supplied point.") {}
};

struct PointQuotient {
    std::optional<mpz_class> dimension;
    std::vector<Powers> monomials;
    Basis derivatives;
};

PointQuotient quotient_at_point(const std::string& operation, const Basis& inputs,
    const std::vector<std::string>& names, const PointRow& point, bool lex,
    bool prefer_modular = false);

// Native quotient primitives. Exponents of standard monomials are in
// centered coordinates; equations and derivative polynomials are affine.
PointQuotient monomial_point_quotient(size_t width, const std::vector<Powers>& heads, bool include_basis);
Basis partial_derivatives(const Polynomial& f, size_t width);
Basis simplify_point_generators(const Basis& inputs, const PointRow& point);
int hypersurface_jet_order(const Polynomial& f, const PointRow& point);
Powers head_at_point(const Polynomial& f, const PointRow& point, bool lex);
}
