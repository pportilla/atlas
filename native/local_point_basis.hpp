#pragma once

#include "local_basis.hpp"
#include <utility>

namespace atlas::local {
using PointRow = std::vector<mpq_class>;
bool use_sparse_point_basis(const Basis& ideal, const PointRow& point);
bool nonzero_at_point(const Polynomial& f, const PointRow& point);
bool independent_at_origin(const Basis& equations, size_t width);
std::optional<Polynomial> bounded_initial_at_point(const Polynomial& f, const PointRow& point);
mpz_class order_at_point(const Polynomial& f, const PointRow& point);
std::pair<Powers, mpq_class> leading_term_at_point(const Polynomial& f, const PointRow& point, bool lex);
Polynomial specialize_coordinate(const Polynomial& f, size_t axis, const mpq_class& coordinate);
Polynomial grouped_point_jet(const Polynomial& f, const PointRow& point, const mpz_class& cutoff,
                             const Powers& axis_bounds = {});
// A Taylor jet up to a nonzero rational scalar. Decline when even the
// relative coefficients of the row would require large integer powers.
std::optional<Polynomial> bounded_ideal_jet(const Polynomial& f, const PointRow& point,
                                            const mpz_class& cutoff);
Basis standard_basis_at_point(const Basis& ideal, const std::vector<std::string>& names,
                              const PointRow& point, bool lex, bool prefer_modular = false);
}
