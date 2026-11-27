#pragma once

#include "local_basis.hpp"

namespace atlas::local {
// Return the first nonzero degree in the completed local ring of the smooth
// frame at the given rational point. A bounded zero jet gives no exact order.
std::optional<unsigned> smooth_point_order(const Basis& frame, const std::vector<mpq_class>& point,
                                         const Polynomial& polynomial, unsigned max_degree);

// Coefficients in the prescribed complementary free parameters, through degree inclusive.
std::optional<Polynomial> smooth_point_jet(const Basis& frame, const std::vector<mpq_class>& point,
                                           const Polynomial& f, const std::vector<bool>& dependent,
                                           unsigned degree, size_t work, void (*check_time)());

// Hold the other prescribed local parameters fixed. A zero jet through d
// implies that every regular multiple of d/d(axis), iterated d times, gives
// zero at the point. False leaves the caller's exact contact test in force.
bool smooth_point_direction_zero(const Basis& frame, const std::vector<mpq_class>& point,
                                 const Polynomial& polynomial, const std::vector<size_t>& coordinates,
                                 size_t axis, unsigned degree);
}
