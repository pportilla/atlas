#pragma once

#include "local_basis.hpp"

namespace atlas {
// The integral character lattice of all supports on a coordinate torus.
bool torus_lattice_compresses(size_t width, const local::Basis& equations);
mpz_class coordinate_torus_euler(size_t width, const local::Basis& equations);
}
