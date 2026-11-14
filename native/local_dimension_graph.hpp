#pragma once

#include "local_basis.hpp"

namespace atlas::local {
int dimension_from_point_graph(const Basis& generators, const std::vector<mpq_class>& point,
                               bool prefer_modular);
}
