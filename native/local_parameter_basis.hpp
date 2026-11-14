#pragma once

#include "local_point_basis.hpp"

namespace atlas::local {
std::optional<Basis> independent_parameter_basis(const Basis& generators,
    const std::vector<std::string>& names, const PointRow& point, bool lex, bool prefer_modular);
}
