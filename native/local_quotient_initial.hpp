#pragma once

#include "local_point_basis.hpp"

namespace atlas::local {
// Exact leading monomials when the initial forms are a regular sequence.
std::optional<std::vector<Powers>> regular_point_heads(const Basis& ideal,
        const std::vector<std::string>& names, const PointRow& point, bool lex, bool prefer_modular);
}
