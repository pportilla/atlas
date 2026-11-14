#pragma once

#include "local_basis.hpp"

namespace atlas::local {
Polynomial parse_expression(const std::vector<std::string>& names, std::string source);
std::string format_polynomial(const Polynomial& f, const std::vector<std::string>& names);
Basis translate(const Basis& inputs, const std::vector<mpq_class>& shifts,
                const std::vector<std::string>& names);
}
