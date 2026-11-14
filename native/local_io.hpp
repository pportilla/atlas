#pragma once

#include "local_basis.hpp"
#include <string_view>

namespace atlas::local {
Polynomial decode_row(std::string_view row, size_t width);
std::string encode_row(const Polynomial& f);
Polynomial parse_expression(const std::vector<std::string>& names, std::string source);
std::string format_polynomial(const Polynomial& f, const std::vector<std::string>& names);
bool has_point_unit(const Basis& generators, const std::vector<mpq_class>& point);
Basis project_monomials(const Basis& generators, const std::vector<mpq_class>& point);
std::optional<Polynomial> polynomial_gcd(const Basis& generators, size_t width);
bool polynomial_divides(const Polynomial& divisor, const Polynomial& f, size_t width);
Basis divide_generators(const Basis& generators, const Polynomial& divisor, size_t width);
Basis translate(const Basis& inputs, const std::vector<mpq_class>& shifts);
Basis translate_unbounded(const Basis& inputs, const std::vector<mpq_class>& shifts);
}
