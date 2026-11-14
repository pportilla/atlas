#pragma once

#include <gmpxx.h>
#include <optional>
#include <vector>

namespace atlas {
// A degree-d equation has precisely these roots in Q[T]/(minimal) if
// the d reduced coordinate rows are distinct and each satisfies it.
bool certify_power_basis_roots(const std::vector<mpq_class>& minimal,
    const std::vector<mpq_class>& equation, const std::vector<std::vector<mpq_class>>& roots);

// Coordinates in Q[T]/(minimal), in ascending powers of T, for all roots
// of a squarefree rational polynomial. Empty optional means a root is outside.
std::optional<std::vector<std::vector<mpq_class>>> roots_in_power_basis(
    const std::vector<mpq_class>& minimal, const std::vector<mpq_class>& equation);
}
