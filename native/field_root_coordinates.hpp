#pragma once

#include <gmpxx.h>
#include <optional>
#include <vector>

namespace atlas {
// Coordinates in Q[T]/(minimal), in ascending powers of T, for all roots
// of a squarefree rational polynomial. Empty optional means a root is outside.
std::optional<std::vector<std::vector<mpq_class>>> roots_in_power_basis(
    const std::vector<mpq_class>& minimal, const std::vector<mpq_class>& equation);
}
