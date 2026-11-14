#pragma once
#include "local_basis.hpp"

namespace atlas {

// Historical factor ordering for the labels on resolution records.
std::vector<size_t> archived_branch_order(
    const std::vector<std::pair<local::Polynomial, mpz_class>>& factors, size_t width);

}
