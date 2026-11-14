#pragma once

#include "local_basis.hpp"

namespace atlas::local {
// Linear coordinate changes for three powers of distinct binary linear forms.
std::optional<Basis> normalized_surface_axes(const Basis& components);
}
