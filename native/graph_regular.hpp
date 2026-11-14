#pragma once

#include <cstddef>
#include <istream>

namespace atlas {
// Certify that the last row acts injectively on the nonzero quotient by the
// preceding equations. A false result leaves the ordinary saturation in charge.
bool graph_regular(std::istream& rows, std::size_t width, std::size_t equation_count);
}
