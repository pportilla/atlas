#pragma once

#include <cstddef>
#include <istream>
#include <string>

namespace atlas {
std::string graph_centre_points(std::istream& input, size_t width);
std::string frame_points(std::istream& input, size_t width, size_t equations, size_t units);
std::string graph_centre_incidence(std::istream& input, size_t width, size_t rows,
                                  size_t parent_width, size_t points);
}
