#pragma once

#include "local_basis.hpp"

namespace atlas::local {
struct PowerCoordinates {
    Basis generators;
    std::vector<mpq_class> point;
    Powers degrees;
};
std::optional<PowerCoordinates> point_power_coordinates(const Basis& ideal, const std::vector<mpq_class>& point);
}
