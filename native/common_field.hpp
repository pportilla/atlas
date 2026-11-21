#pragma once
#include <gmpxx.h>
#include <json/json.h>
#include <vector>

namespace atlas {
struct AlgebraicNumber;
Json::Value common_field(const Json::Value& request);
// Mark a primitive generator for the field of the supplied nonrational
// algebraic values, and express each value in its ascending power basis.
Json::Value coefficient_field(const std::vector<AlgebraicNumber>& values,
                              std::vector<std::vector<mpq_class>>& images);
}
