#pragma once
#include <json/value.h>

namespace atlas::components {
Json::Value decomposition_step(const Json::Value& request);
Json::Value projection_shape(const Json::Value& request);
Json::Value eligible_factors(const Json::Value& request);
Json::Value absolute_rational_factor(const std::string& row, size_t width);
}
