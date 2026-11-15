#pragma once

#include <json/value.h>

namespace atlas {
// A positive certificate for the canonical curve centre on a smooth surface.
// Null means the general surface procedure must choose the centre.
Json::Value surface_centre_choice(const Json::Value& request);
}
