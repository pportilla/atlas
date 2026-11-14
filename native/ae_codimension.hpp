#pragma once

#include "local_basis.hpp"
#include <json/value.h>

namespace atlas::local {

// Extended right-left codimension of a polynomial map germ (QQ^n,0) -> (QQ^p,0).
Json::Value ae_codimension(const Basis& components, const std::vector<std::string>& names);

}
