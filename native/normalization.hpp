#pragma once

#include "local_basis.hpp"
#include <json/json.h>
#include <istream>

namespace atlas::normalization {
using local::Basis;
using local::Polynomial;
struct Fraction {
    Polynomial numerator, denominator;
};
struct Presentation {
    std::vector<std::string> names;
    Basis relations, images;
    std::vector<Fraction> fractions;
};

// The input to the specialized algorithms is a reduced basis of a prime ideal.
std::optional<Presentation> special(const Basis& prime, const std::vector<std::string>& names, bool lex);
// One finite birational extension. Its presentation may still need normalization.
std::optional<Presentation> divide_power_content(const Basis& prime, const std::vector<std::string>& names, bool lex);
Json::Value compute(const Basis& generators, const std::vector<std::string>& names, bool lex);
std::string operation(std::istream& input);
}
