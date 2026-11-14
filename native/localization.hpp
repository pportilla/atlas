#pragma once

#include "local_basis.hpp"
#include <json/json.h>
#include <istream>
#include <stdexcept>

namespace atlas::localization {
using local::Basis;
using local::Polynomial;

struct Fraction {
    Polynomial numerator, denominator;
};

class InvalidInput : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

bool member_lex(const Polynomial& f, const Basis& lex_basis);
Basis contract(const Basis& generators, const Basis& prime_lex, size_t width);
std::optional<Basis> contract_factored(Basis rows, const Basis& prime_lex, size_t width);
std::optional<Basis> special_contraction(const Basis& generators, const Basis& prime_lex, size_t width);
Json::Value compute(const Basis& generators, const Basis& prime, const std::vector<std::string>& names,
                    bool lex, const std::optional<Fraction>& fraction = std::nullopt);
std::string operation(std::istream& input);
}
