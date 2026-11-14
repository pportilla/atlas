#pragma once

#include <gmpxx.h>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace atlas::local {

using Powers = std::vector<mpz_class>;
using Polynomial = std::map<Powers, mpq_class>;
using Basis = std::vector<Polynomial>;

Powers local_head(const Polynomial& polynomial, bool lex);
Basis standard_basis(const Basis& generators, const std::vector<std::string>& names, bool lex, bool prefer_modular = false);
// A standard basis specialized for finite normal forms; decline if the
// finite-jet calculation cannot certify the result within its work budget.
std::optional<Basis> finite_standard_basis(const Basis& generators, const std::vector<std::string>& names, bool lex);

}
