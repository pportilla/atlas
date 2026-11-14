#pragma once

#include <gmpxx.h>
#include <map>
#include <string>
#include <vector>

namespace atlas::local {

using Powers = std::vector<unsigned long>;
using Polynomial = std::map<Powers, mpq_class>;
using Basis = std::vector<Polynomial>;

Powers local_head(const Polynomial& polynomial, bool lex);
Basis standard_basis(const Basis& generators, const std::vector<std::string>& names, bool lex);

}
