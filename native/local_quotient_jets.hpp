#pragma once

#include "local_point_basis.hpp"

namespace atlas::local {
// Each Taylor monomial carries a sparse sum of powers of rational coordinates.
// The numeric bases are determined by the point in both operations.
using SymbolicJet = std::map<Powers,Polynomial>;
std::vector<SymbolicJet> symbolic_point_jets(const Basis& ideal, const PointRow& point, unsigned cutoff);
std::vector<size_t> symbolic_pivot_columns(const PointRow& point, std::vector<std::vector<Polynomial>>&& rows);
std::optional<std::vector<Powers>> finite_point_jet_heads(const Basis& ideal, const PointRow& point, bool lex);
}
