#pragma once
#include "local_basis.hpp"

namespace atlas::ideal_arithmetic {
enum class GeneratorsKind { zero, unit, general };
enum class Combination { left, right, concatenate, pairwise };

// Both sparse inputs and persistent FLINT ideals use these same identities.
inline Combination combine(bool product, GeneratorsKind left, GeneratorsKind right) {
    if (left == GeneratorsKind::zero) return product ? Combination::left : Combination::right;
    if (right == GeneratorsKind::zero) return product ? Combination::right : Combination::left;
    if (left == GeneratorsKind::unit) return product ? Combination::right : Combination::left;
    if (right == GeneratorsKind::unit) return product ? Combination::left : Combination::right;
    return product ? Combination::pairwise : Combination::concatenate;
}

template<class Rows, class Iterator>
Rows concatenate_generators(Rows left, Iterator first_right, Iterator last_right) {
    left.insert(left.end(), first_right, last_right);
    return left;
}

struct UseGenerator {
    template<class Row> const Row& operator()(const Row& f) const { return f; }
};

// Both representations emit f*g in left-major order. A sparse backend may
// prepare each f once, then reuse it for the entire right-hand list.
template<class Left, class Right, class Multiply, class Prepare = UseGenerator>
void multiply_generator_pairs(const Left& left, const Right& right, Multiply multiply, Prepare prepare = {}) {
    for (const auto& row : left) {
        const auto& f = prepare(row);
        for (const auto& g : right) multiply(f, g);
    }
}

bool unit_generator(const local::Polynomial& f);
// Ordered pairwise polynomial products, retaining multiplicity, scale and zero rows.
local::Basis generator_products(const local::Basis& left, const local::Basis& right, size_t width);
// Unreduced generators over QQ, with empty zero ideals and singleton unit ideals.
local::Basis sum(local::Basis left, local::Basis right);
local::Basis product(const local::Basis& left, const local::Basis& right, size_t width);
local::Basis binary(local::Basis left, local::Basis right, size_t width, bool multiply);
}
