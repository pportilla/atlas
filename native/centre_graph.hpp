// A global graph certificate for an ordered list of affine centre generators.
#pragma once

#include "groebner.hpp"
#include <flint/nmod_mat.h>

namespace atlas {

// Peel a row a*x_j + h with a nonzero scalar a when x_j is absent from
// every other remaining row and from h. Reversing the peel order gives a
// triangular polynomial automorphism from the free variables and the given
// rows back to the original coordinates. E.g. peel z-x*y before y-x^2.
// No degree-dependent system or substitutions need to be constructed.
template<class PolynomialAt>
bool triangular_centre(const Ring& ring, size_t count, PolynomialAt polynomial) {
    if (count == 0 || count > static_cast<size_t>(ring.n)) return false;
    static_assert(MAX_VARIABLES < 64);
    u64 used_by[MAX_VARIABLES]{}, solved_in[MAX_VARIABLES]{};
    long exponents[MAX_VARIABLES];
    size_t remaining_terms = 512;
    for (size_t row = 0; row < count; ++row) {
        check_deadline();
        const auto& poly = polynomial(row);
        if (poly.empty() || poly.size() > remaining_terms) return false;
        remaining_terms -= poly.size();
        u64 linear = 0, other = 0;
        for (size_t i = 0; i < poly.size(); ++i) {
            const auto* monomial = poly.m.data() + i * ring.stride;
            if (monomial[0] > 32) return false;
            ring.unpack(monomial, exponents);
            for (int v = 0; v < ring.n; ++v) {
                if (exponents[v] == 0) continue;
                used_by[v] |= u64(1) << row;
                if (monomial[0] == 1) linear |= u64(1) << v;
                else other |= u64(1) << v;
            }
        }
        for (int v = 0; v < ring.n; ++v)
            if ((linear & ~other) & (u64(1) << v)) solved_in[v] |= u64(1) << row;
    }

    u64 remaining = (u64(1) << count) - 1;
    while (remaining) {
        u64 removable = 0;
        for (int v = 0; v < ring.n; ++v) {
            const u64 rows = used_by[v] & remaining;
            if ((rows & (rows - 1)) == 0) removable |= rows & solved_in[v];
        }
        if (!removable) break;
        remaining &= ~removable;
    }
    if (!remaining) return true;

    // A change of generators can hide all the individual pivots. In the
    // remaining rows take the variables that occur only with constant
    // linear coefficients. A surjective coefficient block solves for as
    // many of these variables as there are rows. Its integer rows have
    // already had the rational scalar content removed. A minor surviving
    // modulo the prime is therefore also nonzero over QQ.
    std::vector<size_t> rows;
    std::vector<int> axes;
    for (size_t row = 0; row < count; ++row)
        if (remaining & (u64(1) << row)) rows.push_back(row);
    for (int v = 0; v < ring.n; ++v) {
        const u64 users = used_by[v] & remaining;
        if (users && !(users & ~solved_in[v])) axes.push_back(v);
    }
    if (axes.size() < rows.size()) return false;
    constexpr ulong prime = 65521;
    nmod_mat_t matrix;
    nmod_mat_init(matrix, rows.size(), axes.size(), prime);
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& poly = polynomial(rows[i]);
        for (size_t k = 0; k < poly.size(); ++k) {
            const auto* m = poly.m.data() + k * ring.stride;
            if (m[0] != 1) continue;
            for (size_t j = 0; j < axes.size(); ++j)
                if (Ring::field(m, ring.position(axes[j])))
                    nmod_mat_entry(matrix, i, j) = fmpz_fdiv_ui(&poly.c[k].v, prime);
        }
    }
    const size_t rank = nmod_mat_rank(matrix);
    nmod_mat_clear(matrix);
    return rank == rows.size();
}
}
