#pragma once

#include <flint/fmpz_mat.h>
#include <gmpxx.h>
#include <cstddef>

namespace atlas::local {

// In one weight residue, retain the dual of the previous tangent quotient.
// Rows at the next weight act on that dual plus the new ambient coordinates.
struct WeightedRankLimit {};

class WeightedQuotient {
    fmpz_mat_t dual_;
    fmpz_mat_t relations_;
    size_t& cells_left_;
    size_t& updates_left_;
    int fresh_ = 0;
    void reserve(int rows, int cols);

public:
    WeightedQuotient(size_t& cells_left, size_t& updates_left);
    ~WeightedQuotient();
    WeightedQuotient(const WeightedQuotient&) = delete;
    WeightedQuotient& operator=(const WeightedQuotient&) = delete;

    void begin(int rows, int columns);
    void add(int row, int column, const mpz_class& value);
    int finish();
};

}
