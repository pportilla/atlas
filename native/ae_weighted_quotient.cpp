#include "ae_weighted_quotient.hpp"
#include "small_prime_rank.hpp"

#include <flint/fmpz.h>
#include <cstdlib>

namespace atlas::local {
namespace {

struct Matrix {
    fmpz_mat_t value;
    Matrix(int rows, int cols) { fmpz_mat_init(value,rows,cols); }
    ~Matrix() { fmpz_mat_clear(value); }
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
};

// Scaling a dual vector by a nonzero rational leaves its span unchanged.
void remove_column_content(fmpz_mat_t matrix) {
    fmpz_t content;
    fmpz_init(content);
    for (slong j = 0; j < matrix->c; ++j) {
        fmpz_zero(content);
        for (slong i = 0; i < matrix->r && !fmpz_is_one(content); ++i)
            fmpz_gcd(content,content,fmpz_mat_entry(matrix,i,j));
        if (!fmpz_is_one(content))
            for (slong i = 0; i < matrix->r; ++i)
                fmpz_divexact(fmpz_mat_entry(matrix,i,j),fmpz_mat_entry(matrix,i,j),content);
    }
    fmpz_clear(content);
}
}

WeightedQuotient::WeightedQuotient(size_t& cells_left, size_t& updates_left)
    : cells_left_(cells_left), updates_left_(updates_left) {
    fmpz_mat_init(dual_,0,0);
    fmpz_mat_init(relations_,0,0);
}

WeightedQuotient::~WeightedQuotient() {
    fmpz_mat_clear(dual_);
    fmpz_mat_clear(relations_);
}

void WeightedQuotient::reserve(int rows, int cols) {
    const size_t cells = static_cast<size_t>(rows)*cols;
    if (cells > 180000 || cells > cells_left_) throw WeightedRankLimit{};
    cells_left_ -= cells;
}

void WeightedQuotient::begin(int rows, int columns) {
    fresh_ = columns-dual_->r;
    const int width = fresh_+dual_->c;
    reserve(rows,width);
    Matrix next(rows,width);
    fmpz_mat_swap(next.value,relations_);
}

void WeightedQuotient::add(int row, int column, const mpz_class& value) {
    fmpz_t coefficient;
    const size_t updates = column < dual_->r ? dual_->c : 1;
    if (updates > updates_left_) throw WeightedRankLimit{};
    updates_left_ -= updates;
    fmpz_init_set_readonly(coefficient,value.get_mpz_t());
    if (column < dual_->r) {
        for (slong k = 0; k < dual_->c; ++k)
            fmpz_addmul(fmpz_mat_entry(relations_,row,fresh_+k),coefficient,
                        fmpz_mat_entry(dual_,column,k));
    } else {
        auto entry = fmpz_mat_entry(relations_,row,column-dual_->r);
        fmpz_add(entry,entry,coefficient);
    }
    fmpz_clear_readonly(coefficient);
}

int WeightedQuotient::finish() {
    const int rows = relations_->r, width = relations_->c;
    const int columns = dual_->r+fresh_;
    if (width && rows >= width) {
        // Full rank modulo one prime proves that the rational dual is zero.
        constexpr ulong prime = 65521;
        nmod_mat_t mod;
        nmod_mat_init(mod,rows,width,prime);
        fmpz_mat_get_nmod_mat(mod,relations_);
        const bool full = modular_rank_in_place<prime>(mod) == width;
        nmod_mat_clear(mod);
        if (full) {
            Matrix next(columns,0);
            fmpz_mat_swap(next.value,dual_);
            return 0;
        }
    }
    reserve(width,width);
    // Bound the cost of fraction-free elimination when earlier dual bases
    // have already produced large coefficients.
    if (std::abs(fmpz_mat_max_bits(relations_))*std::min(rows,width) > 40000)
        throw WeightedRankLimit{};
    Matrix kernel(width,width);
    const int nullity = width ? fmpz_mat_nullspace(kernel.value,relations_) : 0;
    reserve(columns,nullity);
    Matrix next(columns,nullity);
    if (nullity) {
        const auto products = static_cast<size_t>(dual_->r)*dual_->c*nullity;
        if (products > updates_left_) throw WeightedRankLimit{};
        updates_left_ -= products;
        // In coordinates on the old ambient space and fresh columns, the
        // null vectors are (old_dual * inherited_part, fresh_part).
        if (dual_->r && dual_->c) {
            fmpz_mat_t inherited, destination;
            fmpz_mat_window_init(inherited,kernel.value,fresh_,0,width,nullity);
            fmpz_mat_window_init(destination,next.value,0,0,dual_->r,nullity);
            fmpz_mat_mul(destination,dual_,inherited);
            fmpz_mat_window_clear(destination);
            fmpz_mat_window_clear(inherited);
        }
        for (int i = 0; i < fresh_; ++i) for (int k = 0; k < nullity; ++k)
            fmpz_set(fmpz_mat_entry(next.value,dual_->r+i,k),fmpz_mat_entry(kernel.value,i,k));
        remove_column_content(next.value);
        if (std::abs(fmpz_mat_max_bits(next.value)) > 4096) throw WeightedRankLimit{};
    }
    fmpz_mat_swap(next.value,dual_);
    return nullity;
}

}
