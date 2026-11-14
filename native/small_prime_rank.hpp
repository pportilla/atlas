#pragma once

#include <flint/nmod_mat.h>
#include <flint/ulong_extras.h>
#include <algorithm>
#include <limits>
#include <vector>

namespace atlas {

// Rank of a matrix of canonical residues over a prime field. The matrix
// may be a strided window and its contents may be overwritten.
template<ulong Prime>
slong modular_rank_in_place(nmod_mat_t matrix) {
    static_assert(Prime >= 2);
    const slong rows = nmod_mat_nrows(matrix);
    const slong cols = nmod_mat_ncols(matrix);
    const slong maximum_rank = std::min(rows, cols);
    if (!maximum_rank) return 0;

    // Each pivot can add at most (Prime-1)^2 to an unreduced entry once.
    // Use FLINT if a whole row cannot be eliminated safely in one word.
    constexpr ulong residue_max = Prime - 1;
    constexpr ulong safe_updates =
        (std::numeric_limits<ulong>::max() - residue_max) / residue_max / residue_max;
    if (matrix->mod.n != Prime || static_cast<ulong>(maximum_rank) > safe_updates)
        return nmod_mat_rank(matrix);

    struct Pivot {
        const ulong* row = nullptr;
        slong end = 0;
    };
    std::vector<Pivot> pivots(cols);
    slong rank = 0;
    for (slong r = 0; r < rows; ++r) {
        auto* row = nmod_mat_entry_ptr(matrix, r, 0);
        for (slong col = 0; col < cols; ++col) {
            const ulong value = row[col] % Prime;
            if (!value) continue;
            auto& pivot = pivots[col];
            if (pivot.row) {
                const ulong multiple = Prime - value;
                for (slong i = col + 1; i < pivot.end; ++i)
                    row[i] += multiple * pivot.row[i];
                continue;
            }
            if (++rank == maximum_rank) return rank;
            slong end = cols;
            while (end > col + 1 && !row[end-1]) --end;
            pivot = {row, end};
            const ulong inverse = n_invmod(value, Prime);
            for (slong i = col + 1; i < end; ++i)
                row[i] = ((row[i] % Prime) * inverse) % Prime;
            break;
        }
    }
    return rank;
}

} // namespace atlas
