#include "small_prime_rank.hpp"

#include <iostream>
#include <random>
#include <utility>

enum class Pattern { zero, sparse, dense, dependent };

template<unsigned long Prime>
bool agrees(slong rows, slong columns, Pattern pattern, std::mt19937& random) {
    nmod_mat_t storage, matrix;
    nmod_mat_init(storage, rows + 2, columns + 3, Prime);
    nmod_mat_window_init(matrix, storage, 1, 2, rows + 1, columns + 2);
    for (slong r = 0; r < rows; ++r) {
        nmod_mat_entry(storage, r + 1, columns + 2) = Prime - 1;
        for (slong c = 0; c < columns; ++c) {
            unsigned long value = random() % Prime;
            if (pattern == Pattern::zero || (pattern == Pattern::sparse && random() % 9))
                value = 0;
            if (pattern == Pattern::dependent) {
                // Trailing zero columns, repeated rows, and negatives of pivot
                // rows make cancellations that depend on the characteristic.
                if (c >= columns / 2) value = 0;
                if (r % 3 == 1) value = nmod_mat_entry(matrix, r - 1, c);
                if (r % 3 == 2) {
                    const auto previous = nmod_mat_entry(matrix, r - 1, c);
                    value = previous == 0 ? 0 : Prime - previous;
                }
            }
            nmod_mat_entry(matrix, r, c) = value;
        }
    }
    const auto expected = nmod_mat_rank(matrix);
    const auto actual = atlas::modular_rank_in_place<Prime>(matrix);
    bool guard = true;
    for (slong r = 0; r < rows; ++r)
        guard &= nmod_mat_entry(storage, r + 1, columns + 2) == Prime - 1;
    nmod_mat_window_clear(matrix);
    nmod_mat_clear(storage);
    if (!guard) std::cerr << "Modified an entry outside the matrix window\n";
    if (expected == actual) return guard;
    std::cerr << rows << " x " << columns << " modulo " << Prime << ", pattern "
              << static_cast<int>(pattern) << ": expected rank " << expected
              << ", obtained " << actual << '\n';
    return false;
}

template<unsigned long Prime>
bool check(std::mt19937& random) {
    const std::pair<slong, slong> shapes[] = {
        {0, 0}, {0, 7}, {5, 0}, {1, 9}, {12, 1}, {5, 37}, {41, 7}, {27, 27}, {99, 106}
    };
    for (auto [rows, columns] : shapes)
        for (auto pattern : {Pattern::zero, Pattern::sparse, Pattern::dense, Pattern::dependent})
            if (!agrees<Prime>(rows, columns, pattern, random)) return false;
    return true;
}

bool check_runtime_prime() {
    // This determinant vanishes modulo 3, while it equals 3 modulo 65521.
    nmod_mat_t matrix;
    nmod_mat_init(matrix, 2, 2, 3);
    nmod_mat_entry(matrix, 0, 0) = nmod_mat_entry(matrix, 1, 1) = 2;
    nmod_mat_entry(matrix, 1, 0) = nmod_mat_entry(matrix, 0, 1) = 1;
    const auto rank = atlas::modular_rank_in_place<65521>(matrix);
    nmod_mat_clear(matrix);
    if (rank == 1) return true;
    std::cerr << "Runtime prime 3 should give rank 1, obtained " << rank << '\n';
    return false;
}

int main() {
    std::mt19937 random(20261114);
    // Near 2^32, even two unreduced prime-field products can overflow 64 bits.
    return check<3>(random) && check<65521>(random) && check<4294967291UL>(random)
           && check_runtime_prime() ? 0 : 1;
}
