#pragma once
#include <cstddef>

extern "C" {
// Handles own their ring and generators. Each successful operation returns a new
// independently owned handle. A null result with an empty error requests a worker.
void* atlas_ideal_ring(size_t width) noexcept;
void atlas_ideal_ring_release(void* ring) noexcept;
void* atlas_ideal_polynomial(void* ring, const char* row) noexcept;
void* atlas_ideal_polynomial_gcd(void* left, void* right, int lex) noexcept;
int atlas_ideal_polynomial_is_zero(void* polynomial) noexcept;
char* atlas_ideal_polynomial_row(void* polynomial) noexcept; // free with atlas_groebner_release
void atlas_ideal_polynomial_release(void* polynomial) noexcept;
void* atlas_ideal_from_polynomials(void* ring, void* const* polynomials, size_t count) noexcept;
void* atlas_ideal_prepare(size_t width, const char* rows, size_t count) noexcept;
void* atlas_ideal_binary(void* left, void* right, int product) noexcept;
void* atlas_ideal_power_small(void* ideal, unsigned long exponent, int lex) noexcept;
size_t atlas_ideal_size(void* ideal) noexcept;
char* atlas_ideal_rows(void* ideal) noexcept; // free with atlas_groebner_release
char* atlas_ideal_expressions(void* ideal, const char* coordinate_names) noexcept; // free likewise
const char* atlas_ideal_error() noexcept;
void atlas_ideal_release(void* ideal) noexcept;
}
