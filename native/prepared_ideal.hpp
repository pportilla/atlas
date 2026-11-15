#pragma once
#include <cstddef>

extern "C" {
// Handles own their ring and generators. Each successful operation returns a new
// independently owned handle. A null result with an empty error requests a worker.
void* atlas_ideal_prepare(size_t width, const char* rows, size_t count) noexcept;
void* atlas_ideal_binary(void* left, void* right, int product) noexcept;
void* atlas_ideal_power_small(void* ideal, unsigned long exponent, int lex) noexcept;
size_t atlas_ideal_size(void* ideal) noexcept;
char* atlas_ideal_rows(void* ideal) noexcept; // free with atlas_groebner_release
char* atlas_ideal_expressions(void* ideal, const char* coordinate_names) noexcept; // free likewise
const char* atlas_ideal_error() noexcept;
void atlas_ideal_release(void* ideal) noexcept;
}
