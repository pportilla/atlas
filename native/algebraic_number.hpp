#pragma once
#include <flint/qqbar.h>
#include <string>

namespace atlas {
struct AlgebraicNumber {
    qqbar_t n;
    AlgebraicNumber(long value = 0) { qqbar_init(n); qqbar_set_si(n, value); }
    AlgebraicNumber(const AlgebraicNumber& other) { qqbar_init(n); qqbar_set(n, other.n); }
    AlgebraicNumber(AlgebraicNumber&& other) noexcept { qqbar_init(n); qqbar_swap(n, other.n); }
    AlgebraicNumber& operator=(AlgebraicNumber other) { qqbar_swap(n, other.n); return *this; }
    ~AlgebraicNumber() { qqbar_clear(n); }
    bool zero() const { return qqbar_is_zero(n); }
};
inline AlgebraicNumber operator+(const AlgebraicNumber& a, const AlgebraicNumber& b) {
    AlgebraicNumber sum;
    qqbar_add(sum.n, a.n, b.n);
    return sum;
}

inline AlgebraicNumber operator*(const AlgebraicNumber& a, const AlgebraicNumber& b) {
    AlgebraicNumber product;
    qqbar_mul(product.n, a.n, b.n);
    return product;
}

inline std::string flint_string(char* data) {
    std::string result(data);
    flint_free(data);
    return result;
}
} // namespace atlas
