#include "quotient_finite_inverse.hpp"
#include <flint/fmpq.h>
#include <flint/fmpq_mat.h>

namespace atlas::quotient_ring {
namespace {
using local::Polynomial;
using local::Powers;

struct Axis {
    size_t degree = 0, tail = 0, stride = 0;
    mpq_class coefficient = 0; // x^degree = coefficient * x^tail
};

struct Matrix {
    fmpq_mat_t value;
    Matrix(size_t rows, size_t cols) { fmpq_mat_init(value, rows, cols); }
    ~Matrix() { fmpq_mat_clear(value); }
    Matrix(const Matrix&) = delete;
};

struct Rational {
    fmpq_t value;
    Rational() { fmpq_init(value); }
    ~Rational() { fmpq_clear(value); }
    Rational(const Rational&) = delete;
};

struct Term {
    std::vector<size_t> powers;
    size_t offset;
    mpq_class coeff;
};

Powers unpack(size_t index, const std::vector<Axis>& axes) {
    Powers powers(axes.size());
    for (size_t i = 0; i < axes.size(); ++i)
        powers[i] = index / axes[i].stride % axes[i].degree;
    return powers;
}

std::optional<Polynomial> triangular_inverse(const std::vector<Axis>& axes, size_t dim,
                                             const std::vector<Term>& terms) {
    if (terms.front().offset != 0) return std::nullopt;
    const auto& constant = terms.front().coeff;
    // Coefficient of x^m in f*g is f(0)*g_m plus terms from earlier
    // indices. Overflowed powers vanish in this truncated polynomial ring.
    std::vector<mpq_class> inverse(dim);
    inverse.front() = 1 / constant;
    for (size_t m = 1; m < dim; ++m) {
        for (size_t j = 1; j < terms.size(); ++j) {
            const auto& term = terms[j];
            bool divides = true;
            for (size_t i = 0; i < axes.size(); ++i) {
                if (term.powers[i] > m / axes[i].stride % axes[i].degree) { divides = false; break; }
            }
            if (divides) inverse[m] -= term.coeff * inverse[m-term.offset];
        }
        inverse[m] /= constant;
    }
    Polynomial out;
    for (size_t m = 0; m < dim; ++m)
        if (inverse[m] != 0) out.emplace(unpack(m, axes), std::move(inverse[m]));
    return out;
}

std::optional<Polynomial> multiplication_inverse(const std::vector<Axis>& axes, size_t dim,
                                                 const std::vector<Term>& terms) {
    // A unit acts by a nonsingular multiplication matrix. Its inverse is the
    // unique vector carried to the coordinate vector of 1.
    Matrix product(dim, dim), one(dim, 1), solution(dim, 1);
    fmpq_one(fmpq_mat_entry(one.value, 0, 0));
    Matrix coeffs(terms.size(), 1), tails(axes.size(), 1);
    for (size_t j = 0; j < terms.size(); ++j)
        fmpq_set_mpq(fmpq_mat_entry(coeffs.value, j, 0), terms[j].coeff.get_mpq_t());
    for (size_t i = 0; i < axes.size(); ++i)
        fmpq_set_mpq(fmpq_mat_entry(tails.value, i, 0), axes[i].coefficient.get_mpq_t());
    Rational coefficient;
    for (size_t col = 0; col < dim; ++col) {
        for (size_t j = 0; j < terms.size(); ++j) {
            size_t row = 0;
            fmpq_set(coefficient.value, fmpq_mat_entry(coeffs.value, j, 0));
            for (size_t i = 0; i < axes.size(); ++i) {
                const auto& axis = axes[i];
                size_t degree = col / axis.stride % axis.degree + terms[j].powers[i];
                auto tail = fmpq_mat_entry(tails.value, i, 0);
                if (degree >= axis.degree && fmpq_is_zero(tail)) { fmpq_zero(coefficient.value); break; }
                while (degree >= axis.degree) {
                    degree -= axis.degree - axis.tail;
                    if (!fmpq_is_one(tail)) fmpq_mul(coefficient.value, coefficient.value, tail);
                }
                row += degree * axis.stride;
            }
            auto entry = fmpq_mat_entry(product.value, row, col);
            fmpq_add(entry, entry, coefficient.value);
        }
    }
    if (!fmpq_mat_solve(solution.value, product.value, one.value)) return std::nullopt;
    Polynomial out;
    for (size_t m = 0; m < dim; ++m) {
        auto entry = fmpq_mat_entry(solution.value, m, 0);
        if (!fmpq_is_zero(entry)) fmpq_get_mpq(out[unpack(m, axes)].get_mpq_t(), entry);
    }
    return out;
}
} // namespace

bool finite_inverse(const local::Basis& basis, const std::vector<local::Powers>& heads,
                    const local::Polynomial& element, size_t width, std::optional<local::Polynomial>& result) {
    if (width < 2 || basis.size() != width || element.empty()) return false;
    std::vector<Axis> axes(width);
    bool truncated = true;
    for (size_t j = 0; j < basis.size(); ++j) {
        if (basis[j].size() > 2) return false;
        size_t axis = width;
        for (size_t i = 0; i < width; ++i) if (heads[j][i] != 0) {
            if (axis != width || heads[j][i] > 1024) return false;
            axis = i;
        }
        if (axis == width || axes[axis].degree != 0) return false;
        auto& a = axes[axis];
        a.degree = heads[j][axis].get_ui();
        for (const auto& [e, c] : basis[j]) if (e != heads[j]) {
            for (size_t i = 0; i < width; ++i) if (i != axis && e[i] != 0) return false;
            if (e[axis] >= heads[j][axis]) return false;
            a.tail = e[axis].get_ui();
            a.coefficient = -c;
            truncated = false;
        }
    }
    // Bound dense memory, with a larger bound for the triangular algorithm.
    const size_t limit = truncated ? 1024 : 64;
    size_t dim = 1;
    for (auto& a : axes) {
        if (a.degree > limit / dim) return false;
        a.stride = dim;
        dim *= a.degree;
    }
    std::vector<Term> terms;
    for (const auto& [e,c] : element) {
        Term term{{}, 0, c};
        for (size_t i = 0; i < width; ++i) {
            const size_t exponent = e[i].get_ui(); // input is already in normal form
            term.powers.push_back(exponent);
            term.offset += axes[i].stride * exponent;
        }
        terms.push_back(std::move(term));
    }
    result = truncated ? triangular_inverse(axes, dim, terms) : multiplication_inverse(axes, dim, terms);
    return true;
}

} // namespace atlas::quotient_ring
