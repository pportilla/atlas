// On each coordinate torus, integrate on the full character lattice of the
// equations. The index of that lattice is the degree of the finite cover.
#include "local_io.hpp"
#include "euler_lattice.hpp"
#include "groebner.hpp"
#include <algorithm>
#include <optional>
#include <sstream>

namespace atlas {
namespace {
using local::Basis;
using local::Polynomial;
using local::Powers;
using Matrix = std::vector<Powers>;

Matrix differences(const Basis& equations) {
    Matrix rows;
    for (const auto& f : equations) {
        if (f.empty()) continue;
        const auto& first = f.begin()->first;
        for (const auto& [m, c] : f) {
            Powers row = m;
            for (size_t j = 0; j < m.size(); ++j) row[j] -= first[j];
            rows.push_back(std::move(row));
        }
    }
    return rows;
}

std::vector<size_t> active_axes(const Matrix& rows, size_t width) {
    std::vector<size_t> axes;
    for (size_t i = 0; i < width; ++i)
        if (std::any_of(rows.begin(), rows.end(), [i](const auto& m) { return m[i] != 0; }))
            axes.push_back(i);
    return axes;
}

Powers project(const Powers& row, const std::vector<size_t>& axes) {
    Powers result;
    for (size_t i : axes) result.push_back(row[i]);
    return result;
}

void subtract(Powers& a, const mpz_class& q, const Powers& b) {
    for (size_t k = 0; k < a.size(); ++k) a[k] -= q * b[k];
}

Matrix lattice_basis(Matrix rows, size_t width) {
    for (size_t j = 0; j < width; ++j) {
        if (j >= rows.size()) return {};
        auto pivot = std::find_if(rows.begin()+j, rows.end(), [j](const auto& r) { return r[j] != 0; });
        if (pivot == rows.end()) return {};
        std::iter_swap(rows.begin()+j, pivot);
        for (size_t i = j+1; i < rows.size(); ++i)
            while (rows[i][j] != 0) {
                mpz_class q;
                mpz_fdiv_q(q.get_mpz_t(), rows[j][j].get_mpz_t(), rows[i][j].get_mpz_t());
                subtract(rows[j], q, rows[i]);
                std::swap(rows[j], rows[i]);
            }
        if (rows[j][j] < 0)
            for (auto& x : rows[j]) x = -x;
        for (size_t i = 0; i < j; ++i) {
            mpz_class rounded = rows[i][j] + rows[j][j]/2, q;
            mpz_fdiv_q(q.get_mpz_t(), rounded.get_mpz_t(), rows[j][j].get_mpz_t());
            subtract(rows[i], q, rows[j]);
        }
    }
    rows.resize(width);
    return rows;
}

Polynomial transform(const Polynomial& equation, const Matrix& basis) {
    const auto& origin = equation.begin()->first;
    Polynomial result;
    Powers minima;
    for (const auto& [m, c] : equation) {
        auto residual = m;
        subtract(residual, 1, origin);
        Powers powers;
        for (size_t i = 0; i < basis.size(); ++i) {
            if (!mpz_divisible_p(residual[i].get_mpz_t(), basis[i][i].get_mpz_t()))
                throw std::logic_error("Character outside the exponent lattice.");
            powers.emplace_back(residual[i] / basis[i][i]);
            subtract(residual, powers.back(), basis[i]);
        }
        if (minima.empty()) minima = powers;
        else for (size_t i = 0; i < minima.size(); ++i) minima[i] = std::min(minima[i], powers[i]);
        result.emplace(std::move(powers), c);
    }
    Polynomial positive;
    while (!result.empty()) {
        auto term = result.extract(result.begin());
        subtract(term.key(), 1, minima);
        positive.insert(std::move(term));
    }
    return positive;
}

// A unimodular shear preserves the torus. Choose one that shortens the
// exponent ranges in the quotient equations before fibre projection.
void shorten_supports(Basis& equations, size_t width) {
    if (width < 2) return;
    for (size_t step = 0; step < width; ++step) {
        mpz_class improvement = 0;
        size_t first = 0, second = 0;
        int direction = 0;
        for (size_t i = 0; i < width; ++i)
            for (size_t j = 0; j < width; ++j) if (j != i)
                for (int sign : {-1, 1}) {
                    mpz_class gain = 0;
                    for (const auto& f : equations) {
                        const auto& initial = f.begin()->first;
                        mpz_class old = 0, low = initial[i] + sign*initial[j], high = low;
                        for (const auto& [m,c] : f) {
                            const mpz_class shifted = m[i] + sign*m[j];
                            low = std::min(low, shifted);
                            high = std::max(high, shifted);
                            old = std::max(old, m[i]);
                        }
                        gain += old - (high-low);
                    }
                    if (gain > improvement) {
                        improvement = gain; first = i; second = j; direction = sign;
                    }
                }
        if (improvement == 0) return;
        for (auto& f : equations) {
            mpz_class lowest = f.begin()->first[first] + direction*f.begin()->first[second];
            for (const auto& [m,c] : f) {
                mpz_class shifted = m[first] + direction*m[second];
                lowest = std::min(lowest, shifted);
            }
            Polynomial changed;
            while (!f.empty()) {
                auto term = f.extract(f.begin());
                auto& m = term.key();
                m[first] += direction*m[second] - lowest;
                changed.insert(std::move(term));
            }
            f = std::move(changed);
        }
    }
}

mpz_class characteristic(size_t width, const Basis& equations, bool torus) {
    // The Euler kernel accepts exact polynomial expressions including wide exponents.
    std::vector<std::string> names;
    std::ostringstream input;
    for (size_t i = 0; i < width; ++i) {
        names.push_back("x" + std::to_string(i));
        input << names.back() << ' ';
    }
    input << '\n' << equations.size();
    if (torus) for (size_t i = 0; i < width; ++i) input << " 1";
    input << '\n';
    for (const auto& f : equations) input << local::format_polynomial(f, names) << '\n';
    if (torus) for (const auto& v : names) input << v << '\n';
    Z answer = Euler::from_text(input.str(), false);
    mpz_class result;
    fmpz_get_mpz(result.get_mpz_t(), &answer.v);
    return result;
}

mpz_class torus_sum(const Basis& equations, size_t width) {
    Matrix support;
    for (const auto& f : equations)
        for (const auto& term : f) support.push_back(term.first);
    auto active = active_axes(support, width);
    if (active.size() > 5) return characteristic(width, equations, false);
    mpz_class total = 0;
    for (unsigned orbit = 0; orbit < (1u << active.size()); ++orbit) {
        std::vector<size_t> axes, zeros;
        for (size_t i = 0; i < active.size(); ++i)
            ((orbit >> i) & 1 ? axes : zeros).push_back(active[i]);
        Basis cuts;
        bool empty = false;
        for (const auto& f : equations) {
            Polynomial cut;
            for (const auto& [m,c] : f)
                if (std::all_of(zeros.begin(), zeros.end(), [&](size_t i) { return m[i] == 0; }))
                    cut.emplace(project(m, axes), c);
            if (cut.size() == 1) { empty = true; break; }
            if (!cut.empty()) cuts.push_back(std::move(cut));
        }
        if (empty) continue;
        if (axes.empty()) { ++total; continue; }
        const auto matrix = lattice_basis(differences(cuts), axes.size());
        if (matrix.empty()) continue;
        mpz_class index = 1;
        for (size_t i = 0; i < axes.size(); ++i) index *= matrix[i][i];
        for (auto& cut : cuts) cut = transform(cut, matrix);
        shorten_supports(cuts, axes.size());
        total += index * characteristic(axes.size(), cuts, true);
    }
    return total;
}
}

bool torus_lattice_compresses(size_t width, const local::Basis& equations) {
    const auto matrix = lattice_basis(differences(equations), width);
    for (size_t i = 0; i < matrix.size(); ++i)
        if (matrix[i][i] > 1) return true;
    return false;
}

mpz_class coordinate_torus_euler(size_t width, const local::Basis& equations) {
    return torus_sum(equations, width);
}

}
