#include "point_jet.hpp"
#include "smooth_point_order.hpp"
#include "groebner.hpp"

#include <flint/fmpq_poly.h>
#include <flint/fmpz_poly.h>
#include <flint/fmpz_poly_factor.h>
#include <array>
#include <numeric>

namespace atlas::certificate {
namespace {
using namespace chart;
using Coeff = FieldCoefficient;
using Series = std::map<JetPowers, Coeff>;
using Matrix = std::vector<std::vector<Coeff>>;
struct Unknown {};

unsigned degree_of(const JetPowers& e) { return std::accumulate(e.begin(), e.end(), 0u); }

void trim(Coeff& a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
}

void height(const Q& q) {
    if (mpz_sizeinbase(q.get_num_mpz_t(), 2) > 8192 || mpz_sizeinbase(q.get_den_mpz_t(), 2) > 8192)
        throw Unknown{};
}

void check(const CertificateCaps& caps) {
    if (caps.check_time) caps.check_time();
    atlas::check_deadline();
}

struct QPoly {
    fmpq_poly_t value;
    QPoly() { fmpq_poly_init(value); }
    explicit QPoly(const Coeff& a) : QPoly() {
        fmpq_t q;
        fmpq_init(q);
        for (size_t i = 0; i < a.size(); ++i) {
            fmpq_set_mpq(q, a[i].get_mpq_t());
            fmpq_poly_set_coeff_fmpq(value, i, q);
        }
        fmpq_clear(q);
    }
    ~QPoly() { fmpq_poly_clear(value); }
    QPoly(const QPoly&) = delete;
    Coeff coefficients() const {
        Coeff a(fmpq_poly_length(value));
        fmpq_t q;
        fmpq_init(q);
        for (size_t i = 0; i < a.size(); ++i) {
            fmpq_poly_get_coeff_fmpq(q, value, i);
            fmpq_get_mpq(a[i].get_mpq_t(), q);
        }
        fmpq_clear(q);
        return a;
    }
};

// A nonzero element on a reducible quotient could vanish at one of its
// components. Only use a field to certify an order shared by conjugates.
bool irreducible(const Coeff& mu) {
    thread_local Coeff last;
    if (mu == last) return true;
    QPoly polynomial(mu);
    fmpz_poly_t primitive;
    fmpz_poly_init(primitive);
    fmpq_poly_get_numerator(primitive, polynomial.value);
    slong bits = fmpz_poly_max_bits(primitive);
    bool ok = bits >= -1024 && bits <= 1024;
    if (ok) {
        fmpz_poly_factor_t factors;
        fmpz_poly_factor_init(factors);
        fmpz_poly_factor(factors, primitive);
        ok = factors->num == 1 && factors->exp[0] == 1;
        fmpz_poly_factor_clear(factors);
    }
    fmpz_poly_clear(primitive);
    if (ok) last = mu;
    return ok;
}

// Multiply the independently translated coordinate columns, truncating by
// total free-parameter degree before constructing each output coefficient.
template<class T, class Product, class Emit>
void expand_columns(const std::vector<std::pair<size_t, std::vector<T>>>& axes, JetPowers& exponent,
                    unsigned room, const T& scalar, Product product, Emit emit) {
    auto visit = [&](auto&& self, size_t i, unsigned left, const T& value) -> void {
        if (i == axes.size()) { emit(exponent, value); return; }
        const auto& [axis, column] = axes[i];
        for (size_t k = 0; k < column.size() && k <= left; ++k) {
            exponent[axis] = k;
            self(self, i + 1, left - k, product(value, column[k]));
        }
    };
    visit(visit, 0, room, scalar);
}

struct Arithmetic {
    const Coeff& mu;  // monic minimal polynomial certified by finite_locus
    const CertificateCaps& caps;
    size_t remaining;
    void spend(size_t n) {
        check(caps);
        if (n > remaining) throw Unknown{};
        remaining -= n;
    }

    void add(Coeff& x, const Coeff& y, bool subtract = false) {
        x.resize(std::max(x.size(), y.size()));
        for (size_t i = 0; i < y.size(); ++i) {
            if (subtract) x[i] -= y[i];
            else x[i] += y[i];
            height(x[i]);
        }
        trim(x);
    }

    Coeff multiply(const Coeff& x, const Coeff& y) {
        if (x.empty() || y.empty()) return {};
        spend(x.size() * y.size());
        if (x.size() == 1 && x[0] == 1) return y;
        if (y.size() == 1 && y[0] == 1) return x;
        Coeff result(x.size() + y.size() - 1);
        for (size_t i = 0; i < x.size(); ++i)
            for (size_t j = 0; j < y.size(); ++j) {
                if (!x[i] || !y[j]) continue;
                result[i+j] += x[i] * y[j];
                height(result[i+j]);
            }
        while (result.size() >= mu.size()) {
            size_t offset = result.size() - mu.size();
            const Q leading = result.back();
            result.pop_back();
            if (!leading) continue;
            for (size_t i = 0; i + 1 < mu.size(); ++i) {
                result[offset+i] -= leading * mu[i];
                height(result[offset+i]);
            }
        }
        trim(result);
        return result;
    }

    Coeff inverse(const Coeff& x) {
        if (x.empty()) throw Unknown{};
        if (x.size() == 1) return {1 / x[0]};
        QPoly a(x), modulus(mu), gcd, inverse, other;
        fmpq_poly_xgcd(gcd.value, inverse.value, other.value, a.value, modulus.value);
        if (!fmpq_poly_is_one(gcd.value)) throw Unknown{};
        Coeff result = inverse.coefficients();
        for (const auto& q : result) height(q);
        return result;
    }

    void accumulate(Series& result, const JetPowers& exponent, const Coeff& value, bool subtract = false) {
        if (value.empty()) return;
        auto it = result.find(exponent);
        if (it == result.end()) {
            if (result.size() >= caps.max_cols) throw Unknown{};
            it = result.emplace(exponent, Coeff{}).first;
        }
        add(it->second, value, subtract);
        if (it->second.empty()) result.erase(it);
    }

    Series product(const Series& x, const Series& y, unsigned degree) {
        Series result;
        for (const auto& [e, a] : x) {
            spend(y.size());
            for (const auto& [v, b] : y) {
                if (degree_of(e) + degree_of(v) > degree) continue;
                JetPowers sum = e;
                for (size_t j = 0; j < e.size(); ++j) sum[j] += v[j];
                accumulate(result, sum, multiply(a, b));
            }
        }
        return result;
    }

    // All powers involving only positive-degree terms disappear past the
    // requested degree, even if the input exponent has millions of bits.
    Series power(Series base, const chart::Z& exponent, const JetPowers& zero, unsigned degree) {
        if (exponent < 0) throw Unknown{};
        if (exponent == 0) return {{zero, {1}}};
        if (base.empty() || (!base.count(zero) && exponent > degree)) return {};
        if (exponent == 1) return base;
        if (base.size() == 1 && base.count(zero)) {
            const auto& c = base.at(zero);
            if (c == Coeff{1} || c == Coeff{-1})
                return {{zero, mpz_odd_p(exponent.get_mpz_t()) ? c : Coeff{1}}};
        }
        if (mpz_sizeinbase(exponent.get_mpz_t(), 2) > 256) throw Unknown{};
        Series result{{zero, {1}}};
        chart::Z e = exponent;
        while (e != 0) {
            if (mpz_odd_p(e.get_mpz_t())) result = product(result, base, degree);
            e >>= 1;
            if (e != 0) base = product(base, base, degree);
        }
        return result;
    }

    Series evaluate(const local::Polynomial& f, const std::vector<Series>& images, size_t width, unsigned degree) {
        JetPowers zero(width);
        Series sum;
        for (const auto& [e, c] : f) {
            spend(1);
            Series term;
            bool constant = true;
            for (size_t i = 0; i < e.size(); ++i) {
                if (!e[i]) continue;
                Series factor = power(images[i], e[i], zero, degree);
                term = constant ? std::move(factor) : product(term, factor, degree);
                constant = false;
                if (term.empty()) break;
            }
            if (constant) {
                accumulate(sum, zero, {c});
                continue;
            }
            spend(term.size());
            for (const auto& [p, value] : term) {
                if (c == 1 || c == -1) accumulate(sum, p, value, c == -1);
                else accumulate(sum, p, multiply({c}, value));
            }
        }
        return sum;
    }

    Coeff scalar_power(Coeff base, chart::Z exponent) {
        if (exponent == 0) return {1};
        if (base.empty()) return {};
        if (base.size() == 1 && (base[0] == 1 || base[0] == -1))
            return {mpz_odd_p(exponent.get_mpz_t()) ? base[0] : Q(1)};
        if (mpz_sizeinbase(exponent.get_mpz_t(), 2) > 256) throw Unknown{};
        Coeff result{1};
        while (exponent != 0) {
            if (mpz_odd_p(exponent.get_mpz_t())) result = multiply(result, base);
            exponent >>= 1;
            if (exponent != 0) base = multiply(base, base);
        }
        return result;
    }

    // Binomial columns at an exact point. Translate one monomial directly
    // instead of multiplying intermediate power-series maps.
    Series taylor(const local::Polynomial& f, const GaloisOrbit& point,
                  const std::vector<size_t>& moving, unsigned degree) {
        Series result;
        const size_t n = point.coordinates.size();
        std::vector<size_t> position(n, n);
        for (size_t j = 0; j < moving.size(); ++j) position[moving[j]] = j;
        JetPowers exponent(moving.size());
        for (const auto& [e, c] : f) {
            unsigned room = degree;
            Coeff scalar{c};
            std::fill(exponent.begin(), exponent.end(), 0);
            std::vector<std::pair<size_t, std::vector<Coeff>>> axes;
            for (size_t i = 0; i < n && !scalar.empty(); ++i) {
                if (!e[i]) continue;
                auto at = point.coordinates[i];
                trim(at);
                if (position[i] == n) {
                    scalar = multiply(scalar, scalar_power(at, e[i]));
                    continue;
                }
                if (at.empty()) {
                    if (e[i] > room) { scalar.clear(); break; }
                    exponent[position[i]] = e[i].get_ui();
                    room -= exponent[position[i]];
                    continue;
                }
                const unsigned top = e[i] < room ? e[i].get_ui() : room;
                spend(top + 1);
                std::vector<Coeff> coefficients(top + 1);
                Coeff value = scalar_power(at, e[i] - top);
                for (unsigned k = top;; --k) {
                    if (k && mpz_sizeinbase(e[i].get_mpz_t(), 2) > 8192 / k) throw Unknown{};
                    chart::Z binomial;
                    mpz_bin_ui(binomial.get_mpz_t(), e[i].get_mpz_t(), k);
                    height(Q(binomial));
                    coefficients[k] = multiply(value, Coeff{Q(binomial)});
                    if (!k) break;
                    value = multiply(value, at);
                }
                axes.emplace_back(position[i], std::move(coefficients));
            }
            if (scalar.empty()) continue;
            expand_columns(axes, exponent, room, scalar,
                [&](const Coeff& a, const Coeff& b) { return multiply(a, b); },
                [&](const JetPowers& p, const Coeff& value) { spend(1); accumulate(result, p, value); });
        }
        return result;
    }

    Matrix invert(Matrix a) {
        const size_t n = a.size();
        Matrix inverse(n, std::vector<Coeff>(n));
        for (size_t i = 0; i < n; ++i) inverse[i][i] = {1};
        for (size_t col = 0; col < n; ++col) {
            size_t pivot = col;
            while (pivot < n && a[pivot][col].empty()) ++pivot;
            if (pivot == n) throw Unknown{};
            std::swap(a[pivot], a[col]);
            std::swap(inverse[pivot], inverse[col]);
            const Coeff scale = this->inverse(a[col][col]);
            for (size_t j = 0; j < n; ++j) {
                a[col][j] = multiply(a[col][j], scale);
                inverse[col][j] = multiply(inverse[col][j], scale);
            }
            for (size_t row = 0; row < n; ++row) if (row != col) {
                const Coeff multiple = a[row][col];
                if (multiple.empty()) continue;
                for (size_t j = 0; j < n; ++j) {
                    add(a[row][j], multiply(multiple, a[col][j]), true);
                    add(inverse[row][j], multiply(multiple, inverse[col][j]), true);
                }
            }
        }
        return inverse;
    }
};

// When theta^d is an integer and every coordinate is a single integral
// multiple of a power of theta, every binomial column stays a field
// monomial. Checked word arithmetic avoids constructing a rational vector at
// each multiplication. Any overflow resumes the general field arithmetic.
struct NotPacked {};
long integral(const Q& q) {
    if (mpz_cmp_ui(q.get_den_mpz_t(), 1) || !mpz_fits_slong_p(q.get_num_mpz_t())) throw NotPacked{};
    return mpz_get_si(q.get_num_mpz_t());
}
long checked_product(long a, long b) {
    long result;
    if (__builtin_mul_overflow(a, b, &result)) throw NotPacked{};
    return result;
}
struct FieldMonomial { long coefficient; unsigned power; };

std::optional<Series> binomial_point_jet(const local::Polynomial& f, const GaloisOrbit& orbit,
                                         unsigned degree, const CertificateCaps& caps) {
    try {
        const unsigned d = orbit.mu.size() - 1;
        for (unsigned i = 1; i < d; ++i) if (orbit.mu[i]) return std::nullopt;
        const long modulus = checked_product(-1, integral(orbit.mu.front()));
        std::vector<FieldMonomial> point;
        for (const auto& row : orbit.coordinates) {
            FieldMonomial x{0, 0};
            for (size_t i = 0; i < row.size(); ++i) if (row[i]) {
                if (x.coefficient) throw NotPacked{};
                x = {integral(row[i]), static_cast<unsigned>(i)};
            }
            point.push_back(x);
        }
        auto multiply = [&](FieldMonomial a, FieldMonomial b) {
            FieldMonomial result{checked_product(a.coefficient, b.coefficient), a.power + b.power};
            if (result.power >= d) {
                result.power -= d;
                result.coefficient = checked_product(result.coefficient, modulus);
            }
            return result;
        };
        using Integers = std::array<long,16>;
        std::map<JetPowers, Integers> terms;
        size_t work = caps.max_nonzeros;
        for (const auto& [e, c] : f) {
            check(caps);
            JetPowers exponent(point.size());
            std::vector<std::pair<size_t, std::vector<FieldMonomial>>> axes;
            for (size_t i = 0; i < point.size(); ++i) if (e[i]) {
                if (e[i] > 32) throw NotPacked{};
                const unsigned n = e[i].get_ui(), top = std::min(n, degree);
                std::vector<FieldMonomial> column(top + 1);
                FieldMonomial value{1, 0};
                // Running binomial(n,j) fits a 64-bit word for n <= 32.
                uint64_t choose = 1;
                for (unsigned j = 0; j <= n; ++j) {
                    if (n - j <= top) {
                        if (choose > static_cast<uint64_t>(LONG_MAX)) throw NotPacked{};
                        column[n - j] = {checked_product(value.coefficient, static_cast<long>(choose)), value.power};
                    }
                    if (j < n) {
                        choose = choose * (n-j) / (j+1);
                        value = multiply(value, point[i]);
                    }
                }
                axes.emplace_back(i, std::move(column));
            }
            expand_columns(axes, exponent, degree, FieldMonomial{integral(c), 0}, multiply,
                [&](const JetPowers& p, FieldMonomial m) {
                    if (!(work & 255)) check(caps);
                    if (!work) throw Unknown{};
                    --work;
                    if (!m.coefficient) return;
                    auto it = terms.find(p);
                    if (it == terms.end()) {
                        if (terms.size() >= std::min(caps.max_cols, caps.max_nonzeros/d)) throw Unknown{};
                        it = terms.emplace(p, Integers{}).first;
                    }
                    long& value = it->second[m.power];
                    if (__builtin_add_overflow(value, m.coefficient, &value)) throw NotPacked{};
                });
        }
        Series result;
        for (const auto& [e, integers] : terms) {
            Coeff value;
            for (unsigned i = 0; i < d; ++i) if (integers[i]) {
                value.resize(i+1);
                value[i] = integers[i];
            }
            if (!value.empty()) result.emplace_hint(result.end(), e, std::move(value));
        }
        return result;
    } catch (const NotPacked&) {
        return std::nullopt;
    }
}

std::vector<Series> coordinates(const GaloisOrbit& orbit, const std::vector<size_t>& moving, unsigned degree) {
    std::vector<Series> images(orbit.coordinates.size());
    JetPowers zero(moving.size());
    for (size_t i = 0; i < images.size(); ++i) {
        auto c = orbit.coordinates[i];
        trim(c);
        if (!c.empty()) images[i].emplace(zero, std::move(c));
    }
    if (degree) for (size_t j = 0; j < moving.size(); ++j) {
        JetPowers variable = zero;
        variable[j] = 1;
        images[moving[j]][variable] = {1};
    }
    return images;
}

std::optional<PointJet> calculate(const Variables& vars, const Ideal& rows, const Variables& dependent,
                                  const Expr& f, const GaloisOrbit& orbit, unsigned degree, const CertificateCaps& caps) {
    check(caps);
    const size_t n = vars.size(), r = rows.size(), field_degree = orbit.mu.size() ? orbit.mu.size() - 1 : 0;
    if (degree > std::min(32u, caps.max_degree) || n > 24 || r > n || dependent.size() != r ||
        r > caps.max_rank || r > caps.max_rows || r > caps.max_cols || !caps.max_cols || !caps.max_nonzeros ||
        !field_degree || field_degree > 16 || orbit.mu.back() != 1 ||
        orbit.coordinates.size() != n || 2*r*r*field_degree > caps.max_nonzeros)
        throw Unknown{};
    for (const auto& q : orbit.mu) height(q);
    for (const auto& c : orbit.coordinates) {
        if (c.size() > field_degree) throw Unknown{};
        for (const Q& q : c) height(q);
    }

    PointJet jet{degree, {}, {}};
    std::vector<bool> chosen(n);
    std::vector<size_t> pivots, free;
    for (const auto& c : dependent) {
        auto found = std::find(vars.begin(), vars.end(), c);
        if (found == vars.end() || chosen[found - vars.begin()]) throw Unknown{};
        pivots.push_back(found - vars.begin());
        chosen[pivots.back()] = true;
    }
    for (size_t i = 0; i < n; ++i) if (!chosen[i]) {
        jet.parameters.push_back(vars[i]);
        free.push_back(i);
    }
    if (std::set<Coord>(vars.begin(), vars.end()).size() != n) throw Unknown{};

    size_t entries = caps.max_nonzeros;
    auto decode = [&](const Expr& g) {
        if (g.terms().size() > entries || g.coords().size() > n) throw Unknown{};
        entries -= g.terms().size();
        std::vector<size_t> where;
        for (const auto& c : g.coords()) {
            auto at = std::find(vars.begin(), vars.end(), c);
            if (at == vars.end()) throw Unknown{};
            const size_t axis = at - vars.begin();
            if (std::find(where.begin(), where.end(), axis) != where.end()) throw Unknown{};
            where.push_back(axis);
        }
        local::Polynomial poly;
        for (const auto& [e, c] : g.terms()) {
            check(caps);
            height(c);
            if (e.size() != where.size()) throw Unknown{};
            Powers p(n);
            for (size_t j = 0; j < e.size(); ++j) {
                if (e[j] < 0) throw Unknown{};
                p[where[j]] = e[j];
            }
            poly.emplace(std::move(p), c);
        }
        return poly;
    };
    local::Basis equations;
    for (const auto& row : rows) equations.push_back(decode(row));
    auto target = decode(f);

    if (field_degree == 1) {
        // The existing rational smooth-point reducer can return its remaining
        // free-parameter terms. Bound its intermediate monomial space before
        // asking it to materialize a translated jet.
        size_t monomials = 1;
        for (size_t j = 1; j <= n; ++j) {
            monomials = monomials * (std::max(1u, degree) + j) / j;
            if (monomials > caps.max_cols) throw Unknown{};
        }
        std::vector<Q> point;
        for (const auto& coord : orbit.coordinates) point.push_back(coord.empty() ? Q(0) : coord[0]);
        auto result = local::smooth_point_jet(equations, point, target, chosen, degree,
                                              caps.max_nonzeros, caps.check_time ? caps.check_time : atlas::check_deadline);
        if (!result) throw Unknown{};
        for (const auto& [e, c] : *result) {
            JetPowers p;
            for (const auto& v : e) p.push_back(v.get_ui());
            jet.coefficients.emplace(std::move(p), Coeff{c});
        }
        return jet;
    }

    if (!irreducible(orbit.mu)) throw Unknown{};
    Arithmetic algebra{orbit.mu, caps, caps.max_nonzeros};
    if (!r) {
        auto packed = binomial_point_jet(target, orbit, degree, caps);
        jet.coefficients = packed ? std::move(*packed) : algebra.taylor(target, orbit, free, degree);
        return jet;
    }
    if (degree && caps.max_cols < 2)
        for (auto i : free)
            if (std::any_of(orbit.coordinates[i].begin(), orbit.coordinates[i].end(),
                            [](const Q& q){ return q != 0; })) throw Unknown{};
    auto images = coordinates(orbit, free, degree);
    // Read the Jacobian in the dependent coordinates from the first jet.
    Matrix jacobian(r, std::vector<Coeff>(r));
    for (size_t i = 0; i < r; ++i) {
        Series row = algebra.taylor(equations[i], orbit, pivots, 1);
        if (row.count(JetPowers(r))) throw Unknown{};
        for (size_t j = 0; j < r; ++j) {
            JetPowers p(r);
            p[j] = 1;
            auto entry = row.find(p);
            if (entry != row.end()) jacobian[i][j] = entry->second;
        }
    }
    const Matrix inverse = algebra.invert(std::move(jacobian));
    for (unsigned k = 1; k <= degree; ++k) {
        // Lower layers already solve R = 0. A change in layer k acts
        // through the constant Jacobian, so -J^(-1)R kills that layer.
        std::vector<Series> residual;
        for (const auto& row : equations) residual.push_back(algebra.evaluate(row, images, free.size(), k));
        for (size_t j = 0; j < r; ++j) for (size_t i = 0; i < r; ++i) {
            if (inverse[j][i].empty()) continue;
            for (const auto& [p, c] : residual[i]) {
                if (degree_of(p) != k) throw Unknown{};
                algebra.accumulate(images[pivots[j]], p, algebra.multiply(inverse[j][i], c), true);
            }
        }
    }
    for (const auto& row : equations)
        if (!algebra.evaluate(row, images, free.size(), degree).empty()) throw Unknown{};
    jet.coefficients = algebra.evaluate(target, images, free.size(), degree);
    return jet;
}
} // namespace

unsigned jet_order(const PointJet& jet) {
    unsigned least = jet.degree + 1;
    for (const auto& item : jet.coefficients) least = std::min(least, degree_of(item.first));
    return least;
}

std::optional<PointJet> frame_jet(const chart::Variables& vars, const chart::Ideal& rows,
                                  const chart::Variables& dep, const chart::Expr& f,
                                  const GaloisOrbit& orbit, unsigned degree, const CertificateCaps& caps) {
    try { return calculate(vars, rows, dep, f, orbit, degree, caps); }
    catch (const Unknown&) { return std::nullopt; }
}

std::optional<PointJet> taylor_jet(const chart::Variables& vars, const chart::Expr& f,
                                   const GaloisOrbit& orbit, unsigned degree, const CertificateCaps& caps) {
    return frame_jet(vars, {}, {}, f, orbit, degree, caps);
}
std::optional<bool> nonzero_on_coordinate_line(const Variables& vars, const Ideal& rows,
                                               const Coord& parameter, const Expr& f,
                                               const GaloisOrbit& orbit, const CertificateCaps& caps) {
    check(caps);
    if (vars.empty() || vars.size() > 8 || vars.size() != rows.size() + 1) return {};
    const auto it = std::find(vars.begin(), vars.end(), parameter);
    if (it == vars.end()) return {};
    const size_t axis = it - vars.begin();
    Variables normal = vars;
    normal.erase(normal.begin() + axis);
    // This checks the point, its field and the full normal Jacobian.
    if (!frame_jet(vars, rows, normal, constant(1), orbit, 1, caps)) return {};
    try {
        Arithmetic algebra{orbit.mu, caps, caps.max_nonzeros};
        auto restriction = [&](const Expr& g) {
            // The degree in the single varying coordinate bounds the full
            // Taylor expansion along the line, including at a translated point.
            algebra.spend(g.terms().size());
            auto poly = spread(g, vars);
            unsigned degree = 0;
            for (const auto& [e, coefficient] : poly) {
                check(caps);
                height(coefficient);
                if (e[axis] > std::min(32u, caps.max_degree)) throw Unknown{};
                degree = std::max(degree, unsigned(e[axis].get_ui()));
            }
            return algebra.taylor(poly, orbit, {axis}, degree);
        };
        // The smooth one-dimensional germ contains this line, so it equals
        // the line locally. A nonzero row on the line does not certify that.
        for (const auto& row : rows)
            if (!restriction(row).empty()) return {};
        return !restriction(f).empty();
    } catch (const Unknown&) {
        return {};
    }
}
} // namespace atlas::certificate
