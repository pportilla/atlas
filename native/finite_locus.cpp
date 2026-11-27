#include "finite_locus.hpp"

#include "groebner.hpp"
#include "local_monomial.hpp"
#include "primary_decomposition.hpp"
#include "separating_form.hpp"
#include <flint/fmpq_mat.h>
#include <flint/fmpq_poly.h>
#include <flint/fmpz_poly.h>
#include <flint/fmpz_poly_factor.h>

namespace atlas::certificate {
namespace {
using namespace chart;
using local::Polynomial;
using Vector = std::vector<Q>;

struct Limit {};
struct PositiveDimension {};
void check(const CertificateCaps& cap) {
    if (cap.check_time) cap.check_time();
    atlas::check_deadline();
}
struct ExactMatrix {
    fmpq_mat_t a;
    ExactMatrix(size_t n, size_t m) { fmpq_mat_init(a, n, m); }
    ExactMatrix(const ExactMatrix& other) { fmpq_mat_init_set(a, other.a); }
    ~ExactMatrix() { fmpq_mat_clear(a); }
    ExactMatrix& operator=(const ExactMatrix&) = delete;
};
struct QPoly {
    fmpq_poly_t p;
    QPoly() { fmpq_poly_init(p); }
    ~QPoly() { fmpq_poly_clear(p); }
    QPoly(const QPoly&) = delete;
};
Q get(const fmpq_t a) {
    Q q;
    fmpq_get_mpq(q.get_mpq_t(), a);
    return q;
}
bool supported(const Expr& g, const CertificateCaps& cap, size_t& budget) {
    if (g.terms().size() > budget) return false;
    budget -= g.terms().size();
    for (const auto& [e, q] : g.terms()) {
        if (std::any_of(e.begin(), e.end(), [](const auto& j) { return j < 0; }) ||
            local::total_degree(e) > cap.max_degree ||
            mpz_sizeinbase(q.get_num_mpz_t(), 2) > 256 || mpz_sizeinbase(q.get_den_mpz_t(), 2) > 256)
            return false;
    }
    return true;
}

class Multiplication {
    const std::vector<local::Polynomial>& basis;
    const CertificateCaps& caps;
    size_t transitions;
    unsigned width;
    std::map<Powers, size_t> basis_index;
    bool standard(const Powers& e) const {
        return std::all_of(basis.begin(), basis.end(), [&](const Polynomial& g) {
            return !local::monomial_divides(g.rbegin()->first, e);
        });
    }
public:
    std::vector<ExactMatrix> multiples;
    slong dimension = 0;

    Multiplication(const local::Basis& b, const CertificateCaps& c, unsigned n)
        : basis(b), caps(c), transitions(std::min<size_t>(c.max_nonzeros, 262144)), width(n) {
        if (b.empty()) throw PositiveDimension{};
        // A capped pure power of each variable in the leading ideal gives a
        // finite quotient. Determine the actual standard monomials by walking
        // outwards from 1 and stopping at leading monomials.
        const size_t maximum = std::min<size_t>({32, caps.max_rank, caps.max_rows, caps.max_cols});
        std::vector<Powers> columns(1, Powers(width));
        basis_index.emplace(columns[0], 0);
        for (unsigned j = 0; j < width; ++j) {
            const bool bounded = std::any_of(basis.begin(), basis.end(), [&](const Polynomial& g) {
                const auto& lead = g.rbegin()->first;
                return lead[j] != 0 && local::total_degree(lead) == lead[j];
            });
            if (!bounded) throw PositiveDimension{};
        }
        for (size_t k = 0; k < columns.size(); ++k) {
            check(caps);
            for (unsigned j = 0; j < width; ++j) {
                auto e = columns[k];
                ++e[j];
                if (basis_index.count(e) || !standard(e)) continue;
                if (columns.size() >= maximum || e[j] > caps.max_degree) throw Limit{};
                basis_index.emplace(e, columns.size());
                columns.push_back(std::move(e));
            }
        }
        dimension = columns.size();
        // Count the multiplication matrices and the subsequent trial's
        // work matrices before assigning any modulus or allocating a column.
        if (!maximum || (width + 9) * columns.size() * columns.size() > caps.max_nonzeros) throw Limit{};
        multiples.reserve(width);
        for (unsigned j = 0; j < width; ++j) {
            auto& matrix = multiples.emplace_back(dimension, dimension);
            for (size_t k = 0; k < columns.size(); ++k) {
                auto e = columns[k];
                ++e[j];
                reduce(std::move(e), k, matrix);
            }
        }
    }
    void reduce(Powers e, size_t col, ExactMatrix& matrix) {
        Polynomial remaining;
        remaining.emplace(std::move(e), Q(1));
        while (!remaining.empty()) {
            check(caps);
            const auto [exp, q] = *remaining.rbegin();
            if (const auto it = basis_index.find(exp); it != basis_index.end()) {
                fmpq_set_mpq(fmpq_mat_entry(matrix.a, it->second, col), q.get_mpq_t());
                remaining.erase(exp);
                continue;
            }
            const auto divisor = std::find_if(basis.begin(), basis.end(), [&](const Polynomial& g) {
                return local::monomial_divides(g.rbegin()->first, exp);
            });
            if (divisor == basis.end() || divisor->size() > transitions) throw Limit{};
            transitions -= divisor->size();
            const auto& [lead, lc] = *divisor->rbegin();
            const Q scale = q / lc;
            for (const auto& [v, c] : *divisor) {
                Powers p(width);
                for (unsigned j = 0; j < width; ++j) p[j] = exp[j] - lead[j] + v[j];
                if (local::total_degree(p) > caps.max_degree * size_t(2)) throw Limit{};
                auto& z = remaining[p];
                z -= scale * c;
                if (z == 0) remaining.erase(p);
            }
        }
    }
};

std::vector<std::pair<Vector,long>> factor(const QPoly& chi) {
    struct Factors {
        fmpz_poly_t polynomial;
        fmpz_poly_factor_t split;
        Factors() { fmpz_poly_init(polynomial); fmpz_poly_factor_init(split); }
        ~Factors() { fmpz_poly_factor_clear(split); fmpz_poly_clear(polynomial); }
    } exact;
    fmpq_poly_get_numerator(exact.polynomial, chi.p);
    if (fmpz_poly_max_bits(exact.polynomial) > 1024 || fmpz_poly_max_bits(exact.polynomial) < -1024) throw Limit{};
    fmpz_poly_factor(exact.split, exact.polynomial);
    std::vector<std::pair<Vector,long>> out;
    for (slong i = 0; i < exact.split->num; ++i) {
        const auto f = exact.split->p + i;
        Vector v;
        for (slong j = 0; j <= fmpz_poly_degree(f); ++j) {
            chart::Z z;
            fmpz_get_mpz(z.get_mpz_t(), f->coeffs + j);
            v.emplace_back(z);
        }
        const Q lc = v.back();
        for (auto& k : v) k /= lc;
        out.emplace_back(std::move(v), exact.split->exp[i]);
    }
    return out;
}

void add_axis(ExactMatrix& out, const ExactMatrix& f, long scalar) {
    auto n = fmpq_mat_nrows(f.a);
    ExactMatrix v(n, n);
    fmpq_mat_scalar_mul_si(v.a, f.a, scalar);
    fmpq_mat_add(out.a, out.a, v.a);
}
void assign(QPoly& p, const Vector& v) {
    fmpq_t q;
    fmpq_init(q);
    for (size_t k = 0; k < v.size(); ++k) {
        fmpq_set_mpq(q, v[k].get_mpq_t());
        fmpq_poly_set_coeff_fmpq(p.p, k, q);
    }
    fmpq_clear(q);
}
Vector reduced(const Vector& v, const GaloisOrbit& o) {
    QPoly p, u;
    assign(p, v);
    assign(u, o.mu);
    fmpq_poly_rem(p.p, p.p, u.p);
    Vector out;
    fmpq_t q;
    fmpq_init(q);
    for (slong i = 0; i <= fmpq_poly_degree(p.p); ++i) {
        fmpq_poly_get_coeff_fmpq(q, p.p, i);
        out.push_back(get(q));
    }
    fmpq_clear(q);
    return out;
}

bool vanishes(const Polynomial& g, const GaloisOrbit& o) {
    QPoly sum, mu;
    assign(mu, o.mu);
    for (const auto& [e,c] : g) {
        QPoly partial;
        assign(partial, {c});
        for (size_t j = 0; j < e.size(); ++j) {
            if (!e[j]) continue;
            QPoly parent, x;
            assign(parent, o.coordinates[j]);
            for (unsigned long k = 0; k < e[j].get_ui(); ++k) {
                fmpq_poly_mul(x.p, partial.p, parent.p);
                fmpq_poly_rem(partial.p, x.p, mu.p);
            }
        }
        fmpq_poly_add(sum.p, sum.p, partial.p);
    }
    return fmpq_poly_is_zero(sum.p);
}

// On every component x_j = r_j(L) precisely when the multiplication matrix
// of x_j - r_j(L) acts nilpotently. Merely proving that the zero set of an
// eliminant contains the finite locus misses points when L does not separate.
bool agrees_on_all_points(const ExactMatrix& frame, const ExactMatrix& target, const Vector& v) {
    const auto n = fmpq_mat_nrows(frame.a);
    ExactMatrix test(n,n), q(n,n);
    fmpq_t value;
    fmpq_init(value);
    for (size_t i = v.size(); i-- > 0;) {
        fmpq_mat_mul(q.a, test.a, frame.a);
        fmpq_set_mpq(value, v[i].get_mpq_t());
        for (slong k = 0; k < n; ++k)
            fmpq_add(fmpq_mat_entry(q.a, k, k), fmpq_mat_entry(q.a,k,k), value);
        fmpq_mat_swap(test.a, q.a);
    }
    fmpq_clear(value);
    fmpq_mat_sub(test.a, test.a, target.a);
    QPoly end;
    fmpq_mat_charpoly(end.p, test.a);
    return end.p->length == n+1 && fmpz_cmp_ui(end.p->coeffs + n, 1) == 0 &&
           std::all_of(end.p->coeffs, end.p->coeffs+n, [](const fmpz& t) { return fmpz_is_zero(&t); });
}

std::optional<FiniteLocus> joint(const Multiplication& product, const local::Basis& eqs,
                                 const local::Basis& units, const CertificateCaps& caps, size_t trial) {
    check(caps);
    const auto width = product.multiples.size();
    const auto d = product.dimension;
    ExactMatrix frame(d,d);
    const auto weights = atlas::separating_weights(width, trial);
    for (size_t j = 0; j < width; ++j) add_axis(frame, product.multiples[j], weights[j]);
    QPoly chi;
    fmpq_mat_charpoly(chi.p, frame.a);
    auto factors = factor(chi);
    size_t count = 0;
    for (const auto& term : factors) count += term.first.size()-1;
    // A trace of L^k is the sum of its eigenvalues to the kth power, with
    // multiplicities. The product with x_j selects the corresponding linear
    // combination of coordinate values. The resulting exact matrix solves
    // for x_j = r_j(L) in the quotient by the nilradical.
    ExactMatrix moments(count,count), values(count,width), answers(count,width), u(d,d), v(d,d);
    fmpq_mat_one(u.a);
    for (size_t k = 0; k < 2*count-1; ++k) {
        check(caps);
        fmpq_t tr;
        fmpq_init(tr);
        fmpq_mat_trace(tr, u.a);
        for (size_t j = 0; j < count; ++j)
            if (j <= k && k-j < count) fmpq_set(fmpq_mat_entry(moments.a,j,k-j), tr);
        if (k < count) {
            for (size_t j = 0; j < width; ++j) {
                fmpq_mat_mul(v.a, u.a, product.multiples[j].a);
                fmpq_mat_trace(fmpq_mat_entry(values.a, k, j), v.a);
            }
        }
        fmpq_mat_mul(v.a, u.a, frame.a);
        fmpq_mat_swap(u.a, v.a);
        fmpq_clear(tr);
    }
    if (!fmpq_mat_solve(answers.a, moments.a, values.a)) return {};
    std::vector<Vector> coefficients(width);
    for (size_t j = 0; j < width; ++j) {
        auto& entries = coefficients[j];
        for (size_t k = 0; k < count; ++k) entries.push_back(get(fmpq_mat_entry(answers.a,k,j)));
        while (!entries.empty() && entries.back() == 0) entries.pop_back();
        if (!agrees_on_all_points(frame, product.multiples[j], entries)) return {};
    }
    FiniteLocus out{true, {}};
    for (auto& component : factors) {
        GaloisOrbit o;
        o.mu = std::move(component.first);
        for (const auto& k : coefficients) o.coordinates.push_back(reduced(k, o));
        check(caps);
        for (const auto& h : eqs) if (!vanishes(h, o)) return {};
        if (std::none_of(units.begin(), units.end(), [&](const auto& h) { return vanishes(h, o); }))
            out.orbits.push_back(std::move(o));
    }
    return out;
}

FiniteLocus frame(const local::Basis& equations, const local::Basis& units, unsigned width,
                  const CertificateCaps& caps) {
    const unsigned long safe = std::min<size_t>(caps.max_nonzeros, 524288);
    if (!safe || width == 0 || width > 8) return {};
    atlas::primary::CertificateBudget budget(safe);
    const auto band = atlas::primary::reduced(equations, width, true);
    if (band.size() == 1 && local::max_total_degree(band.front()) == 0) return {true, {}};
    const Multiplication bounds(band, caps, width);
    for (size_t i = 0; i < 3; ++i)
        if (auto result = joint(bounds, equations, units, caps, i)) return *result;
    return {};
}
FiniteLocus bounded_locus(const Variables& given, const Ideal& eqs, const Ideal& units, const CertificateCaps& caps) {
    check(caps);
    if (given.empty() || given.size() > 8 || eqs.size() > caps.max_rows || units.size() > caps.max_cols ||
        given.size() > caps.max_cols) return {};
    size_t count = caps.max_nonzeros;
    local::Basis rows, tests;
    for (const auto& e : eqs) {
        if (!supported(e,caps,count)) return {};
        if (!e.zero() && e.number()) return {true, {}};
        if (!e.zero()) rows.push_back(spread(e, given));
    }
    for (const auto& u : units) {
        if (!supported(u,caps,count)) return {};
        if (u.zero() || std::find(eqs.begin(), eqs.end(), u) != eqs.end()) return {true, {}};
        if (!u.number()) tests.push_back(spread(u, given));
    }
    // With fewer independent equations than variables, every nonempty
    // component has positive dimension, also after restricting to an open.
    if (rows.size() < given.size()) return {};
    bool positive_dimension = false;
    auto budgeted = [&](const local::Basis& givens, const local::Basis& open, size_t width) {
        try {
            return frame(givens, open, width, caps);
        } catch (const PositiveDimension&) {
            positive_dimension = true;
            return FiniteLocus{};
        } catch (const Limit&) {
            return FiniteLocus{};
        } catch (const atlas::Limit&) {
            return FiniteLocus{};
        } catch (const atlas::Expired&) {
            throw;
        } catch (const std::runtime_error&) {
            check(caps);  // allow the time limit to propagate
            return FiniteLocus{};
        }
    };
    auto result = budgeted(rows, tests, given.size());
    if (result.certified || !positive_dimension || tests.empty() || given.size() + tests.size() > 8 ||
        rows.size() + tests.size() > caps.max_rows) return result;
    // A component with u_i=0 may keep the closed scheme infinite even when
    // the stated open is finite. Introduce t_i*u_i-1 at that stage. Every
    // remaining point extends uniquely, including its coordinate field.
    const size_t n = given.size(), width = n + tests.size();
    auto extend = [&](const Polynomial& g, size_t index) {
        Polynomial h;
        for (const auto& [e,q] : g) {
            auto a = e;
            a.resize(width);
            if (index < tests.size()) a[n+index] = 1;
            h.emplace(std::move(a), q);
        }
        return h;
    };
    local::Basis lifted;
    for (const auto& e : rows) lifted.push_back(extend(e, tests.size()));
    for (size_t j = 0; j < tests.size(); ++j) {
        auto w = extend(tests[j], j);
        w.emplace(Powers(width), chart::Q(-1));
        lifted.push_back(std::move(w));
    }
    result = budgeted(lifted, {}, width);
    for (auto& o : result.orbits) o.coordinates.resize(n);
    return result;
}
}  // namespace

FiniteLocus finite_locus(const Variables& given, const Ideal& equations, const Ideal& units,
                         const CertificateCaps& caps) {
    try {
        atlas::DeadlineScope tentative(std::chrono::steady_clock::now() + std::chrono::milliseconds(12));
        return bounded_locus(given, equations, units, caps);
    } catch (const atlas::Expired&) {
        check(caps);  // the caller's deadline is now restored
        return {};
    }
}
}  // namespace atlas::certificate
