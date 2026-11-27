#include "membership_certificate.hpp"
#include "groebner.hpp"

#include <flint/fmpq_mat.h>
#include <flint/nmod.h>
#include <cstdlib>
#include <cstdio>
#include <limits>

namespace atlas::certificate {
using namespace chart;
namespace {
using Exponent = std::vector<unsigned>;
using ExactTerm = std::pair<Exponent, const Q*>;
using ExactRow = std::vector<std::pair<size_t, const Q*>>;
struct Row {
    size_t generator;
    Exponent shift;
    ExactRow terms;
};

void check(const CertificateCaps& caps) {
    if (caps.check_time) caps.check_time();
    atlas::check_deadline();
}

// A nonzero constant coefficient means the origin fails to be in the locus
// of this polynomial. Nonconstant terms of a polynomial have positive order.
bool has_constant(const Expr& f) {
    if (f.zero()) return false;
    const auto& e = f.terms().begin()->first;
    return std::all_of(e.begin(), e.end(), [](const auto& k) { return k == 0; });
}

bool read_terms(const Variables& over, const Expr& f, const CertificateCaps& caps,
                std::vector<ExactTerm>& out, unsigned& degree) {
    std::vector<size_t> axes;
    for (const auto& c : f.coords()) {
        auto it = std::find(over.begin(), over.end(), c);
        if (it == over.end()) return false;
        axes.push_back(it - over.begin());
    }
    for (const auto& [a, coefficient] : f.terms()) {
        unsigned total = 0;
        Exponent e(over.size());
        for (size_t j = 0; j < a.size(); ++j) {
            if (a[j] < 0 || a[j] > caps.max_degree - total) return false;
            e[axes[j]] = a[j].get_ui();
            total += e[axes[j]];
        }
        out.emplace_back(std::move(e), &coefficient);
        degree = std::max(degree, total);
    }
    return true;
}

// The number of monomials in t variables of degree at most d. A lower
// bound larger than cap is enough to abandon this degree.
size_t number_of_shifts(unsigned t, unsigned d, size_t cap) {
    size_t count = 1;
    for (unsigned j = 1; j <= std::min(t, d); ++j) {
        const size_t factor = std::max(t, d) + j;
        if (count > std::numeric_limits<size_t>::max() / factor) return cap + 1;
        count = count * factor / j;
        if (count > cap) return cap + 1;
    }
    return count;
}

template<class F>
bool shifts(Exponent& e, size_t i, unsigned d, F&& visit) {
    if (i == e.size()) return visit(e);
    for (unsigned j = 0; j <= d; ++j) {
        e[i] = j;
        if (!shifts(e, i + 1, d-j, visit)) return false;
    }
    return true;
}

struct RatMatrix {
    fmpq_mat_t m;
    RatMatrix(size_t r, size_t c) { fmpq_mat_init(m, r, c); }
    ~RatMatrix() { fmpq_mat_clear(m); }
    RatMatrix(const RatMatrix&) = delete;
};

bool verified_solution(const Variables& over, const Ideal& gens, const Expr& target,
                       const std::vector<Row>& rows, const ExactRow& last, size_t ncols,
                       const std::vector<size_t>& pivots, const std::vector<size_t>& multipliers,
                       const CertificateCaps& caps, MemberCertificate& result) {
    const size_t rank = pivots.size();
    if (!rank || rank > caps.max_rank || rank > caps.max_nonzeros / rank) return false;
    check(caps);
    std::vector<size_t> at(ncols, rank);
    for (size_t k = 0; k < rank; ++k) at[pivots[k]] = k;
    RatMatrix A(rank, rank), b(rank, 1), solution(rank, 1);
    for (size_t k = 0; k < rank; ++k)
        for (const auto& [j, value] : rows[multipliers[k]].terms)
            if (at[j] < rank) fmpq_set_mpq(fmpq_mat_entry(A.m, at[j], k), value->get_mpq_t());
    for (const auto& [j, value] : last)
        if (at[j] < rank) fmpq_set_mpq(fmpq_mat_entry(b.m, at[j], 0), value->get_mpq_t());
    // FLINT solves the shared-denominator equation. The nonsingular minor on
    // the full matrix was chosen over the finite field and need not give
    // the correct answers at the omitted coordinates.
    if (!fmpq_mat_solve_dixon(solution.m, A.m, b.m)) return false;
    std::vector<std::map<Powers, Q>> terms(gens.size());
    for (size_t k = 0; k < rank; ++k) {
        const auto& r = rows[multipliers[k]];
        Q value;
        fmpq_get_mpq(value.get_mpq_t(), fmpq_mat_entry(solution.m, k, 0));
        if (value != 0) terms[r.generator].emplace(Powers(r.shift.begin(), r.shift.end()), std::move(value));
    }
    Ideal cofactors;
    for (const auto& row : terms) cofactors.push_back(normal(over, row));
    check(caps);
    if (!verify_member(over, gens, target, cofactors)) return false;
    result.cofactors = std::move(cofactors);
    return true;
}

// Choose a pivot in a column containing the fewest nonzero coefficients.
size_t pivot_column(const std::vector<ulong>& v, const std::vector<size_t>& off) {
    size_t answer = v.size();
    for (size_t j = 0; j < v.size(); ++j)
        if (v[j] && (answer == v.size() || off[j] < off[answer])) {
            answer = j;
            if (off[j] == 1) break;
        }
    return answer;
}

bool solve_at_prime(const Variables& over, const Ideal& gens, const Expr& target, const std::vector<Row>& rows,
           const ExactRow& last, size_t ncols, const CertificateCaps& caps, ulong prime,
           MemberCertificate& result) {
    nmod_t field;
    nmod_init(&field, prime);
    std::vector<size_t> incidence(ncols);
    for (const auto& r : rows)
        for (const auto& entry : r.terms) ++incidence[entry.first];
    auto load_residues = [&](const ExactRow& h, std::vector<ulong>& v) {
        for (const auto& [j, value] : h) {
            const ulong den = mpz_fdiv_ui(value->get_den_mpz_t(), prime);
            if (!den) return false;
            const ulong num = mpz_fdiv_ui(value->get_num_mpz_t(), prime);
            v[j] = den == 1 ? num : nmod_mul(num, nmod_inv(den, field), field);
        }
        return true;
    };
    using Polynomial = std::vector<std::pair<size_t, ulong>>;
    std::vector<Polynomial> reduced;
    std::vector<size_t> pivots, multipliers;
    std::vector<ulong> v(ncols);
    size_t remaining = caps.max_nonzeros;
    auto reduce = [&](const ExactRow& h) {
        std::fill(v.begin(), v.end(), 0);
        if (!load_residues(h, v)) return false;
        for (size_t k = 0; k < reduced.size(); ++k) {
            const ulong scalar = v[pivots[k]];
            if (!scalar) continue;
            for (const auto& [j,c] : reduced[k]) v[j] = nmod_sub(v[j], nmod_mul(scalar, c, field), field);
        }
        return true;
    };
    for (size_t i = 0; i < rows.size(); ++i) {
        check(caps);
        if (!reduce(rows[i].terms)) return false;
        const size_t col = pivot_column(v, incidence);
        if (col == ncols) continue;
        if (pivots.size() == caps.max_rank) return false;
        const ulong inverse = nmod_inv(v[col], field);
        Polynomial next;
        for (size_t j = 0; j < ncols; ++j) if (v[j]) {
            if (!remaining) return false;
            --remaining;
            next.emplace_back(j, nmod_mul(v[j], inverse, field));
        }
        pivots.push_back(col);
        multipliers.push_back(i);
        reduced.push_back(std::move(next));
    }
    if (!reduce(last) || std::any_of(v.begin(), v.end(), [](ulong c) { return c != 0; })) return false;
    return verified_solution(over, gens, target, rows, last, ncols, pivots, multipliers, caps, result);
}
}  // namespace

bool verify_member(const Variables& input, const Ideal& gens, const Expr& target, const Ideal& cofactors) {
    if (gens.size() != cofactors.size()) return false;
    auto over = input;
    std::sort(over.begin(), over.end());
    flint_q::Context ctx(over.size());
    flint_q::Poly g(ctx), a(ctx), term(ctx), sum(ctx), f(ctx);
    if (!flint_q::load(f, target, over)) return false;
    for (size_t i = 0; i < gens.size(); ++i) {
        if (cofactors[i].zero()) continue;
        fmpq_mpoly_zero(g.p, ctx.ctx);
        fmpq_mpoly_zero(a.p, ctx.ctx);
        if (!flint_q::load(g, gens[i], over) || !flint_q::load(a, cofactors[i], over)) return false;
        fmpq_mpoly_mul(term.p, a.p, g.p, ctx.ctx);
        fmpq_mpoly_add(sum.p, sum.p, term.p, ctx.ctx);
    }
    return fmpq_mpoly_equal(sum.p, f.p, ctx.ctx);
}

MemberCertificate certify_member(const Variables& over, const Ideal& gens, const Expr& target,
                                 const CertificateCaps& caps) {
    check(caps);
    if (over.size() > 32 || caps.max_degree > 65535 || !caps.max_rows || !caps.max_cols ||
        caps.max_rows > 50000 || caps.max_cols > 50000 || !caps.max_nonzeros || gens.size() > caps.max_rows)
        return {};
    // No need to open a search if the origin is a witness of nonmembership.
    if (gens.empty() && !target.zero()) return {};
    if (has_constant(target) && std::none_of(gens.begin(), gens.end(), has_constant)) return {};
    size_t count = target.terms().size();
    if (count > caps.max_nonzeros) return {};
    for (const auto& f : gens) {
        if (count > caps.max_nonzeros || f.terms().size() > caps.max_nonzeros - count) return {};
        count += f.terms().size();
    }
    if (has_constant(target) && !std::any_of(gens.begin(), gens.end(), has_constant)) return {};
    std::vector<std::vector<ExactTerm>> basis(gens.size());
    std::vector<unsigned> degrees(gens.size());
    std::vector<ExactTerm> parts;
    unsigned start = 0;
    if (!read_terms(over, target, caps, parts, start)) return {};
    for (size_t j = 0; j < gens.size(); ++j) {
        if (!read_terms(over, gens[j], caps, basis[j], degrees[j])) return {};
        start = std::max(start, degrees[j]);
    }
    if (target.zero()) return {true, Ideal(gens.size()), start, {}};
    for (size_t j = 0; j < gens.size(); ++j) {
        if (!gens[j].zero() && (gens[j].number() || gens[j] == target)) {
            Ideal one(gens.size());
            one[j] = gens[j].number() ? scaled(target, Q(1) / gens[j].scalar()) : constant(1);
            if (verify_member(over, gens, target, one)) return {true, std::move(one), start, {}};
        }
    }
    if (gens.empty()) return {};
    for (unsigned d = start; d - start <= caps.degree_slack && d <= caps.max_degree; ++d) {
        check(caps);
        size_t row_count = 0, nonzeros = 0;
        for (size_t j = 0; j < gens.size(); ++j) if (!basis[j].empty()) {
            const size_t n = number_of_shifts(over.size(), d - degrees[j], caps.max_rows - row_count);
            if (n > caps.max_rows - row_count || n > (caps.max_nonzeros - nonzeros) / basis[j].size())
                return {};
            row_count += n;
            nonzeros += n * basis[j].size();
        }
        std::map<Exponent, size_t> columns;
        auto extend = [&](const Exponent& shift, const std::vector<ExactTerm>& terms, ExactRow& cells) {
            for (const auto& [a, value] : terms) {
                Exponent e = shift;
                for (size_t k = 0; k < a.size(); ++k) e[k] += a[k];
                auto pos = columns.lower_bound(e);
                if (pos == columns.end() || pos->first != e) {
                    if (columns.size() == caps.max_cols) return false;
                    pos = columns.emplace_hint(pos, std::move(e), columns.size());
                }
                cells.emplace_back(pos->second, value);
            }
            return true;
        };
        Exponent shift(over.size(), 0);
        ExactRow last;
        if (!extend(shift, parts, last)) return {};
        std::vector<Row> rows;
        rows.reserve(row_count);
        for (size_t j = 0; j < gens.size(); ++j) {
            if (basis[j].empty()) continue;
            if (!shifts(shift, 0, d - degrees[j], [&](const Exponent& e) {
                rows.push_back({j, e, {}});
                return extend(e, basis[j], rows.back().terms);
            })) return {};
        }
        MemberCertificate result;
        // A noninvertible denominator or an unlucky minor can vanish at one prime.
#if FLINT_BITS == 64
        static const ulong primes[] = {4611686018427387847UL, 4611686018427387817UL};
#else
        static const ulong primes[] = {2147483647UL, 2147483629UL};
#endif
        for (auto p : primes) if (solve_at_prime(over, gens, target, rows, last, columns.size(), caps, p, result)) {
            result.degree = d;
            result.certified = true;
            return result;
        }
    }
    return {};
}

MemberCertificate certify_unit(const Variables& over, const Ideal& gens, const CertificateCaps& caps) {
    return certify_member(over, gens, constant(1), caps);
}

MemberCertificate certify_empty_on_open(const Variables& over, const Ideal& gens, const Expr& unit,
                                       const CertificateCaps& caps) {
    if (unit.number() && !unit.zero()) return certify_unit(over, gens, caps);
    // A point is on the open exactly when it lifts to u * t = 1.
    if (unit.terms().size() >= caps.max_nonzeros || over.size() >= 32) return {};
    if (!unit.zero() && has_constant(unit) && !std::any_of(gens.begin(), gens.end(), has_constant)) return {};
    Coord t{"t", 0};
    while (std::find(over.begin(), over.end(), t) != over.end()) ++t.identity;
    auto both = over;
    both.push_back(t);
    Ideal grid = gens;
    grid.push_back(symbol(t) * unit - constant(1));
    auto result = certify_unit(both, grid, caps);
    if (result.certified) result.auxiliary = t;
    return result;
}
MemberCertificate certify_radical_member(const Variables& over, const Ideal& gens, const Expr& f,
                                        const CertificateCaps& caps) {
    return certify_empty_on_open(over, gens, f, caps);
}

bool enabled() {
    const char* value = std::getenv("ATLAS_LINEAR_ALGEBRA");
    return !value || std::string_view(value) != "0";
}
bool trace(const char* site, bool certified) {
    if (std::getenv("ATLAS_TRACE_CERTIFICATES"))
        std::fprintf(stderr, "ATLAS_CERTIFICATE %s %s\n", site, certified ? "certified" : "unknown");
    return certified;
}
}  // namespace atlas::certificate
