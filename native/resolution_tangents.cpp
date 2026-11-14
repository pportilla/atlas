#include <gmpxx.h>
#include "resolution_tangents.hpp"

#include <flint/fmpq_mpoly_factor.h>
#include <map>
#include <numeric>

namespace atlas {

// Filtering preserves the term order. Recompute the rational content after
// discarding terms, as the retained integer coefficients can have a common factor.
void lowest_form(fmpq_mpoly_t out, const fmpq_mpoly_t f, const std::vector<slong>& axes,
                 const fmpq_mpoly_ctx_t ctx) {
    Exponents powers(fmpq_mpoly_ctx_nvars(ctx));
    Z degree, minimum, coefficient;
    for (slong i = 0; i < fmpq_mpoly_length(f, ctx); ++i) {
        powers.read(f, i, ctx);
        fmpz_zero(&degree.v);
        for (auto axis : axes) fmpz_add(&degree.v, &degree.v, powers.entries[axis]);
        const int cmp = i ? fmpz_cmp(&degree.v, &minimum.v) : -1;
        if (cmp > 0) continue;
        if (cmp < 0) {
            minimum = degree;
            fmpq_mpoly_zero(out, ctx);
        }
        fmpz_mpoly_get_term_coeff_fmpz(&coefficient.v, f->zpoly, i, ctx->zctx);
        fmpz_mpoly_push_term_fmpz_fmpz(out->zpoly, &coefficient.v, powers.entries.data(), ctx->zctx);
    }
    fmpq_set(out->content, f->content);
    fmpq_mpoly_reduce(out, ctx);
}

std::string support_profile(const fmpq_mpoly_t f, const fmpq_mpoly_ctx_t ctx) {
    const slong n = fmpq_mpoly_ctx_nvars(ctx);
    Z degree;
    fmpq_mpoly_total_degree_fmpz(&degree.v, f, ctx);
    // The search regards zero as the singleton zero monomial.
    if (fmpz_sgn(&degree.v) < 0) fmpz_zero(&degree.v);
    char* raw = fmpz_get_str(nullptr, 10, &degree.v);
    std::string out(raw);
    flint_free(raw);
    out += " " + std::to_string(std::max<slong>(1, fmpq_mpoly_length(f, ctx)));
    std::vector<int> used(n);
    fmpq_mpoly_used_vars(used.data(), f, ctx);
    for (slong j = 0; j < n; ++j) if (used[j]) out += " " + std::to_string(j);
    return out + "\n";
}

std::string singular_subspaces(const fmpq_mpoly_t f, const std::vector<std::vector<slong>>& axes,
                               const fmpq_mpoly_ctx_t ctx) {
    const slong n = fmpq_mpoly_ctx_nvars(ctx);
    std::vector<bool> eligible(axes.size(), !fmpq_mpoly_is_zero(f, ctx));
    for (const auto& subset : axes)
        for (slong j : subset)
            if (j < 0 || j >= n) throw std::invalid_argument("Invalid coordinate subspace.");
    Exponents exponents(n);
    std::vector<int> truncated(n);
    for (slong i = 0; i < fmpq_mpoly_length(f, ctx); ++i) {
        exponents.read(f, i, ctx);
        for (slong j = 0; j < n; ++j)
            truncated[j] = fmpz_cmp_ui(exponents.entries[j], 1) > 0 ? 2 :
                           static_cast<int>(fmpz_get_ui(exponents.entries[j]));
        bool any = false;
        for (size_t k = 0; k < axes.size(); ++k) {
            if (!eligible[k]) continue;
            // Membership in the square of a coordinate ideal is termwise.
            int order = 0;
            for (slong j : axes[k]) {
                order += truncated[j];
                if (order >= 2) break;
            }
            eligible[k] = order >= 2;
            any = any || eligible[k];
        }
        if (!any) break;
    }
    std::string out;
    for (size_t i = 0; i < axes.size(); ++i) if (eligible[i]) out += std::to_string(i) + "\n";
    return out;
}

std::string repeated_directions(const fmpq_mpoly_t f, slong x, slong y,
                                const fmpq_mpoly_ctx_t ctx) {
    const slong n = fmpq_mpoly_ctx_nvars(ctx);
    if (x < 0 || x >= y || y >= n) throw std::invalid_argument("Invalid tangent axes.");
    if (fmpq_mpoly_is_zero(f, ctx)) return {};
    PolyValue tangent(ctx), common(ctx);
    lowest_form(tangent.p, f, {x, y}, ctx);
    std::vector<slong> base;
    for (slong j = 0; j < n; ++j) if (j != x && j != y) base.push_back(j);
    // Writing the tangent form by its base monomials, a repeated constant
    // section must divide every coefficient form in the two normal variables.
    if (base.empty()) {
        fmpq_mpoly_set(common.p, tangent.p, ctx);
    } else if (!fmpq_mpoly_content_vars(common.p, tangent.p, base.data(), base.size(), ctx)) {
        throw Limit("FLINT could not compute the common tangent form.");
    }

    struct Factors {
        fmpq_mpoly_factor_t v;
        const fmpq_mpoly_ctx_struct* ctx;
        explicit Factors(const fmpq_mpoly_ctx_struct* c) : ctx(c) { fmpq_mpoly_factor_init(v, c); }
        ~Factors() { fmpq_mpoly_factor_clear(v, ctx); }
    } factors(ctx);
    if (!fmpq_mpoly_factor(factors.v, common.p, ctx))
        throw Limit("FLINT could not factor the relative tangent form.");

    std::string result;
    std::vector<mpq_class> slopes;
    std::vector<ulong> axis(n);
    fmpq_t a, b;
    fmpq_init(a);
    fmpq_init(b);
    for (slong i = 0; i < factors.v->num; ++i) {
        if (fmpz_cmp_ui(factors.v->exp+i, 2) < 0) continue;
        auto factor = factors.v->poly+i;
        if (!fmpq_mpoly_total_degree_fits_si(factor, ctx) || fmpq_mpoly_total_degree_si(factor, ctx) != 1)
            continue;
        axis[x] = 1;
        fmpq_mpoly_get_coeff_fmpq_ui(a, factor, axis.data(), ctx);
        axis[x] = 0;
        axis[y] = 1;
        fmpq_mpoly_get_coeff_fmpq_ui(b, factor, axis.data(), ctx);
        axis[y] = 0;
        if (fmpq_is_zero(b)) {
            result += "vertical\n";
        } else {
            fmpq_div(a, a, b);
            fmpq_neg(a, a);
            fmpq_get_mpq(slopes.emplace_back().get_mpq_t(),a);
        }
    }
    fmpq_clear(a);
    fmpq_clear(b);
    std::sort(slopes.begin(),slopes.end());
    for (const auto& slope : slopes) result += slope.get_str() + "\n";
    return result;
}

namespace {

using IndexSet = std::vector<slong>;
bool advance(IndexSet& indices, slong limit) {
    for (slong j = static_cast<slong>(indices.size()) - 1; j >= 0; --j) {
        if (indices[j] == limit - static_cast<slong>(indices.size()) + j) continue;
        ++indices[j];
        for (size_t k = j+1; k < indices.size(); ++k) indices[k] = indices[k-1]+1;
        return true;
    }
    return false;
}

} // namespace

std::vector<PolyValue> reduced_minors(const std::vector<fmpq_mpoly_struct*>& matrix, slong rows,
                                      slong rank, std::vector<fmpq_mpoly_struct*> basis,
                                      const fmpq_mpoly_ctx_t ctx) {
    const slong columns = fmpq_mpoly_ctx_nvars(ctx);
    if (rank < 0 || rows < 0 || static_cast<size_t>(rows) * columns != matrix.size())
        throw std::invalid_argument("Invalid Jacobian matrix dimensions.");
    std::vector<PolyValue> result;
    if (rank > columns || rank > rows) return result;
    std::vector<PolyValue> scratch;
    std::vector<fmpq_mpoly_struct*> quotients;
    scratch.reserve(basis.size());
    for (size_t j = 0; j < basis.size(); ++j) quotients.push_back(scratch.emplace_back(ctx).p);

    // Laplace expansion in the quotient ring. Reducing at each level avoids
    // expanding strict transforms of high degree off the owned fibre.
    using Key = std::pair<IndexSet,IndexSet>;
    std::map<Key, PolyValue> cache;
    auto minor = [&](auto&& self, const IndexSet& r, const IndexSet& c) -> const PolyValue& {
        auto [entry, fresh] = cache.try_emplace({r, c}, ctx);
        auto& value = entry->second;
        if (!fresh) return value;
        if (r.empty()) {
            fmpq_mpoly_one(value.p, ctx);
            return value;
        }
        IndexSet subrows(r.begin(), r.end()-1);
        PolyValue term(ctx), sum(ctx);
        for (size_t j = 0; j < c.size(); ++j) {
            auto coefficient = matrix[r.back()*columns + c[j]];
            if (fmpq_mpoly_is_zero(coefficient, ctx)) continue;
            IndexSet subcolumns(c);
            subcolumns.erase(subcolumns.begin()+j);
            auto& previous = self(self, subrows, subcolumns);
            fmpq_mpoly_mul(term.p, coefficient, previous.p, ctx);
            if ((subrows.size()+j)%2) fmpq_mpoly_sub(sum.p, sum.p, term.p, ctx);
            else fmpq_mpoly_add(sum.p, sum.p, term.p, ctx);
        }
        if (basis.empty()) fmpq_mpoly_swap(value.p, sum.p, ctx);
        else fmpq_mpoly_divrem_ideal(quotients.data(), value.p, sum.p, basis.data(), basis.size(), ctx);
        return value;
    };
    IndexSet row(rank);
    std::iota(row.begin(), row.end(), 0);
    do {
        IndexSet column(rank);
        std::iota(column.begin(), column.end(), 0);
        do {
            const auto& value = minor(minor, row, column);
            if (fmpq_mpoly_is_zero(value.p, ctx)) continue;
            if (fmpq_mpoly_is_fmpq(value.p, ctx)) {
                result.clear();
                fmpq_mpoly_one(result.emplace_back(ctx).p, ctx);
                return result;
            }
            if (std::none_of(result.begin(), result.end(), [&](const auto& p) {
                    return fmpq_mpoly_equal(p.p, value.p, ctx); }))
                fmpq_mpoly_set(result.emplace_back(ctx).p, value.p, ctx);
        } while (advance(column, columns));
    } while (advance(row, rows));
    return result;
}

} // namespace atlas
