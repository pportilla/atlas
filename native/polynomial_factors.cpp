#include "polynomial_factors.hpp"
#include "polynomial_gcd.hpp"
#include "flint_polynomial.hpp"
#include "triton_factor_division.hpp"

namespace atlas {
namespace {

std::string decimal(const Z& value) {
    auto raw = fmpz_get_str(nullptr, 10, &value.v);
    std::string text(raw);
    flint_free(raw);
    return text;
}

bool has_linear_variable(const fmpq_mpoly_t polynomial, const fmpq_mpoly_ctx_t ctx) {
    const slong n = fmpq_mpoly_ctx_nvars(ctx);
    std::vector<bool> seen(n), linear(n);
    Exponents powers(n);
    for (slong i = 0; i < fmpq_mpoly_length(polynomial, ctx); ++i) {
        powers.read(polynomial, i, ctx);
        slong width = 0;
        for (const auto power : powers.entries)
            if (!fmpz_is_zero(power)) ++width;
        for (slong j = 0; j < n; ++j) {
            if (fmpz_is_zero(powers.entries[j])) continue;
            linear[j] = !seen[j] && width == 1 && fmpz_is_one(powers.entries[j]);
            seen[j] = true;
        }
    }
    // Solving c*x_j + h = 0 gives the polynomial ring in the remaining axes.
    return std::find(linear.begin(), linear.end(), true) != linear.end();
}

bool factor_primitive_binomial(fmpq_mpoly_factor_t out, const fmpq_mpoly_t f,
                               const fmpq_mpoly_ctx_t ctx) {
    if (fmpq_mpoly_length(f, ctx) != 2) return false;
    const slong n = fmpq_mpoly_ctx_nvars(ctx);
    Exponents first(n), second(n), common(n);
    first.read(f, 0, ctx);
    second.read(f, 1, ctx);
    Z gcd, difference;
    for (slong j = 0; j < n; ++j) {
        fmpz_sub(&difference.v, first.entries[j], second.entries[j]);
        fmpz_gcd(&gcd.v, &gcd.v, &difference.v);
        common.values[j] = fmpz_cmp(first.entries[j], second.entries[j]) < 0 ? first.values[j] : second.values[j];
    }
    // After removing the common monomial a primitive exponent difference is
    // a lattice basis element: the Laurent binomial is a unit times T-c.
    if (!fmpz_is_one(&gcd.v)) return false;
    fmpq_mpoly_factor_fit_length(out, n + 1, ctx);
    for (slong j = 0; j < n; ++j) {
        if (fmpz_is_zero(common.entries[j])) continue;
        fmpq_mpoly_gen(out->poly + out->num, j, ctx);
        fmpz_set(out->exp + out->num++, common.entries[j]);
    }
    auto reduced = out->poly + out->num;
    fmpz_one(out->exp + out->num++);
    fmpq_mpoly_make_monic(reduced, f, ctx);
    for (slong j = 0; j < n; ++j) {
        fmpz_sub(first.entries[j], first.entries[j], common.entries[j]);
        fmpz_sub(second.entries[j], second.entries[j], common.entries[j]);
    }
    fmpq_mpoly_set_term_exp_fmpz(reduced, 0, first.entries.data(), ctx);
    fmpq_mpoly_set_term_exp_fmpz(reduced, 1, second.entries.data(), ctx);
    return true;
}

} // namespace

const char* ordered_factors(fmpq_mpoly_factor_t out, const fmpq_mpoly_t f,
                            const fmpq_mpoly_ctx_t ctx) {
    if (fmpq_mpoly_is_zero(f, ctx))
        throw std::invalid_argument("The zero polynomial has no irreducible factorization.");
    // A failed FLINT attempt may have populated part of the output.
    out->num = 0;
    fmpq_mpoly_get_term_coeff_fmpq(out->constant, f, 0, ctx);
    if (fmpq_mpoly_is_fmpq(f, ctx)) return "rational_constant";
    if (has_linear_variable(f, ctx)) {
        fmpq_mpoly_factor_fit_length(out, 1, ctx);
        out->num = 1;
        fmpq_mpoly_make_monic(out->poly, f, ctx);
        fmpz_one(out->exp);
        return "linear_variable";
    }
    if (factor_primitive_binomial(out, f, ctx)) return "primitive_binomial";
    if (triton_linear_factors(out, f, ctx)) return "modular_root_screen";
    if (fmpq_mpoly_factor(out, f, ctx) && fmpq_mpoly_factor_make_monic(out, ctx))
        return "flint";
    exhaustive_rational_factors(out, f, ctx);
    return "sparse_kronecker";
}

const char* squarefree_factors(fmpq_mpoly_factor_t out, const fmpq_mpoly_t f, const fmpq_mpoly_ctx_t ctx) {
    if (fmpq_mpoly_is_zero(f,ctx)) throw std::invalid_argument("The zero polynomial has no square-free factorization.");
    if (fmpq_mpoly_is_fmpq(f,ctx)) return ordered_factors(out,f,ctx);
    const char* route;
    if (triton_linear_factors(out,f,ctx,true)) route = "modular_root_screen";
    else if (fmpq_mpoly_degrees_fit_si(f,ctx) && fmpq_mpoly_factor_squarefree(out,f,ctx)
             && fmpq_mpoly_factor_make_monic(out,ctx)) route = "flint_squarefree";
    else route = ordered_factors(out,f,ctx);
    // FLINT's monomial factors and the screened linear factors can contribute
    // to the same multiplicity. Compact them without narrowing their GMP powers.
    slong count=0;
    for (slong i=0;i<out->num;++i) {
        slong j=0;
        while (j<count && !fmpz_equal(out->exp+j,out->exp+i)) ++j;
        if (j<count) fmpq_mpoly_mul(out->poly+j,out->poly+j,out->poly+i,ctx);
        else {
            if (count!=i) {
                fmpq_mpoly_swap(out->poly+count,out->poly+i,ctx);
                fmpz_swap(out->exp+count,out->exp+i);
            }
            ++count;
        }
    }
    out->num=count;
    return route;
}

void rational_squarefree_part(fmpq_mpoly_t out, const fmpq_mpoly_t f, const fmpq_mpoly_ctx_t ctx) {
    if (fmpq_mpoly_is_zero(f, ctx))
        throw std::invalid_argument("Expected a nonzero polynomial for its squarefree part.");
    if (fmpq_mpoly_length(f, ctx) == 1) {
        Exponents powers(fmpq_mpoly_ctx_nvars(ctx));
        powers.read(f, 0, ctx);
        for (auto exponent : powers.entries)
            if (!fmpz_is_zero(exponent)) fmpz_one(exponent);
        fmpq_mpoly_one(out, ctx);
        fmpq_mpoly_set_term_exp_fmpz(out, 0, powers.entries.data(), ctx);
        return;
    }
    PolyValue common(ctx), derivative(ctx);
    fmpq_mpoly_set(common.p, f, ctx);
    for (slong j = 0; j < fmpq_mpoly_ctx_nvars(ctx) && !fmpq_mpoly_is_fmpq(common.p, ctx); ++j) {
        fmpq_mpoly_derivative(derivative.p, f, j, ctx);
        monic_polynomial_gcd(out, common.p, derivative.p, ctx);
        fmpq_mpoly_swap(out, common.p, ctx);
    }
    if (!fmpq_mpoly_divides(out, f, common.p, ctx)) throw Limit("Inexact squarefree quotient.");
    fmpq_mpoly_make_monic(out, out, ctx);
}

std::string factor_multiplicity(const fmpq_mpoly_t value, const fmpq_mpoly_t divisor,
                                slong axes, std::vector<fmpq_mpoly_struct*> field_basis,
                                const fmpq_mpoly_ctx_t ctx) {
    const slong n = fmpq_mpoly_ctx_nvars(ctx);
    if (axes < 1 || (axes != n && axes != n-1) || fmpq_mpoly_is_zero(divisor, ctx))
        throw std::invalid_argument("Expected a divisor in its coordinate space.");
    if (fmpq_mpoly_is_zero(value, ctx))
        throw std::invalid_argument("The zero polynomial has no finite factor multiplicity.");

    Exponents head(n), powers(n);
    head.read(divisor, 0, ctx);
    bool monomial = true;
    for (slong i = 1; monomial && i < fmpq_mpoly_length(divisor, ctx); ++i) {
        powers.read(divisor, i, ctx);
        for (slong j = 0; j < axes; ++j)
            if (!fmpz_equal(powers.entries[j], head.entries[j])) monomial = false;
    }
    Z multiplicity;
    if (monomial) {
        // Each nonzero field coefficient is a unit. Only the ambient powers
        // enter the valuation, including when the coefficient has many terms.
        bool found = false;
        Z quotient;
        for (slong i = 0; i < fmpq_mpoly_length(value, ctx); ++i) {
            powers.read(value, i, ctx);
            for (slong j = 0; j < axes; ++j) {
                if (fmpz_is_zero(head.entries[j])) continue;
                fmpz_fdiv_q(&quotient.v, powers.entries[j], head.entries[j]);
                if (!found || fmpz_cmp(&quotient.v, &multiplicity.v) < 0)
                    multiplicity = quotient;
                found = true;
            }
        }
        if (!found) throw std::invalid_argument("Multiplicity requires a nonconstant polynomial factor.");
        return decimal(multiplicity);
    }

    if (axes < n && field_basis.empty()) {
        std::vector<int> used(n);
        fmpq_mpoly_used_vars(used.data(), divisor, ctx);
        if (used.back()) return "field_basis";
    }
    PolyValue remaining(ctx), quotient(ctx), relation_quotient(ctx), remainder(ctx);
    fmpq_mpoly_set(remaining.p, value, ctx);
    if (!field_basis.empty() && field_basis.size() != 2)
        throw std::invalid_argument("Expected the field divisor and its defining relation.");
    while (true) {
        if (field_basis.empty()) {
            if (!fmpq_mpoly_divides(quotient.p, remaining.p, divisor, ctx)) break;
        } else {
            fmpq_mpoly_struct* quotients[] = {quotient.p, relation_quotient.p};
            fmpq_mpoly_divrem_ideal(quotients, remainder.p, remaining.p,
                                    field_basis.data(), 2, ctx);
            if (!fmpq_mpoly_is_zero(remainder.p, ctx)) break;
            fmpq_mpoly_divrem(relation_quotient.p, remainder.p, quotient.p, field_basis[1], ctx);
            fmpq_mpoly_swap(quotient.p, remainder.p, ctx);
        }
        fmpq_mpoly_swap(remaining.p, quotient.p, ctx);
        fmpz_add_ui(&multiplicity.v, &multiplicity.v, 1);
    }
    return decimal(multiplicity);
}

} // namespace atlas
