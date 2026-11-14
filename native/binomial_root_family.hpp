#pragma once

#include <flint/fexpr.h>
#include <flint/fmpq.h>
#include "algebraic_number.hpp"
#include <flint/acb.h>
#include <flint/fmpq_mat.h>
#include <flint/fmpq_poly.h>
#include <json/json.h>
#include <vector>

namespace atlas {
namespace binomial_field {
struct Matrix {
    fmpq_mat_t value;
    Matrix(slong rows, slong cols) { fmpq_mat_init(value,rows,cols); }
    ~Matrix() { fmpq_mat_clear(value); }
    Matrix(const Matrix&) = delete;
};

// For x^n=c, c != 0, the n values a*zeta^i are exactly its simple roots.
// An enclosure overlapping just one FLINT root enclosure certifies its
// index. Ambiguous enclosures merely defer to the general field routine.
inline std::vector<size_t> orbit_indices(const std::vector<AlgebraicNumber>& roots,
                                         size_t anchor, const AlgebraicNumber& unity) {
    acb_t current, step;
    acb_init(current);
    acb_init(step);
    acb_set(current,QQBAR_ENCLOSURE(roots[anchor].n));
    acb_set(step,QQBAR_ENCLOSURE(unity.n));
    std::vector<size_t> indices;
    std::vector<bool> used(roots.size());
    for (size_t i=0; i<roots.size(); ++i) {
        size_t match=roots.size();
        for (size_t j=0; j<roots.size(); ++j) {
            if (!acb_overlaps(current,QQBAR_ENCLOSURE(roots[j].n))) continue;
            if (match!=roots.size()) { match=roots.size(); break; }
            match=j;
        }
        if (match==roots.size() || used[match]) break;
        indices.push_back(match);
        used[match]=true;
        acb_mul(current,current,step,128);
    }
    acb_clear(step);
    acb_clear(current);
    return indices;
}
} // namespace binomial_field

// The caller supplies the FLINT-indexed roots and an initialized power-basis
// polynomial for each requested value. Mixed and partial families continue
// through the ordinary primitive-element method.
template<class Element>
bool binomial_root_family(const Json::Value& requests, const std::vector<AlgebraicNumber>& roots,
                          AlgebraicNumber& generator, std::vector<Element>& coordinates) {
    const auto& equation=requests[0u]["coefficients"];
    const size_t n=roots.size();
    if (n<3 || n+1!=equation.size() || equation[0u].asString()=="0" ||
        equation[equation.size()-1].asString()=="0") return false;
    for (Json::ArrayIndex i=1; i+1<equation.size(); ++i)
        if (equation[i].asString()!="0") return false;
    std::vector<bool> selected(n);
    size_t anchor=0;
    for (size_t i=0; i<n; ++i) {
        const auto& row=requests[static_cast<Json::ArrayIndex>(i)];
        const size_t index=row["index"].asUInt64()-1;
        if (row["coefficients"]!=equation || index>=n || selected[index] || roots[i].zero()) return false;
        selected[index]=true;
        // Choose the simplest available mark, independently of input order.
        if (qqbar_degree(roots[i].n)<qqbar_degree(roots[anchor].n) ||
            (qqbar_degree(roots[i].n)==qqbar_degree(roots[anchor].n) &&
             index<requests[static_cast<Json::ArrayIndex>(anchor)]["index"].asUInt64()-1)) anchor=i;
    }
    AlgebraicNumber unity;
    qqbar_root_of_unity(unity.n,1,n);
    const auto indices=binomial_field::orbit_indices(roots,anchor,unity);
    if (indices.size()!=n) return false;
    Element cyclotomic, shift, inverse, gcd, unused;
    fmpq_poly_set_fmpz_poly(cyclotomic.poly,QQBAR_POLY(unity.n));
    if (qqbar_is_rational(roots[anchor].n)) {
        // A rational anchor gives the splitting field Q(zeta) directly.
        fmpq_t a;
        fmpq_init(a);
        qqbar_get_fmpq(a,roots[anchor].n);
        fmpq_poly_set_fmpq(inverse.poly,a);
        fmpq_clear(a);
        for (size_t j : indices) {
            fmpq_poly_set(coordinates[j].poly,inverse.poly);
            fmpq_poly_shift_left(inverse.poly,inverse.poly,1);
            fmpq_poly_rem(inverse.poly,inverse.poly,cyclotomic.poly);
        }
        generator=std::move(unity);
        return true;
    }
    // Let theta=a*(2+zeta) and y=theta^n=c*(2+zeta)^n. If 1,y,...,y^(d-1)
    // span Q(zeta), one exact solve recovers zeta and 1/(2+zeta) in that
    // basis. Thus theta/(2+zeta) is the anchor and zeta generates its orbit.
    // In particular Q(theta) contains a and zeta, and is the splitting field.
    const slong d=qqbar_degree(unity.n);
    if (n>static_cast<size_t>((WORD_MAX-1)/d)) return false;
    fmpq_poly_set_coeff_si(shift.poly,0,2);
    fmpq_poly_set_coeff_si(shift.poly,1,1);
    fmpq_poly_xgcd(gcd.poly,inverse.poly,unused.poly,shift.poly,cyclotomic.poly);
    if (!fmpq_poly_is_one(gcd.poly)) return false;
    Element y, product, power;
    fmpq_poly_pow(y.poly,shift.poly,n);
    fmpq_poly_rem(y.poly,y.poly,cyclotomic.poly);
    fmpq_t c;
    fmpq_init(c);
    fmpz_set_str(fmpq_numref(c),equation[0u].asCString(),10);
    fmpz_set_str(fmpq_denref(c),equation[equation.size()-1].asCString(),10);
    if (fmpz_is_zero(fmpq_denref(c))) { fmpq_clear(c); return false; }
    fmpq_neg(c,c);
    fmpq_canonicalise(c);
    fmpq_poly_scalar_mul_fmpq(y.poly,y.poly,c);
    fmpq_clear(c);
    binomial_field::Matrix basis(d,d), targets(d,2), solution(d,2);
    fmpq_poly_one(power.poly);
    for (slong j=0; j<d; ++j) {
        for (slong i=0; i<d; ++i)
            fmpq_poly_get_coeff_fmpq(fmpq_mat_entry(basis.value,i,j),power.poly,i);
        fmpq_poly_mul(product.poly,power.poly,y.poly);
        fmpq_poly_rem(power.poly,product.poly,cyclotomic.poly);
    }
    // d=1 occurs for n=1,2, handled by the generic constructor above.
    fmpq_one(fmpq_mat_entry(targets.value,1,1));
    for (slong i=0; i<d; ++i)
        fmpq_poly_get_coeff_fmpq(fmpq_mat_entry(targets.value,i,0),inverse.poly,i);
    if (!fmpq_mat_solve(solution.value,basis.value,targets.value)) return false;
    AlgebraicNumber factor;
    qqbar_add_si(factor.n,unity.n,2);
    generator=roots[anchor]*factor;
    Element modulus;
    fmpq_poly_set_fmpz_poly(modulus.poly,QQBAR_POLY(generator.n));
    Element root_image, unity_image;
    for (slong i=0; i<d; ++i) {
        fmpq_poly_set_coeff_fmpq(root_image.poly,1+i*n,fmpq_mat_entry(solution.value,i,0));
        fmpq_poly_set_coeff_fmpq(unity_image.poly,i*n,fmpq_mat_entry(solution.value,i,1));
    }
    fmpq_poly_rem(root_image.poly,root_image.poly,modulus.poly);
    fmpq_poly_rem(unity_image.poly,unity_image.poly,modulus.poly);
    for (size_t j : indices) {
        fmpq_poly_set(coordinates[j].poly,root_image.poly);
        fmpq_poly_mul(product.poly,root_image.poly,unity_image.poly);
        fmpq_poly_rem(root_image.poly,product.poly,modulus.poly);
    }
    return true;
}
} // namespace atlas
