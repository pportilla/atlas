// Exact Macaulay elimination for small plane complete intersections.
#include "ideal_powers.hpp"
#include "groebner.hpp"
#include <flint/fmpz_poly.h>
#include <algorithm>

namespace atlas::ideal_powers {
std::optional<std::vector<Poly>> echelon_products(const Ring&,const std::vector<Poly>&,size_t,size_t);
namespace {
using LinearRow = std::vector<Z>;

void primitive_row(LinearRow& row, size_t first) {
    Z divisor;
    for (size_t j=first;j<row.size();++j) {
        fmpz_gcd(&divisor.v,&divisor.v,&row[j].v);
        if (fmpz_is_one(&divisor.v)) return;
    }
    if (!fmpz_is_zero(&divisor.v))
        for (size_t j=first;j<row.size();++j) fmpz_divexact(&row[j].v,&row[j].v,&divisor.v);
}

void clear_pivot(LinearRow& row, const LinearRow& pivot, size_t column, size_t first) {
    Z common,left,right;
    fmpz_gcd(&common.v,&row[column].v,&pivot[column].v);
    fmpz_divexact(&left.v,&pivot[column].v,&common.v);
    fmpz_divexact(&right.v,&row[column].v,&common.v);
    fmpz_zero(&row[column].v);
    for (size_t j=first;j<row.size();++j) if (j != column) {
        if (!fmpz_is_one(&left.v)) fmpz_mul(&row[j].v,&row[j].v,&left.v);
        fmpz_submul(&row[j].v,&pivot[j].v,&right.v);
    }
    primitive_row(row,first);
}
} // namespace

// For two degree-d forms with relatively prime highest homogeneous parts,
// the points at infinity are empty. R/I has length d², and
// length(R/I^n) = d²*n(n+1)/2 by the free conormal filtration. Every row
// below belongs to I^n; a zero-dimensional candidate initial ideal of this
// colength therefore certifies completeness without further pair reductions.
std::optional<std::vector<Poly>> plane_products(const Ring& ring, const std::vector<Poly>& inputs,
                                                  const std::vector<Poly>& products, size_t n) {
    if (ring.lex || ring.n != 2 || inputs.size()!=2 || n < 2) return std::nullopt;
    const size_t d = inputs[0].m.front();
    if (inputs[1].m.front()!=d || !d || (n+1)*d>24) return std::nullopt;
    // A nonzero binary homogeneous resultant is equivalent to coprimality.
    // Euclidean gcd over QQ of f(1,t),g(1,t) plus the possible root [0:1].
    fmpz_poly_t a,b,g;
    fmpz_poly_init(a); fmpz_poly_init(b); fmpz_poly_init(g);
    for (size_t i=0;i<2;++i) for (size_t j=0;j<inputs[i].size();++j) {
        auto m = inputs[i].m.data()+ring.stride*j;
        if (m[0] != d) break;
        fmpz_poly_set_coeff_fmpz(i ? b : a,Ring::field(m,ring.position(1)),&inputs[i].c[j].v);
    }
    bool coprime = std::max(fmpz_poly_degree(a),fmpz_poly_degree(b)) == static_cast<slong>(d);
    if (coprime) { fmpz_poly_gcd(g,a,b); coprime = fmpz_poly_degree(g)==0; }
    fmpz_poly_clear(a); fmpz_poly_clear(b); fmpz_poly_clear(g);
    if (!coprime) return std::nullopt;
    if (auto result=echelon_products(ring,products,d-1,d*d*n*(n+1)/2)) return result;
    const size_t top = (n+1)*d-1;
    const size_t columns = (top+1)*(top+2)/2;
    auto column = [columns](size_t degree,size_t x) { return columns-1-(degree*(degree+1)/2+x); };
    std::vector<LinearRow> echelon(columns);
    for (size_t extra=0;extra<d;++extra) for(size_t x=0;x<=extra;++x) for(const auto& p : products) {
        LinearRow row(columns);
        for (size_t j=0;j<p.size();++j) {
            auto m = p.m.data()+ring.stride*j;
            fmpz_set(&row[column(m[0]+extra,Ring::field(m,ring.position(0))+x)].v,&p.c[j].v);
        }
        for (size_t k=0;k<columns;++k) if (!fmpz_is_zero(&row[k].v)) {
            if (echelon[k].empty()) { echelon[k]=std::move(row); break; }
            clear_pivot(row,echelon[k],k,k+1);
        }
    }
    std::vector<size_t> needed;
    size_t length = 0;
    for (size_t degree=0;degree<=top;++degree) for(size_t x=0;x<=degree;++x) {
        const auto k=column(degree,x);
        if (echelon[k].empty()) { if (degree==top) return std::nullopt; ++length; continue; }
        if ((!x || echelon[column(degree-1,x-1)].empty()) &&
            (x==degree || echelon[column(degree-1,x)].empty())) needed.push_back(k);
    }
    if(length!=d*d*n*(n+1)/2) return std::nullopt;
    std::reverse(needed.begin(),needed.end());
    std::vector<Poly> basis;
    for(auto k:needed) {
        auto& row=echelon[k];
        for(size_t j=k+1;j<columns;++j) if(!fmpz_is_zero(&row[j].v) && !echelon[j].empty())
            clear_pivot(row,echelon[j],j,k);
        Poly p;
        for(int deg=top;deg>=0;--deg) for(int x=deg;x>=0;--x) {
            auto i=column(deg,x);
            if(fmpz_is_zero(&row[i].v)) continue;
            p.c.emplace_back(std::move(row[i]));
            long powers[2]={x,deg-x}; u64 m[MAX_STRIDE];ring.pack(m,powers);
            p.m.insert(p.m.end(),m,m+ring.stride);
        }
        basis.push_back(std::move(p));
    }
    return basis;
}
} // namespace atlas::ideal_powers
