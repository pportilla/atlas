// Compact fraction-free elimination for bounded Macaulay matrices. If an
// exact row operation overflows, the caller reuses the arbitrary-precision path.
#include "groebner.hpp"
#include <numeric>
#include <optional>
#include <set>

namespace atlas::ideal_powers {
namespace {
using Row = std::vector<long>;

bool clear(Row& row, const Row& pivot, size_t col, size_t first) {
    long a=row[col], b=pivot[col], divisor=std::gcd(a,b);
    a/=divisor; b/=divisor;
    row[col]=0;
    long content=0;
    for(size_t i=first;i<row.size();++i) if(i!=col) {
        long u,v;
        if(__builtin_mul_overflow(b,row[i],&u) || __builtin_mul_overflow(a,pivot[i],&v) ||
           __builtin_sub_overflow(u,v,&row[i]) || row[i]==LONG_MIN) return false;
        if(content!=1) content=std::gcd(content,row[i]);
    }
    if(content>1) for(size_t j=first;j<row.size();++j) row[j]/=content;
    return true;
}
} // namespace

// The certificate is supplied independently of row elimination: for a
// triangular change of coordinates the products already have the prescribed
// initial monomials; for a plane complete intersection the target is its length.
std::optional<std::vector<Poly>> echelon_products(const Ring& ring, const std::vector<Poly>& products,
                                                   size_t shifts, size_t target_length) {
    if(ring.n>4 || products.empty()) return std::nullopt;
    auto key = [&](const u64* m) {
        // One packed word stores grevlex exponents in reverse variable order.
        // Within a fixed total degree its complement sorts the terms.
        u64 k=0;
        for(int i=0;i<ring.n;++i) k=(k<<8) | Ring::field(m,i);
        return (m[0]<<40) | (ring.lex ? k : ((u64(1)<<(ring.n*8))-1-k));
    };
    struct Term {u64 key,exps;long coefficient;};
    size_t top=0;
    std::vector<std::vector<Term>> generators;
    std::vector<u64> keys;
    for(size_t j=0;j<=shifts;++j) for(size_t x=0;x<=j;++x) for(const auto& p : products) {
        std::vector<Term> row;
        long powers[MAX_VARIABLES]={}; powers[0]=x;if(ring.n>1)powers[1]=j-x;
        u64 shift[MAX_STRIDE];ring.pack(shift,powers);
        for(size_t i=0;i<p.size();++i) {
            u64 m[MAX_STRIDE];
            ring.multiply(m,p.m.data()+i*ring.stride,shift);
            if(m[0]>=255 || !fmpz_fits_si(&p.c[i].v)) return std::nullopt;
            long value=fmpz_get_si(&p.c[i].v);
            if(value==LONG_MIN) return std::nullopt;
            top=std::max(top,static_cast<size_t>(m[0]));
            auto k=key(m); row.push_back({k,m[1],value}); keys.push_back(k);
        }
        generators.push_back(std::move(row));
    }
    if (ring.lex || top>48 || keys.size()>12000) return std::nullopt;
    if(target_length) {
        for(size_t d=0;d<=top;++d) for(size_t x=0;x<=d;++x) {
            long powers[2]={static_cast<long>(x),static_cast<long>(d-x)};u64 m[MAX_STRIDE];ring.pack(m,powers);
            keys.push_back(key(m));
        }
    }
    std::sort(keys.begin(),keys.end(),std::greater<>());
    keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
    std::vector<Row> echelon(keys.size());
    std::vector<u64> exps(keys.size());
    for(const auto& terms:generators) {
        Row row(keys.size());
        for(auto t:terms) {
            size_t i=std::lower_bound(keys.begin(),keys.end(),t.key,std::greater<>())-keys.begin();
            row[i]=t.coefficient;exps[i]=t.exps;
        }
        for(size_t k=0;k<keys.size();++k) if(row[k]) {
            if(echelon[k].empty()) {echelon[k]=std::move(row);break;}
            if(!clear(row,echelon[k],k,k+1)) return std::nullopt;
        }
    }
    size_t missing=0;
    for(size_t i=0;i<keys.size();++i) if(echelon[i].empty()) {
        if(target_length && (keys[i]>>40)==top) return std::nullopt;
        ++missing;
    }
    if(target_length && missing!=target_length) return std::nullopt;
    std::vector<Poly> basis;
    for(size_t k=0;k<keys.size();++k) {
        auto& row=echelon[k];
        if(row.empty()) continue;
        bool minimal=true;
        for(size_t j=k+1;j<keys.size();++j) if(!echelon[j].empty() && (((exps[k]|HI)-exps[j])&HI)==HI) {
            minimal=false; break;
        }
        if(!minimal) continue;
        for(size_t j=k+1;j<keys.size();++j) if(row[j] && !echelon[j].empty())
            if(!clear(row,echelon[j],j,k)) return std::nullopt;
        // With no degree shifts all generators are images of equigenerated
        // monomials. Other terms of degree at most top can only be reduced
        // by a monomial of that ideal if they are themselves a pivot.
        Poly p;
        for(size_t j=k;j<keys.size();++j) if(row[j]) {
            p.m.push_back(keys[j]>>40);p.m.push_back(exps[j]);
            fmpz_set_si(&p.c.emplace_back().v,row[j]);
        }
        basis.push_back(std::move(p));
    }
    return basis;
}
} // namespace atlas::ideal_powers
